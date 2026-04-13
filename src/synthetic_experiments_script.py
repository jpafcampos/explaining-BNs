import networkx as nx
import numpy as np
from bngenerator import *
from matplotlib import pyplot as plt
from pgmpy.readwrite.XMLBeliefNetwork import XBNReader, XBNWriter
from pgmpy.readwrite import BIFWriter, BIFReader
from pgmpy.models import BayesianModel
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination, ApproxInference, BeliefPropagation
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.estimators import BayesianEstimator
from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import BDeuScore, K2Score, BicScore
from pgmpy.metrics import structure_score
from pgmpy.utils import get_example_model
from pgmpy.estimators import ScoreCache
from pgmpy.inference.CausalInference import CausalInference
import random
import itertools
import numpy as np
import math
from same_decision_probability_calculation import *
from utils import *
from monte_carlo_sdp import *
import os
import glob
import time
import tracemalloc

def parse_bn_filename(filename):
    base = os.path.basename(filename).replace('.bif', '').replace('bn_', '')
    parts = base.split('_')
    n_nodes = int(parts[0].replace('n', ''))
    density = int(parts[1].replace('w', ''))
    # Take every remaining part as type_CPT, joined by underscores
    type_CPT = '_'.join(parts[2:])
    return n_nodes, density, type_CPT

def select_optimal_target_node(bn):
    """
    Selects a target node deeply embedded in the network (highest degree).
    Highly connected nodes provide a smooth, responsive probability gradient 
    for the Hill Climber, making extreme SDP values (0.90+) reachable.
    """
    best_node = None
    max_degree = -1
    
    for node in bn.nodes():
        # Ensure it's a binary node
        if len(bn.get_cpds(node).state_names[node]) != 2:
            continue
            
        degree = len(bn.get_parents(node)) + len(bn.get_children(node))
        if degree > max_degree:
            max_degree = degree
            best_node = node
            
    # Fallback if no binary nodes exist (rare)
    if best_node is None:
        return random.choice(list(bn.nodes()))
        
    return best_node



def run_for_time(func, *args, **kwargs):
    """Runs natively at maximum speed to record pure execution time."""
    start_time = time.time()
    try:
        result = func(*args, **kwargs)
        return result, (time.time() - start_time), True
    except Exception as e:
        return None, np.nan, False # Failed

def run_for_memory(func, *args, **kwargs):
    """Runs with tracemalloc to record peak memory. Ignores execution time."""
    tracemalloc.start()
    try:
        func(*args, **kwargs)
    except Exception:
        pass # We just want to see how high memory got before it crashed/finished
        
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak_mem / (1024 * 1024) # Return MB

def run_targeted_sdp_experiment(bif_directory, output_csv="targeted_sdp_random_bns.csv"):
    bif_files = glob.glob(os.path.join(bif_directory, "*.bif"))
    print(f"Found {len(bif_files)} BIF files in {bif_directory}")
    results = []
    
    H_RATIOS = [0.1, 0.3, 0.5, 0.70, 0.90] # Hidden variable ratios to test
    DECISION_THRESHOLD = 0.5
    TARGET_BUCKETS = [0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0]
    MCMC_TRIALS = 20
    
    for file in bif_files:
        n_nodes, density, rigidity = parse_bn_filename(file)
        print(f"\n========================================")
        print(f"Loading: {os.path.basename(file)}")
        
        bn = BIFReader(file).get_model()
        all_nodes = list(bn.nodes())
        
        target = select_optimal_target_node(bn)
        target_states = bn.get_cpds(target).state_names[target]
        target_value = target_states[1] if len(target_states) > 1 else target_states[0]
        
        available_nodes = [n for n in all_nodes if n != target]

        for H_RATIO in H_RATIOS:
            print(f"\n--- Hidden Ratio: {H_RATIO:.0%} ---")
            
            n_hidden = max(1, int(len(available_nodes) * H_RATIO))
            n_evidence = len(available_nodes) - n_hidden
            #evidence_vars = [n for n in available_nodes if n not in hidden_vars]
            
            # Run the Harvester 
            #harvested_data = harvest_patients_for_all_buckets(
            #    bn, target, target_value, DECISION_THRESHOLD, evidence_vars, TARGET_BUCKETS
            #)
            harvested_data = find_exact_experimental_patients_random(bn, target, target_value, DECISION_THRESHOLD,
                                                                    n_evidence, buckets=TARGET_BUCKETS, batch_size=10_000)
            
            # Now process whatever it managed to find
            for target_sdp, result in harvested_data.items():
                if result is None:
                    continue # We didn't find a patient for this specific bucket in this network
                    
                patient, exact_sdp = result
                hidden_vars = [n for n in bn.nodes() if n not in patient and n != target]
                print(f"\n  -> Benchmarking found patient for bucket {target_sdp} (Exact: {exact_sdp:.4f})")
                
                # ========================================================
                # EXACT SDP EVALUATION
                # ========================================================
                partitions = get_partitions(bn, hidden_vars, target, patient)
                print(f"       -> Running Exact SDP...")
                
                # Pass 1: Time
                exact_sdp, exact_time, exact_success = run_for_time(
                    fast_broadcast_sdp, bn, target, target_value, patient, DECISION_THRESHOLD, partitions
                )
                
                # Pass 2: Memory
                exact_mem_mb = run_for_memory(
                    fast_broadcast_sdp, bn, target, target_value, patient, DECISION_THRESHOLD, partitions
                )
                
                if exact_success:
                    print(f"          Time: {exact_time:.4f} sec | Peak Memory: {exact_mem_mb:.2f} MB")
                else:
                    print(f"          [FAILED]: Crashed at {exact_mem_mb:.2f} MB")

                # ========================================================
                # MCMC EVALUATION
                # ========================================================
                mcmc_estimates = []
                mcmc_times = []
                
                print(f"       -> Running MCMC SDP (Trials: {MCMC_TRIALS})...")
                
                # Pass 1: Pure Time (across all trials)
                for trial in range(MCMC_TRIALS):
                    est_sdp, t_time, _ = run_for_time(
                        fast_mcmc_sdp_estimation, bn, target, target_value, patient, DECISION_THRESHOLD,
                        n_samples=11000, burn_in=1000, thinning=10
                    )
                    mcmc_estimates.append(est_sdp)
                    mcmc_times.append(t_time)
                    
                mcmc_mean = np.mean(mcmc_estimates)
                mcmc_avg_time = np.mean(mcmc_times)
                mcmc_variance = np.var(mcmc_estimates)

                # Pass 2: Peak Memory
                mcmc_mem_mb = run_for_memory(
                    fast_mcmc_sdp_estimation, bn, target, target_value, patient, DECISION_THRESHOLD,
                    n_samples=1000, burn_in=50, thinning=5
                )
                
                print(f"          Avg Time: {mcmc_avg_time:.4f} sec | Peak Memory: {mcmc_mem_mb:.2f} MB")
                
                absolute_error = abs(exact_sdp - mcmc_mean)
                
                # Record everything to the dataset
                results.append({
                    'Network': os.path.basename(file),
                    'N_Nodes': n_nodes,
                    'Density': density,
                    'Rigidity': rigidity,
                    'Target_Bucket': target_sdp,
                    'Target_Node': target,
                    'Target_Value': target_value,
                    'Exact_SDP': exact_sdp,
                    'Exact_Time_sec': exact_time,
                    'MCMC_Mean_SDP': mcmc_mean,
                    'MCMC_Variance': mcmc_variance,
                    'MCMC_Avg_Time_sec': mcmc_avg_time,
                    'Absolute_Error': absolute_error
                })
                
                # Save progressively
                pd.DataFrame(results).to_csv(output_csv, index=False)

    print(f"\nExperiment Complete! Results saved to {output_csv}")
    return pd.DataFrame(results)


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='results/targeted_sdp_random_bns.csv')
    parser.add_argument('--bif-dir', type=str, default='./generated_bif_files')
    args = parser.parse_args()

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    run_targeted_sdp_experiment(args.bif_dir, output_csv=args.output)