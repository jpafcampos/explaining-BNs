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
    Selects a binary target node that is well-embedded as a CHILD in the
    network — i.e. has parents whose states can shift its posterior.

    A node with no parents has a fixed prior: evidence can never update it,
    the decision boundary never moves, and interesting SDP values are
    unreachable. We therefore require at least one parent and rank by
    n_parents first, n_children second.
    """
    best_node   = None
    best_score  = (-1, -1)   # (n_parents, n_children)

    for node in bn.nodes():
        cpd = bn.get_cpds(node)

        # Must be binary
        if len(cpd.state_names[node]) != 2:
            continue

        n_parents  = len(bn.get_parents(node))
        n_children = len(list(bn.get_children(node)))

        # Hard requirement: must have at least one parent so that evidence
        # can influence its posterior through the network
        if n_parents == 0:
            continue

        score = (n_parents, n_children)
        if score > best_score:
            best_score = score
            best_node  = node

    # Fallback: if every binary node is a root (unusual), relax the
    # parent requirement and just pick highest total degree
    if best_node is None:
        print("Warning: no binary node with parents found — "
              "falling back to highest-degree binary node.")
        for node in bn.nodes():
            if len(bn.get_cpds(node).state_names[node]) != 2:
                continue
            n_parents  = len(bn.get_parents(node))
            n_children = len(list(bn.get_children(node)))
            score = (n_parents + n_children, n_parents)
            if score > best_score:
                best_score = score
                best_node  = node

    if best_node is None:
        best_node = random.choice(list(bn.nodes()))
        print(f"Warning: using random fallback target: {best_node}")

    n_pa = len(bn.get_parents(best_node))
    n_ch = len(list(bn.get_children(best_node)))
    print(f"Selected target: {best_node}  "
          f"(parents={n_pa}, children={n_ch})")
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

def run_targeted_sdp_experiment(bif_directory, output_csv="targeted_sdp_random_bns.csv", job_idx=0, n_jobs=1):
    bif_files = sorted(glob.glob(os.path.join(bif_directory, "*.bif")))
    # Interleave: job 0 gets files 0, 50, 100...
    #             job 1 gets files 1, 51, 101... etc.
    bif_files = bif_files[job_idx::n_jobs]
    print(f"Job {job_idx} processing {len(bif_files)} files...")
    results = []
    
    H_RATIOS = [0.25] # Hidden variable ratios to test
    DECISION_THRESHOLD = 0.5
    TARGET_BUCKETS = [0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0]
    MCMC_TRIALS = 10
    
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
                        n_samples=1000, burn_in=2000, thinning=50
                    )
                    mcmc_estimates.append(est_sdp)
                    mcmc_times.append(t_time)
                    
                mcmc_mean = np.mean(mcmc_estimates)
                mcmc_avg_time = np.mean(mcmc_times)
                mcmc_variance = np.var(mcmc_estimates)

                # Pass 2: Peak Memory
                mcmc_mem_mb = run_for_memory(
                    fast_mcmc_sdp_estimation, bn, target, target_value, patient, DECISION_THRESHOLD,
                    n_samples=100, burn_in=50, thinning=5
                )
                
                print(f"          Avg Time: {mcmc_avg_time:.4f} sec | Peak Memory: {mcmc_mem_mb:.2f} MB")
                
                absolute_error = abs(exact_sdp - mcmc_mean)

                # ========================================================
                # PARALLEL TEMPERING MCMC EVALUATION
                # ========================================================

                pt_mcmc_estimates = []
                pt_mcmc_times = []

                print(f"       -> Running Parallel Tempering MCMC SDP (Trials: {MCMC_TRIALS})...")              
                
                # Pass 1: Pure Time (across all trials)
                for trial in range(MCMC_TRIALS):
                    est_sdp, t_time, _ = run_for_time(
                        pt_mcmc_sdp_estimation, bn, target, target_value, patient, DECISION_THRESHOLD,
                        n_samples=1000, burn_in=2000, thinning=50, n_chains=4, max_temp=40.0
                    )
                    pt_mcmc_estimates.append(est_sdp)
                    pt_mcmc_times.append(t_time)

                pt_mcmc_mean = np.mean(pt_mcmc_estimates)
                pt_mcmc_avg_time = np.mean(pt_mcmc_times)
                pt_mcmc_variance = np.var(pt_mcmc_estimates)

                # Pass 2: Peak Memory
                pt_mcmc_mem_mb = run_for_memory(
                    pt_mcmc_sdp_estimation, bn, target, target_value, patient, DECISION_THRESHOLD,
                    n_samples=100, burn_in=50, thinning=5, n_chains=4, max_temp=10.0
                )

                print(f"          Avg Time: {pt_mcmc_avg_time:.4f} sec | Peak Memory: {pt_mcmc_mem_mb:.2f} MB")

                absolute_error_pt = abs(exact_sdp - pt_mcmc_mean)
                
            
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
                    'Absolute_Error': absolute_error,
                    'PT_MCMC_Mean_SDP': pt_mcmc_mean,
                    'PT_MCMC_Variance': pt_mcmc_variance,
                    'PT_MCMC_Avg_Time_sec': pt_mcmc_avg_time,
                    'PT_Absolute_Error': absolute_error_pt
                })
                
                # Save progressively
                pd.DataFrame(results).to_csv(output_csv, index=False)

    print(f"\nExperiment Complete! Results saved to {output_csv}")
    return pd.DataFrame(results)


def run_targeted_sdp_experiment_toy(bif_directory, output_csv="targeted_sdp_random_bns.csv"):
    bif_files = glob.glob(os.path.join(bif_directory, "*.bif"))
    results = []
    
    H_RATIOS = [0.25] # Hidden variable ratios to test
    DECISION_THRESHOLD = 0.5
    TARGET_BUCKETS = [0.4, 0.5, 0.6, 0.70, 0.80, 0.9, 1.0]
    MCMC_TRIALS = 2
    
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
                        n_samples=50, burn_in=2, thinning=2
                    )
                    mcmc_estimates.append(est_sdp)
                    mcmc_times.append(t_time)
                    
                mcmc_mean = np.mean(mcmc_estimates)
                mcmc_avg_time = np.mean(mcmc_times)
                mcmc_variance = np.var(mcmc_estimates)

                # Pass 2: Peak Memory
                mcmc_mem_mb = run_for_memory(
                    fast_mcmc_sdp_estimation, bn, target, target_value, patient, DECISION_THRESHOLD,
                    n_samples=10, burn_in=5, thinning=2
                )
                
                print(f"          Avg Time: {mcmc_avg_time:.4f} sec | Peak Memory: {mcmc_mem_mb:.2f} MB")
                
                absolute_error = abs(exact_sdp - mcmc_mean)

                # ========================================================
                # PARALLEL TEMPERING MCMC EVALUATION
                # ========================================================

                pt_mcmc_estimates = []
                pt_mcmc_times = []

                print(f"       -> Running Parallel Tempering MCMC SDP (Trials: {MCMC_TRIALS})...")              
                
                # Pass 1: Pure Time (across all trials)
                for trial in range(MCMC_TRIALS):
                    est_sdp, t_time, _ = run_for_time(
                        pt_mcmc_sdp_estimation, bn, target, target_value, patient, DECISION_THRESHOLD,
                        n_samples=50, burn_in=2, thinning=2, n_chains=4, max_temp=40.0
                    )
                    pt_mcmc_estimates.append(est_sdp)
                    pt_mcmc_times.append(t_time)

                pt_mcmc_mean = np.mean(pt_mcmc_estimates)
                pt_mcmc_avg_time = np.mean(pt_mcmc_times)
                pt_mcmc_variance = np.var(pt_mcmc_estimates)

                # Pass 2: Peak Memory
                pt_mcmc_mem_mb = run_for_memory(
                    pt_mcmc_sdp_estimation, bn, target, target_value, patient, DECISION_THRESHOLD,
                    n_samples=10, burn_in=5, thinning=2, n_chains=4, max_temp=10.0
                )

                print(f"          Avg Time: {pt_mcmc_avg_time:.4f} sec | Peak Memory: {pt_mcmc_mem_mb:.2f} MB")

                absolute_error_pt = abs(exact_sdp - pt_mcmc_mean)
                
            
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
                    'Absolute_Error': absolute_error,
                    'PT_MCMC_Mean_SDP': pt_mcmc_mean,
                    'PT_MCMC_Variance': pt_mcmc_variance,
                    'PT_MCMC_Avg_Time_sec': pt_mcmc_avg_time,
                    'PT_Absolute_Error': absolute_error_pt
                })
                
                # Save progressively
                pd.DataFrame(results).to_csv(output_csv, index=False)

    print(f"\nExperiment Complete! Results saved to {output_csv}")
    return pd.DataFrame(results)

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--toy', action='store_true', help='Run the toy experiment with small BNs')
    parser.add_argument('--output', type=str, default='results/targeted_sdp_random_bns.csv')
    parser.add_argument('--job-idx', type=int, default=0)
    parser.add_argument('--n-jobs', type=int, default=1)
    
    args = parser.parse_args()

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    if parser.parse_args().toy:
        print("Running toy experiment with small BNs...")
        run_targeted_sdp_experiment_toy(bif_directory='./toy_experiment', output_csv='results/toy_targeted_sdp_small_bns.csv')
        exit(0)

    else:
        print("Running full experiment with all BNs...")
        run_targeted_sdp_experiment(bif_directory='./generated_bif_files', output_csv=args.output, job_idx=args.job_idx, n_jobs=args.n_jobs)