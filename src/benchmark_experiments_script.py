import pandas as pd
import numpy as np
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
import networkx as nx
import itertools
import math
import networkx as nx
import matplotlib.pyplot as plt
from same_decision_probability_calculation import *
from utils import *
import time

from monte_carlo_sdp import *
from pgmpy.utils import get_example_model
from ucimlrepo import fetch_ucirepo
import psutil
import tracemalloc
import gc
import threading
import os

def get_target(model):
    targets = {
        'child': 'Sick',
        'alarm': 'HYPOVOLEMIA',
        'barley': 'pesticid',
        'insurance': 'Theft',
        'hailfinder': 'ScenRelAMCIN',
        'hepar': 'hepatomegaly',
        'win95pts': 'PrtMem',
        'voting': 'Class',
        'chess': 'skach',
        'andes': 'NEED36',
        'link': 'N21_d_m',
        'pathfinder': 'F97'
    }

    return targets[model.name]

def run_for_time(func, *args, **kwargs):
    """Runs natively at maximum speed to record pure execution time."""
    start_time = time.time()
    try:
        result = func(*args, **kwargs)
        return result, (time.time() - start_time), True
    except Exception as e:
        return None, np.nan, False # Failed
    
def run_for_memory(func, *args, **kwargs):
    """
    Measures peak memory using BOTH tracemalloc (Python-level) and
    thread-sampled RSS (OS-level, catches numpy C allocations).
    Returns the max of both.
    """
    process = psutil.Process(os.getpid())

    gc.collect()
    baseline_rss = process.memory_info().rss
    peak_rss = [baseline_rss]
    stop_event = threading.Event()

    def sampler():
        while not stop_event.is_set():
            try:
                current = process.memory_info().rss
                if current > peak_rss[0]:
                    peak_rss[0] = current
            except Exception:
                pass
            time.sleep(0.001)  # 1ms poll

    sampler_thread = threading.Thread(target=sampler, daemon=True)
    sampler_thread.start()

    tracemalloc.start()
    try:
        func(*args, **kwargs)
    except MemoryError:
        pass
    except Exception as e:
        print(f"\n[!] Memory Tracker Warning: {func.__name__} failed with {type(e).__name__}: {e}")
    finally:
        stop_event.set()
        sampler_thread.join(timeout=2.0)

    _, python_peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    python_peak_mb = python_peak_bytes / (1024 * 1024)
    rss_delta_mb = (peak_rss[0] - baseline_rss) / (1024 * 1024)

    return (python_peak_mb, rss_delta_mb)

def run_targeted_sdp_experiment(output_csv="targeted_sdp_benchmark.csv", models_to_run=None):
    
    results = []
    raw_results = []
    H_RATIOS = [0.25, 0.50, 0.75, 0.9] 
    DECISION_THRESHOLD = 0.5
    TARGET_BUCKETS = [0.30, 0.50, 0.70, 0.9, 1.0]
    MCMC_TRIALS = 10 
    
    for bn in models_to_run:
        for H_RATIO in H_RATIOS:

            print(f"\n=== Starting experiments for BN: {bn.name} with H_RATIO: {H_RATIO} ===")

            n_nodes = bn.number_of_nodes()

            all_nodes = list(bn.nodes())
            
            target = get_target(bn)

            target_states = bn.get_cpds(target).state_names[target]
            target_value = target_states[1] if len(target_states) > 1 else target_states[0]
            print(f"Target Node: {target}, Target Value: {target_value}")
            
            available_nodes = [n for n in all_nodes if n != target]

            n_hidden = max(1, int(len(available_nodes) * H_RATIO))
             
            print(f"using {n_hidden} H variables")
            n_evidence = len(available_nodes) - n_hidden
            print(f"and {n_evidence} evidence variables")
           
            harvested_data = find_exact_experimental_patients_random(bn, target, target_value, DECISION_THRESHOLD,
                                                            n_evidence, buckets=TARGET_BUCKETS, batch_size=500, max_batches=1, max_partition_size=28)
            
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
                max_partition_size = max(len(p) for p in partitions)
                print(f"       -> Running Exact SDP...")
                
                # Pass 1: Time
                exact_sdp, exact_time, exact_success = run_for_time(
                    fast_broadcast_sdp, bn, target, target_value, patient, DECISION_THRESHOLD, partitions
                )
                
                # Pass 2: Memory
                exact_mem_mb_python, exact_mem_mb_rss = run_for_memory(
                    fast_broadcast_sdp, bn, target, target_value, patient, DECISION_THRESHOLD, partitions
                )
                
                if exact_success:
                    print(f"          Time: {exact_time:.4f} sec | Peak Memory: {exact_mem_mb_python:.2f} MB")
                else:
                    print(f"          [FAILED]: Crashed at {exact_mem_mb_python:.2f} MB")

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
                mcmc_mem_mb_python, mcmc_mem_mb_rss = run_for_memory(
                    fast_mcmc_sdp_estimation, bn, target, target_value, patient, DECISION_THRESHOLD,
                    n_samples=100, burn_in=50, thinning=5
                )
                
                print(f"          Avg Time: {mcmc_avg_time:.4f} sec | Peak Memory: {mcmc_mem_mb_python:.2f} MB")
                
                absolute_error = abs(exact_sdp - mcmc_mean)

                # ========================================================
                # PARALLEL TEMPERING MCMC EVALUATION
                # ========================================================

                #pt_mcmc_estimates = []
                #pt_mcmc_times = []
#
                #print(f"       -> Running Parallel Tempering MCMC SDP (Trials: {MCMC_TRIALS})...")              
                #
                ## Pass 1: Pure Time (across all trials)
                #for trial in range(MCMC_TRIALS):
                #    est_sdp, t_time, _ = run_for_time(
                #        pt_mcmc_sdp_estimation, bn, target, target_value, patient, DECISION_THRESHOLD,
                #        n_samples=1000, burn_in=2000, thinning=50, n_chains=4, max_temp=40.0
                #    )
                #    pt_mcmc_estimates.append(est_sdp)
                #    pt_mcmc_times.append(t_time)
#
                #pt_mcmc_mean = np.mean(pt_mcmc_estimates)
                #pt_mcmc_avg_time = np.mean(pt_mcmc_times)
                #pt_mcmc_variance = np.var(pt_mcmc_estimates)
#
                ## Pass 2: Peak Memory
                #pt_mcmc_mem_mb = run_for_memory(
                #    pt_mcmc_sdp_estimation, bn, target, target_value, patient, DECISION_THRESHOLD,
                #    n_samples=100, burn_in=50, thinning=5, n_chains=4, max_temp=10.0
                #)
#
                #print(f"          Avg Time: {pt_mcmc_avg_time:.4f} sec | Peak Memory: {pt_mcmc_mem_mb:.2f} MB")
#
                #absolute_error_pt = abs(exact_sdp - pt_mcmc_mean)
                
            
                # Record everything to the dataset
                results.append({
                    'Network': bn.name,
                    'N_Nodes': n_nodes,
                    'Target_Bucket': target_sdp,
                    'H_Ratio': H_RATIO,
                    'Target_Node': target,
                    'Target_Value': target_value,
                    'Exact_SDP': exact_sdp,
                    'Exact_Time_sec': exact_time,
                    'Exact_Mem_MB_Python': exact_mem_mb_python,
                    'Exact_Mem_MB_RSS': exact_mem_mb_rss,
                    'Max_Partition_Size': max_partition_size,
                    'MCMC_Mean_SDP': mcmc_mean,
                    'MCMC_Variance': mcmc_variance,
                    'MCMC_Avg_Time_sec': mcmc_avg_time,
                    'MCMC_Mem_MB_Python': mcmc_mem_mb_python,
                    'MCMC_Mem_MB_RSS': mcmc_mem_mb_rss,
                    'Absolute_Error': absolute_error,
                })
                
                # Save progressively
                pd.DataFrame(results).to_csv(output_csv, index=False)
                pd.DataFrame(raw_results).to_csv("raw_" + output_csv, index=False)

    print(f"\nExperiment Complete! Results saved to {output_csv}")
    return pd.DataFrame(results)


if __name__ == "__main__":



    # Model Loading
    print("Loading models...")
    alarm_model = get_example_model('alarm')
    child_model = get_example_model('child')
    insurance_model = get_example_model('insurance')
    hailfinder_model = get_example_model('hailfinder')
    hepar_model = get_example_model('hepar2')
    barley_model = get_example_model('barley')
    win95pts_model = get_example_model('win95pts')
    andes_model = get_example_model('andes')
    link_model = get_example_model('link')
    pathfinder_model = get_example_model('pathfinder')

    
    # ── VOTING ──────────────────────────────────────────────────────────────────
    voting = fetch_ucirepo(id=105)
    df_voting = pd.concat([voting.data.features, voting.data.targets], axis=1)
    df_voting.columns = [c.strip() for c in df_voting.columns]

    # Replace '?' missing values — Naive Bayes needs complete data
    df_voting = df_voting.replace('?', pd.NA).dropna()

    # All values must be strings/categories for pgmpy
    df_voting = df_voting.astype(str)

    target_voting = 'Class'   # 'democrat' / 'republican'

    voting_model = NaiveBayes()
    voting_model.fit(df_voting, target_voting,
                    estimator=MaximumLikelihoodEstimator)

    # ── CHESS ────────────────────────────────────────────────────────────────────
    chess = fetch_ucirepo(id=22)
    df_chess = pd.concat([chess.data.features, chess.data.targets], axis=1)
    df_chess = df_chess.astype(str)

    target_chess = 'skach' 

    chess_model = NaiveBayes()
    chess_model.fit(df_chess, target_chess,
                    estimator=MaximumLikelihoodEstimator)
    voting_model = BayesianNetwork(voting_model.edges())
    chess_model = BayesianNetwork(chess_model.edges())

    # fit
    voting_model.fit(df_voting, estimator=MaximumLikelihoodEstimator)
    chess_model.fit(df_chess, estimator=MaximumLikelihoodEstimator)

    child_model.name = 'child'
    insurance_model.name = 'insurance'
    alarm_model.name = 'alarm'
    hepar_model.name = 'hepar'
    hailfinder_model.name = 'hailfinder'
    win95pts_model.name = 'win95pts'
    barley_model.name = 'barley'
    voting_model.name = 'voting'
    chess_model.name = 'chess'
    andes_model.name = 'andes'
    link_model.name = 'link'
    pathfinder_model.name = 'pathfinder'

    # Ensure everymodel has a unique name
    models = [child_model, insurance_model, alarm_model, 
              hepar_model, hailfinder_model, win95pts_model, 
              barley_model, voting_model, chess_model, 
              andes_model, link_model, pathfinder_model]
    
    model_names = [model.name for model in models]

    assert len(set(model_names)) == len(models), "Model names must be unique!"

    # ensure all targets are present in the respective models
    for model in [child_model, alarm_model, barley_model, insurance_model, hailfinder_model, hepar_model, win95pts_model, voting_model, chess_model, andes_model, link_model, pathfinder_model]:
        print(f"Checking target node for model '{model.name}'...")
        target = get_target(model)
        if target not in model.nodes():
            raise ValueError(f"Target node '{target}' not found in model '{model.name}'")

    # Ensure all targets are binary
    for model in models:
        target = get_target(model)
        cpd = model.get_cpds(target)
        if len(cpd.state_names[target]) != 2:
            raise ValueError(f"Target node '{target}' in model '{model.name}' does not have exactly 2 states.")
    
    print("Models loaded successfully:")
    for model in models:
        print(f" - {model.name}")

    # Sort models by number of nodes (ascending)
    models.sort(key=lambda m: m.number_of_nodes())

    print("\nModel order for experiments (sorted by number of nodes):")
    for model in models:
        print(f" - {model.name} ({model.number_of_nodes()} nodes)")

    
    toy_models = models[:2]

    # Run the experiment
    #results_df_toy = run_targeted_sdp_experiment(output_csv="targeted_sdp_benchmark_toy.csv", models_to_run=toy_models)
    
    results_df_full = run_targeted_sdp_experiment(output_csv="targeted_sdp_benchmark_full.csv", models_to_run=models)




