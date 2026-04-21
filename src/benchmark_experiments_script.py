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

from math import prod
import gc
import random
from pgmpy.inference import VariableElimination


def compute_tensor_size(bn, partition):
    return prod(len(bn.get_cpds(v).state_names[v]) for v in partition)


def compute_max_tensor_size(bn, partitions):
    if not partitions:
        return 0
    return max(compute_tensor_size(bn, p) for p in partitions)


def memory_aware_random_harvester(bn, target_node, target_value, decision_threshold,
                                            n_evidence,
                                            buckets=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                                            tolerance=0.05,
                                            batch_size=500,
                                            max_batches=1,
                                            max_tensor_entries=500_000_000):
    """
    -> Same logic as function find_exact_experimental_patients_random, 
    but with an explicit memory wall check
    
    Random-evidence harvester that fills target SDP buckets with example patients.
    
    Uses tensor size (product of hidden-variable cardinalities) as the memory
    safety metric, so it works correctly on networks with mixed cardinalities
    like Child (not just binary synthetic BNs).
    
    Parameters
    ----------
    max_tensor_entries : int
        Maximum number of entries allowed in any single tensor during SDP.
        Peak memory is roughly max_tensor_entries * 8 bytes * ~6 copies.
        500M ≈ 24 GB peak, safe for 64 GB budget.
        1B  ≈ 48 GB peak, safe for 128 GB budget.
    
    Returns
    -------
    dict with keys:
        'buckets'       : {bucket_value: (patient_dict, exact_sdp) | None}
        'wall_hits'     : int — patients rejected for exceeding the memory wall
        'attempts_ok'   : int — patients that passed the wall and got evaluated
        'sdp_failures'  : int — patients that passed the wall but crashed in SDP
        'inference_failures' : int — patients whose base inference crashed
    """
    all_nodes = list(bn.nodes())
    available_nodes = [n for n in all_nodes if n != target_node]

    unfilled_buckets = {b: None for b in buckets}
    wall_hits = 0
    attempts_ok = 0
    sdp_failures = 0
    inference_failures = 0

    print(f"\nHunting for patients... (Locking {n_evidence} variables as evidence)")
    print(f"Memory wall: tensor size ≤ {max_tensor_entries:,} entries "
          f"(~{max_tensor_entries * 8 / 1024**3:.1f} GB raw)")

    base_inference = VariableElimination(bn)
    batch_count = 0

    while any(v is None for v in unfilled_buckets.values()) and batch_count < max_batches:
        batch_count += 1
        print(f"Generating batch {batch_count}/{max_batches} of {batch_size} random realities...")

        for i in range(batch_size):
            # 1. Generate a random patient with randomly-sampled evidence
            evidence_vars = random.sample(available_nodes, min(n_evidence, len(available_nodes)))
            hidden_vars = [n for n in all_nodes if n not in evidence_vars and n != target_node]

            temp_patient = {
                var: random.choice(bn.get_cpds(var).state_names[var])
                for var in evidence_vars
            }

            # 2. Build partitions for this patient
            partitions = get_partitions(bn, hidden_vars, target_node, temp_patient)

            if not partitions:
                continue  # nothing to compute

            # 3. MEMORY SAFETY — reject if any tensor exceeds the wall
            max_tensor = compute_max_tensor_size(bn, partitions)
            if max_tensor > max_tensor_entries:
                wall_hits += 1
                continue

            # 4. Check base decision meets the threshold
            try:
                base_dist = base_inference.query(
                    variables=[target_node], evidence=temp_patient, show_progress=False
                )
                if base_dist.get_value(**{target_node: target_value}) < decision_threshold:
                    continue  # legitimate rejection, not a failure
            except (ValueError, MemoryError) as e:
                inference_failures += 1
                print(f"    [!] Base inference failed ({type(e).__name__}): {e}")
                gc.collect()
                continue  # try another patient rather than bailing out

            # 5. Calculate exact SDP
            try:
                exact_sdp = fast_broadcast_sdp(
                    bn, target_node, target_value, temp_patient,
                    decision_threshold, partitions
                )
            except (ValueError, MemoryError) as e:
                sdp_failures += 1
                print(f"    [!] SDP failed despite passing wall ({type(e).__name__}): {e}")
                gc.collect()
                continue  # try another patient

            attempts_ok += 1

            # 6. Try to fit this SDP into an empty bucket
            empty_targets = [b for b, v in unfilled_buckets.items() if v is None]
            for b in empty_targets:
                if abs(exact_sdp - b) <= tolerance:
                    unfilled_buckets[b] = (temp_patient.copy(), exact_sdp)
                    print(f"--> Filled bucket {b} with Exact SDP: {exact_sdp:.4f} "
                          f"(tensor size: {max_tensor:,})")
                    break

            # 7. Early exit if all buckets filled
            if not any(v is None for v in unfilled_buckets.values()):
                break

    # Summary
    missing = [b for b, v in unfilled_buckets.items() if v is None]
    filled = len(buckets) - len(missing)

    print(f"\nHarvest summary:")
    print(f"  Buckets filled:      {filled}/{len(buckets)}")
    if missing:
        print(f"  Missing buckets:     {missing}")
    print(f"  Wall hits:           {wall_hits} (tensor too big)")
    print(f"  Attempts past wall:  {attempts_ok}")
    print(f"  SDP failures:        {sdp_failures}")
    print(f"  Inference failures:  {inference_failures}")

    # Diagnose intractable cases
    if filled == 0 and wall_hits > 0 and attempts_ok == 0:
        print(f"  -> INTRACTABLE: every random patient exceeded the memory wall")

    return {
        'buckets': unfilled_buckets,
        'wall_hits': wall_hits,
        'attempts_ok': attempts_ok,
        'sdp_failures': sdp_failures,
        'inference_failures': inference_failures,
    }

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

    

    child_model.name = 'child'
    insurance_model.name = 'insurance'
    alarm_model.name = 'alarm'
    hepar_model.name = 'hepar'
    hailfinder_model.name = 'hailfinder'
    win95pts_model.name = 'win95pts'
    barley_model.name = 'barley'
    andes_model.name = 'andes'
    link_model.name = 'link'
    pathfinder_model.name = 'pathfinder'

    # Ensure everymodel has a unique name
    models = [child_model, insurance_model, alarm_model, 
              hepar_model, hailfinder_model, win95pts_model, 
              barley_model, 
              andes_model, link_model, pathfinder_model]
    
    model_names = [model.name for model in models]

    assert len(set(model_names)) == len(models), "Model names must be unique!"

    # ensure all targets are present in the respective models
    for model in [child_model, alarm_model, barley_model, insurance_model, hailfinder_model, hepar_model, win95pts_model, andes_model, link_model, pathfinder_model]:
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




