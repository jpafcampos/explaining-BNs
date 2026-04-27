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
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pgmpy")
import random
random.seed(42)

MAX_TENSOR_SIZE_64 = 33_554_432   # = 2^25, 64gb RAM
MAX_TENSOR_SIZE_128 = 67_108_864   # = 2^26, 128gb RAM
MAX_TENSOR_SIZE_40 = 16_777_216  # = 2^24, ~40gb RAM 

#MAX_TENSOR_ALLOWED = MAX_TENSOR_SIZE_128  # Set this to the desired memory wall for the experiment
MAX_TENSOR_ALLOWED = 90_000_000
# =================================================================================
# / -------------------------- HELPER FUNCTIONS --------------------------
# =================================================================================

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
        print(f"\n[!] run_for_time: {func.__name__} failed with "
              f"{type(e).__name__}: {e}")
        return None, (time.time() - start_time), False
    
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
            #time.sleep(0.001)
            time.sleep(0.1) #CHANGED HERE TO REDUCE SAMPLING OVERHEAD 

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

def generate_random_patient(bn, target_node, n_evidence):
    all_nodes = list(bn.nodes())
    available_nodes = [n for n in all_nodes if n != target_node]
    evidence_vars = random.sample(available_nodes, min(n_evidence, len(available_nodes)))
    return {
        var: random.choice(bn.get_cpds(var).state_names[var])
        for var in evidence_vars
    }

# =================================================================================
# /--------------------------- HARVESTER FUNCTION WITH MEMORY CHECK --------------------------
# =================================================================================

def memory_aware_random_harvester(bn, target_node, target_value, decision_threshold,
                                            n_evidence,
                                            buckets=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                                            tolerance=0.05,
                                            batch_size=500,
                                            max_batches=1,
                                            max_tensor_entries=MAX_TENSOR_ALLOWED):
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
            if max_tensor >= max_tensor_entries:
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

# =================================================================================
#/--------------------------- FULL BENCHMARKING FUNCTION --------------------------
# =================================================================================

def run_targeted_sdp_experiment(output_csv="targeted_sdp_benchmark.csv", models_to_run=None, burn_in=5_000, thinning = 100,
                                 max_tensor_entries=MAX_TENSOR_ALLOWED):
    """
    Runs the full SDP benchmark across networks, H_RATIOs, and target buckets.
    
    Records every (network, H_RATIO, bucket) combination with one of these statuses:
      - OK: both exact SDP and MCMC ran successfully
      - BUCKET_NOT_FOUND: harvester was tractable but couldn't match this SDP range
      - SDP_INTRACTABLE_MCMC_ONLY: exact SDP hit the memory wall, MCMC still ran
      - INTRACTABLE: rare case where even MCMC couldn't run
    
    Parameters
    ----------
    max_tensor_entries : int
        Memory wall for exact SDP.
    """
    results = []
    H_RATIOS = [0.10, 0.25, 0.50, 0.75, 0.90]
    DECISION_THRESHOLD = 0.5
    TARGET_BUCKETS = [0.40, 0.50, 0.70, 0.8, 0.9, 1.0]
    TARGET_BUCKETS = [0.7, 1.0]
    MCMC_TRIALS = 10

    # Fixed schema — every row has these columns
    EMPTY_ROW = {
        'Network': None, 'N_Nodes': None, 'Target_Bucket': None, 'H_Ratio': None,
        'N_Hidden': None, 'Target_Node': None, 'Target_Value': None, 'Status': None,
        'Wall_Hits': None, 'Attempts_OK': None, 'SDP_Failures': None,
        'Exact_SDP': None, 'Exact_Time_sec': None,
        'Exact_Mem_MB_Python': None, 'Exact_Mem_MB_RSS': None,
        'Max_Partition_Size': None, 'Max_Tensor_Size': None,
        'MCMC_Mean_SDP': None, 'MCMC_Variance': None, 'MCMC_Avg_Time_sec': None,
        'MCMC_Mem_MB_Python': None, 'MCMC_Mem_MB_RSS': None,
        'Absolute_Error': None,
    }

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
            n_evidence = len(available_nodes) - n_hidden
            print(f"using {n_hidden} H variables and {n_evidence} evidence variables")

            # --- Run the harvester ---
            harvested = memory_aware_random_harvester(
                bn, target, target_value, DECISION_THRESHOLD,
                n_evidence,
                buckets=TARGET_BUCKETS,
                batch_size=600,
                max_batches=2,
                max_tensor_entries=max_tensor_entries,
            )
            harvested_data = harvested['buckets']
            intractable = harvested['attempts_ok'] == 0 and harvested['wall_hits'] > 0

            # ==========================================================
            # CASE 1 — Exact SDP intractable: MCMC-only benchmark
            # ==========================================================
            if intractable:
                print(f"\n[!] Network intractable for exact SDP at H_RATIO={H_RATIO}. "
                      f"Running MCMC-only benchmark.")

                patient = generate_random_patient(bn, target, n_evidence)
                hidden_vars = [n for n in bn.nodes() if n not in patient and n != target]

                mcmc_estimates = []
                mcmc_times = []
                mcmc_success = True

                print(f"       -> Running MCMC SDP (Trials: {MCMC_TRIALS})...")
                for trial in range(MCMC_TRIALS):
                    est_sdp, t_time, ok = run_for_time(
                        fast_mcmc_sdp_estimation_new, bn, target, target_value, patient,
                        DECISION_THRESHOLD, n_samples=3000, burn_in=burn_in, thinning=thinning, use_lw_seed=True
                    )
                    if not ok:
                        mcmc_success = False
                        break
                    mcmc_estimates.append(est_sdp)
                    mcmc_times.append(t_time)

                if mcmc_success and mcmc_estimates:
                    mcmc_mean = np.mean(mcmc_estimates)
                    mcmc_avg_time = np.mean(mcmc_times)
                    mcmc_variance = np.var(mcmc_estimates)

                    mcmc_mem_mb_python, mcmc_mem_mb_rss = run_for_memory(
                        fast_mcmc_sdp_estimation_new, bn, target, target_value, patient,
                        DECISION_THRESHOLD, n_samples=10, burn_in=0, thinning=5
                    )

                    print(f"          MCMC-only: {mcmc_mean:.4f} | Time: {mcmc_avg_time:.4f}s | "
                          f"Memory: {mcmc_mem_mb_python:.2f} MB")
                    status = 'SDP_INTRACTABLE_MCMC_ONLY'
                else:
                    print(f"          [!] MCMC also failed — fully intractable.")
                    mcmc_mean = None
                    mcmc_avg_time = None
                    mcmc_variance = None
                    mcmc_mem_mb_python = None
                    mcmc_mem_mb_rss = None
                    status = 'INTRACTABLE'

                # One row per bucket — same MCMC result across buckets
                for target_bucket in TARGET_BUCKETS:
                    row = dict(EMPTY_ROW)
                    row.update({
                        'Network': bn.name,
                        'N_Nodes': n_nodes,
                        'Target_Bucket': target_bucket,
                        'H_Ratio': H_RATIO,
                        'N_Hidden': len(hidden_vars),
                        'Target_Node': target,
                        'Target_Value': target_value,
                        'Status': status,
                        'Wall_Hits': harvested['wall_hits'],
                        'Attempts_OK': 0,
                        'SDP_Failures': harvested['sdp_failures'],
                        'MCMC_Mean_SDP': mcmc_mean,
                        'MCMC_Variance': mcmc_variance,
                        'MCMC_Avg_Time_sec': mcmc_avg_time,
                        'MCMC_Mem_MB_Python': mcmc_mem_mb_python,
                        'MCMC_Mem_MB_RSS': mcmc_mem_mb_rss,
                    })
                    results.append(row)

                pd.DataFrame(results).to_csv(output_csv, index=False)
                continue  # move to next H_RATIO

            # ==========================================================
            # CASE 2 — Normal path: process each bucket
            # ==========================================================
            for target_bucket in TARGET_BUCKETS:
                result = harvested_data.get(target_bucket)

                # Bucket not filled despite a tractable network
                if result is None:
                    row = dict(EMPTY_ROW)
                    row.update({
                        'Network': bn.name,
                        'N_Nodes': n_nodes,
                        'Target_Bucket': target_bucket,
                        'H_Ratio': H_RATIO,
                        'N_Hidden': n_hidden,
                        'Target_Node': target,
                        'Target_Value': target_value,
                        'Status': 'BUCKET_NOT_FOUND',
                        'Wall_Hits': harvested['wall_hits'],
                        'Attempts_OK': harvested['attempts_ok'],
                        'SDP_Failures': harvested['sdp_failures'],
                    })
                    results.append(row)
                    pd.DataFrame(results).to_csv(output_csv, index=False)
                    continue

                # Success path — full benchmark
                patient, exact_sdp = result
                hidden_vars = [n for n in bn.nodes() if n not in patient and n != target]
                partitions = get_partitions(bn, hidden_vars, target, patient)
                max_partition_size = max(len(p) for p in partitions)
                max_tensor_size = compute_max_tensor_size(bn, partitions)

                print(f"\n  -> Benchmarking patient for bucket {target_bucket} "
                      f"(Exact: {exact_sdp:.4f}, partition: {max_partition_size}, "
                      f"tensor: {max_tensor_size:,})")

                # ========================================================
                # EXACT SDP EVALUATION
                # ========================================================
                print(f"       -> Running Exact SDP...")

                # Pass 1: Time
                exact_sdp, exact_time, exact_success = run_for_time(
                    fast_broadcast_sdp, bn, target, target_value, patient,
                    DECISION_THRESHOLD, partitions
                )

                # Pass 2: Memory
                exact_mem_mb_python, exact_mem_mb_rss = run_for_memory(
                    fast_broadcast_sdp, bn, target, target_value, patient,
                    DECISION_THRESHOLD, partitions
                )

                if exact_success:
                    print(f"          Time: {exact_time:.4f} sec | "
                          f"Mem Python: {exact_mem_mb_python:.2f} MB | "
                          f"Mem RSS: {exact_mem_mb_rss:.2f} MB")
                else:
                    print(f"          [FAILED]: Crashed at {exact_mem_mb_python:.2f} MB")

                # ========================================================
                # MCMC EVALUATION
                # ========================================================
                mcmc_estimates = []
                mcmc_times = []
                print(f"       -> Running MCMC SDP (Trials: {MCMC_TRIALS})...")

                for trial in range(MCMC_TRIALS):
                    est_sdp, t_time, _ = run_for_time(
                        fast_mcmc_sdp_estimation_new, bn, target, target_value, patient,
                        DECISION_THRESHOLD, n_samples=3000, burn_in=burn_in, thinning=thinning, use_lw_seed = True
                    )
                    mcmc_estimates.append(est_sdp)
                    mcmc_times.append(t_time)

                mcmc_mean = np.mean(mcmc_estimates)
                mcmc_avg_time = np.mean(mcmc_times)
                mcmc_variance = np.var(mcmc_estimates)

                mcmc_mem_mb_python, mcmc_mem_mb_rss = run_for_memory(
                    fast_mcmc_sdp_estimation_new, bn, target, target_value, patient,
                    DECISION_THRESHOLD, n_samples=100, burn_in=50, thinning=5, use_lw_seed = False
                )

                print(f"          Avg Time: {mcmc_avg_time:.4f} sec | "
                      f"Estimated SDP: {mcmc_mean:.4f} |  "
                      f"Mem Python: {mcmc_mem_mb_python:.2f} MB | "
                      f"Mem RSS: {mcmc_mem_mb_rss:.2f} MB")

                absolute_error = abs(exact_sdp - mcmc_mean) if exact_success else None

                # Record the full result
                row = dict(EMPTY_ROW)
                row.update({
                    'Network': bn.name,
                    'N_Nodes': n_nodes,
                    'Target_Bucket': target_bucket,
                    'H_Ratio': H_RATIO,
                    'N_Hidden': len(hidden_vars),
                    'Target_Node': target,
                    'Target_Value': target_value,
                    'Status': 'OK' if exact_success else 'EXACT_CRASHED',
                    'Wall_Hits': harvested['wall_hits'],
                    'Attempts_OK': harvested['attempts_ok'],
                    'SDP_Failures': harvested['sdp_failures'],
                    'Exact_SDP': exact_sdp if exact_success else None,
                    'Exact_Time_sec': exact_time if exact_success else None,
                    'Exact_Mem_MB_Python': exact_mem_mb_python,
                    'Exact_Mem_MB_RSS': exact_mem_mb_rss,
                    'Max_Partition_Size': max_partition_size,
                    'Max_Tensor_Size': max_tensor_size,
                    'MCMC_Mean_SDP': mcmc_mean,
                    'MCMC_Variance': mcmc_variance,
                    'MCMC_Avg_Time_sec': mcmc_avg_time,
                    'MCMC_Mem_MB_Python': mcmc_mem_mb_python,
                    'MCMC_Mem_MB_RSS': mcmc_mem_mb_rss,
                    'Absolute_Error': absolute_error,
                })
                results.append(row)
                pd.DataFrame(results).to_csv(output_csv, index=False)

    print(f"\nExperiment Complete! Results saved to {output_csv}")
    return pd.DataFrame(results)


def run_large_network_experiment(output_csv="targeted_sdp_benchmark_large_networks.csv",
                                  models_to_run=None,
                                  max_tensor_entries=67_108_864,
                                  n_patients=10):
    """
    Experiment loop for large networks where bucket harvesting is impractical.
    Generates n_patients random patients per (network, H_RATIO) combination,
    computes exact SDP and MCMC, and records results in the same CSV format
    as run_targeted_sdp_experiment.

    Target_Bucket is filled with the nearest standard bucket value to the
    actual exact SDP, so results are compatible with the main experiment CSV.
    """
    STANDARD_BUCKETS = [0.3, 0.5, 0.7, 0.9, 1.0]
    H_RATIOS         = [0.10, 0.25, 0.50, 0.75, 0.9]
    DECISION_THRESHOLD = 0.5
    MCMC_TRIALS      = 10

    results = []

    EMPTY_ROW = {
        'Network': None, 'N_Nodes': None, 'Target_Bucket': None, 'H_Ratio': None,
        'N_Hidden': None, 'Target_Node': None, 'Target_Value': None, 'Status': None,
        'Wall_Hits': None, 'Attempts_OK': None, 'SDP_Failures': None,
        'Exact_SDP': None, 'Exact_Time_sec': None,
        'Exact_Mem_MB_Python': None, 'Exact_Mem_MB_RSS': None,
        'Max_Partition_Size': None, 'Max_Tensor_Size': None,
        'MCMC_Mean_SDP': None, 'MCMC_Variance': None, 'MCMC_Avg_Time_sec': None,
        'MCMC_Mem_MB_Python': None, 'MCMC_Mem_MB_RSS': None,
        'Absolute_Error': None,
    }

    def nearest_bucket(sdp_value):
        return min(STANDARD_BUCKETS, key=lambda b: abs(b - sdp_value))

    for bn in models_to_run:
        for H_RATIO in H_RATIOS:
            print(f"\n=== Large network experiment: {bn.name} | H_RATIO={H_RATIO} ===")

            n_nodes         = bn.number_of_nodes()
            all_nodes       = list(bn.nodes())
            target          = get_target(bn)
            target_states   = bn.get_cpds(target).state_names[target]
            target_value    = target_states[1] if len(target_states) > 1 else target_states[0]
            available_nodes = [n for n in all_nodes if n != target]
            n_hidden        = max(1, int(len(available_nodes) * H_RATIO))
            n_evidence      = len(available_nodes) - n_hidden

            print(f"Target: {target} = {target_value} | "
                  f"H={n_hidden} hidden | E={n_evidence} evidence")

            base_inference  = VariableElimination(bn)
            patients_found  = 0
            attempts        = 0
            wall_hits       = 0
            sdp_failures    = 0
            inference_failures = 0
            max_attempts    = n_patients * 100

            while patients_found < n_patients and attempts < max_attempts:
                attempts += 1

                # 1. Random patient
                evidence_vars = random.sample(available_nodes,
                                              min(n_evidence, len(available_nodes)))
                hidden_vars   = [n for n in all_nodes
                                 if n not in evidence_vars and n != target]
                patient       = {
                    var: random.choice(bn.get_cpds(var).state_names[var])
                    for var in evidence_vars
                }

                # 2. Memory wall check
                partitions = get_partitions(bn, hidden_vars, target, patient)
                if not partitions:
                    continue
                max_tensor = compute_max_tensor_size(bn, partitions)
                if max_tensor >= max_tensor_entries:
                    wall_hits += 1
                    continue

                # 3. Base decision must meet threshold
                try:
                    base_dist = base_inference.query(
                        variables=[target], evidence=patient, show_progress=False
                    )
                    if base_dist.get_value(**{target: target_value}) < DECISION_THRESHOLD:
                        continue
                except (ValueError, MemoryError) as e:
                    inference_failures += 1
                    continue

                # 4. Exact SDP — time pass
                max_partition_size = max(len(p) for p in partitions)
                ## Debug — check what partitions look like
                #print(f"    DEBUG: partitions={partitions}, n_parts={len(partitions)}")
                #partitions_test = get_partitions(bn, hidden_vars, target, patient)
                #print(f"    DEBUG: hidden_vars={hidden_vars}")
                #print(f"    DEBUG: patient keys={list(patient.keys())[:5]}...")
                exact_sdp, exact_time, exact_success = run_for_time(
                    fast_broadcast_sdp, bn, target, target_value, patient,
                    DECISION_THRESHOLD, partitions
                )
                if not exact_success:
                    sdp_failures += 1
                    continue

                patients_found += 1
                bucket = nearest_bucket(exact_sdp)

                print(f"  Patient {patients_found}/{n_patients} | "
                      f"exact_sdp={exact_sdp:.4f} → bucket={bucket} | "
                      f"partition={max_partition_size} | tensor={max_tensor:,} | "
                      f"time={exact_time:.4f}s")

                # 5. Exact SDP — memory pass (skip for large tensors)
                if max_tensor > max_tensor_entries:
                    exact_mem_mb_python, exact_mem_mb_rss = np.nan, np.nan
                else:
                    gc.collect()
                    exact_mem_mb_python, exact_mem_mb_rss = run_for_memory(
                        fast_broadcast_sdp, bn, target, target_value, patient,
                        DECISION_THRESHOLD, partitions
                    )

                # 6. MCMC — no LW seed for large networks
                print(f"       -> Running MCMC SDP (Trials: {MCMC_TRIALS})...")
                mcmc_estimates = []
                mcmc_times     = []
                for trial in range(MCMC_TRIALS):
                    est_sdp, t_time, _ = run_for_time(
                        fast_mcmc_sdp_estimation_new, bn, target, target_value, patient,
                        DECISION_THRESHOLD,
                        n_samples=1000, burn_in=5000, thinning=100,
                        use_lw_seed=False
                    )
                    mcmc_estimates.append(est_sdp)
                    mcmc_times.append(t_time)

                mcmc_mean     = np.mean(mcmc_estimates)
                mcmc_avg_time = np.mean(mcmc_times)
                mcmc_variance = np.var(mcmc_estimates)

                gc.collect()
                mcmc_mem_mb_python, mcmc_mem_mb_rss = run_for_memory(
                    fast_mcmc_sdp_estimation_new, bn, target, target_value, patient,
                    DECISION_THRESHOLD,
                    n_samples=100, burn_in=50, thinning=5,
                    use_lw_seed=False
                )

                absolute_error = abs(exact_sdp - mcmc_mean)
                print(f"          mcmc={mcmc_mean:.4f} | "
                      f"error={absolute_error:.4f} | "
                      f"time={mcmc_avg_time:.4f}s")

                # 7. Record result
                row = dict(EMPTY_ROW)
                row.update({
                    'Network':             bn.name,
                    'N_Nodes':             n_nodes,
                    'Target_Bucket':       bucket,
                    'H_Ratio':             H_RATIO,
                    'N_Hidden':            len(hidden_vars),
                    'Target_Node':         target,
                    'Target_Value':        target_value,
                    'Status':              'RANDOM_PATIENT',
                    'Wall_Hits':           wall_hits,
                    'Attempts_OK':         patients_found,
                    'SDP_Failures':        sdp_failures,
                    'Exact_SDP':           exact_sdp,
                    'Exact_Time_sec':      exact_time,
                    'Exact_Mem_MB_Python': exact_mem_mb_python,
                    'Exact_Mem_MB_RSS':    exact_mem_mb_rss,
                    'Max_Partition_Size':  max_partition_size,
                    'Max_Tensor_Size':     max_tensor,
                    'MCMC_Mean_SDP':       mcmc_mean,
                    'MCMC_Variance':       mcmc_variance,
                    'MCMC_Avg_Time_sec':   mcmc_avg_time,
                    'MCMC_Mem_MB_Python':  mcmc_mem_mb_python,
                    'MCMC_Mem_MB_RSS':     mcmc_mem_mb_rss,
                    'Absolute_Error':      absolute_error,
                })
                results.append(row)
                pd.DataFrame(results).to_csv(output_csv, index=False)

            # End of (bn, H_RATIO) loop
            if patients_found < n_patients:
                print(f"\n  Warning: only found {patients_found}/{n_patients} valid "
                      f"patients after {attempts} attempts "
                      f"(wall_hits={wall_hits})")

                # If no patients found at all, record a single intractable row
                if patients_found == 0:
                    for bucket in STANDARD_BUCKETS:
                        row = dict(EMPTY_ROW)
                        row.update({
                            'Network':      bn.name,
                            'N_Nodes':      n_nodes,
                            'Target_Bucket': bucket,
                            'H_Ratio':      H_RATIO,
                            'N_Hidden':     n_hidden,
                            'Target_Node':  target,
                            'Target_Value': target_value,
                            'Status':       'INTRACTABLE',
                            'Wall_Hits':    wall_hits,
                            'Attempts_OK':  0,
                            'SDP_Failures': sdp_failures,
                        })
                        results.append(row)
                    pd.DataFrame(results).to_csv(output_csv, index=False)

    print(f"\nLarge network experiment complete! Results saved to {output_csv}")
    return pd.DataFrame(results)

if __name__ == "__main__":

    alarm_model = get_example_model('alarm')
    alarm_model.name = 'alarm'
    alarm_debug = run_targeted_sdp_experiment(output_csv="alarm_debug.csv", models_to_run=[alarm_model])

    print(f"Starting Targeted SDP Benchmark Experiment using {MAX_TENSOR_ALLOWED} MB")

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
    
    models_to_run = [child_model, insurance_model, alarm_model, 
              hepar_model, hailfinder_model, 
              barley_model]
    
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
    models_to_run.sort(key=lambda m: m.number_of_nodes())

    print("\nModel order for experiments (sorted by number of nodes):")
    for model in models:
        print(f" - {model.name} ({model.number_of_nodes()} nodes)")

    
    toy_models = models[:2]

    # Run the experiment
    #results_df_toy = run_targeted_sdp_experiment(output_csv="targeted_sdp_benchmark_toy.csv", models_to_run=[child_model], max_tensor_entries=15_000_000)
    
    results_medium = run_targeted_sdp_experiment(output_csv="targeted_sdp_benchmark_full_isambard_medium_nets_seed_true.csv", models_to_run=[alarm_model])
    #results_large = run_large_network_experiment(
    #output_csv="targeted_sdp_benchmark_large_networks.csv",
    #models_to_run=[andes_model, link_model, pathfinder_model],
    #max_tensor_entries=MAX_TENSOR_SIZE_128,
    #n_patients=10
    #)
