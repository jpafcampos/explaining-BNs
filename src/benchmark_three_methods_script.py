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
MAX_TENSOR_ALLOWED = 130_000_000
# =================================================================================
# / -------------------------- HELPER FUNCTIONS --------------------------
# =================================================================================

def estimate_ve_cost_min_fill(bn, evidence, target):
    """
    Estimates the maximum factor (tensor) size during Variable Elimination
    by simulating the elimination process using the Min-Fill heuristic.
    """
    elim_vars = set(v for v in bn.nodes() if v != target and v not in evidence)
    
    # Initialize the moral graph and get cardinalities
    moral = bn.to_markov_model() 
    cards = {n: len(bn.get_cpds(n).state_names[n]) for n in bn.nodes()}
    
    max_tensor_size = 1
    
    while elim_vars:
        # --- MIN-FILL HEURISTIC ---
        best_var = None
        min_fill_count = float('inf')
        best_tensor_size = float('inf') # Tie-breaker
        
        for v in elim_vars:
            neighbors = list(moral.neighbors(v))
            fill_count = 0
            
            # Count how many edges are missing between neighbors
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    if not moral.has_edge(neighbors[i], neighbors[j]):
                        fill_count += 1
            
            # Tie-breaker: If two nodes create the same number of fill edges,
            # eliminate the one that generates the smaller tensor.
            if fill_count < min_fill_count:
                min_fill_count = fill_count
                best_var = v
                best_tensor_size = prod(cards[n] for n in neighbors + [v])
            elif fill_count == min_fill_count:
                current_tensor = prod(cards[n] for n in neighbors + [v])
                if current_tensor < best_tensor_size:
                    best_var = v
                    best_tensor_size = current_tensor
                    
        # --- ELIMINATION ---
        neighbors = list(moral.neighbors(best_var))
        
        # 1. Update our max tensor size tracker
        max_tensor_size = max(max_tensor_size, best_tensor_size)
        
        # 2. Add the fill-in edges to the moral graph
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                if not moral.has_edge(neighbors[i], neighbors[j]):
                    moral.add_edge(neighbors[i], neighbors[j])
                    
        # 3. Remove the node
        moral.remove_node(best_var)
        elim_vars.remove(best_var)
        
    return max_tensor_size

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
            #print("-> Testing memory wall...")
            max_tensor = compute_max_tensor_size(bn, partitions)
            if max_tensor >= max_tensor_entries:
                wall_hits += 1
                continue

            max_tensor_ve = estimate_ve_cost_min_fill(bn, temp_patient, target_node)
            if max_tensor_ve >= max_tensor_entries:
                wall_hits += 1
                continue

            # 4. Check base decision meets the threshold
            try:
                #print("-> Trying base distribution query...")
                #print(f"--> Estimated Max Tensor VE: {max_tensor_ve}")
                base_dist = base_inference.query(
                    variables=[target_node], evidence=temp_patient, elimination_order='MinFill', show_progress=False
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

def run_3_method_targeted_sdp(output_csv="three_method_sdp_benchmark.csv",
                               models_to_run=None,
                               burn_in=5_000,
                               thinning=100,
                               max_tensor_entries=MAX_TENSOR_ALLOWED,
                               pt_n_chains=4,
                               pt_max_temp=40.0):
    """
    Three-method SDP benchmark: exact, plain MCMC (LW-seeded), and
    parallel tempering. Same structure as run_targeted_sdp_experiment,
    extended with PT columns.
 
    For every (network, H_RATIO, bucket) combination, records one of:
      - OK                         : all three methods ran successfully
      - BUCKET_NOT_FOUND           : harvester tractable but no bucket match
      - SDP_INTRACTABLE_MCMC_ONLY  : exact unavailable, both samplers ran
      - INTRACTABLE                : even the samplers failed
      - EXACT_CRASHED              : exact crashed, samplers may still have run
 
    Parameters
    ----------
    pt_n_chains : int
        Number of chains in the PT ladder.
    pt_max_temp : float
        Hottest temperature (cold = 1.0, ladder is geometric).
    """
    results = []
    H_RATIOS = [0.75, 0.90]
    DECISION_THRESHOLD = 0.5
    TARGET_BUCKETS = [0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
    MCMC_TRIALS = 10
 
    # Fixed schema — every row has these columns
    EMPTY_ROW = {
        'Network': None, 'N_Nodes': None, 'Target_Bucket': None, 'H_Ratio': None,
        'N_Hidden': None, 'Target_Node': None, 'Target_Value': None, 'Status': None,
        'Wall_Hits': None, 'Attempts_OK': None, 'SDP_Failures': None,
 
        # Exact
        'Exact_SDP': None, 'Exact_Time_sec': None,
        'Exact_Mem_MB_Python': None, 'Exact_Mem_MB_RSS': None,
        'Max_Partition_Size': None, 'Max_Tensor_Size': None,
 
        # Plain MCMC (LW-seeded, single chain)
        'MCMC_Mean_SDP': None, 'MCMC_Variance': None, 'MCMC_Avg_Time_sec': None,
        'MCMC_Mem_MB_Python': None, 'MCMC_Mem_MB_RSS': None,
        'Absolute_Error': None,
 
        # Parallel tempering
        'PT_Mean_SDP': None, 'PT_Variance': None, 'PT_Avg_Time_sec': None,
        'PT_Mem_MB_Python': None, 'PT_Mem_MB_RSS': None,
        'PT_Absolute_Error': None,
        'PT_N_Chains': None, 'PT_Max_Temp': None,
    }
 
    # ──────────────────────────────────────────────────────────────────
    # Local helpers — keep the main loop readable
    # ──────────────────────────────────────────────────────────────────
    def benchmark_plain_mcmc(bn, target, target_value, patient):
        """Run MCMC_TRIALS times and measure memory once. Returns dict."""
        ests, times = [], []
        success = True
        for _ in range(MCMC_TRIALS):
            est, t, ok = run_for_time(
                fast_mcmc_sdp_estimation_new, bn, target, target_value, patient,
                DECISION_THRESHOLD, n_samples=1000, burn_in=burn_in,
                thinning=thinning, use_lw_seed=False
            )
            if not ok:
                success = False
                break
            ests.append(est)
            times.append(t)
 
        if not success or not ests:
            return {'mean': None, 'var': None, 'avg_time': None,
                    'mem_py': None, 'mem_rss': None, 'success': False}
 
        mem_py, mem_rss = run_for_memory(
            fast_mcmc_sdp_estimation_new, bn, target, target_value, patient,
            DECISION_THRESHOLD, n_samples=10, burn_in=5, thinning=5,
            use_lw_seed=False
        )
        return {
            'mean': float(np.mean(ests)),
            'var':  float(np.var(ests)),
            'avg_time': float(np.mean(times)),
            'mem_py': mem_py, 'mem_rss': mem_rss,
            'success': True,
        }
 
    def benchmark_pt(bn, target, target_value, patient):
        """Run MCMC_TRIALS times and measure memory once. Returns dict."""
        ests, times = [], []
        success = True
        for _ in range(MCMC_TRIALS):
            est, t, ok = run_for_time(
                vectorized_pt_mcmc_sdp_estimation, bn, target, target_value, patient,
                DECISION_THRESHOLD, n_samples=1000, burn_in=burn_in,
                thinning=thinning, n_chains=pt_n_chains, max_temp=pt_max_temp,
                use_ancestral_seed=False
            )
            if not ok:
                success = False
                break
            ests.append(est)
            times.append(t)
 
        if not success or not ests:
            return {'mean': None, 'var': None, 'avg_time': None,
                    'mem_py': None, 'mem_rss': None, 'success': False}
 
        mem_py, mem_rss = run_for_memory(
            vectorized_pt_mcmc_sdp_estimation, bn, target, target_value, patient,
            DECISION_THRESHOLD, n_samples=10, burn_in=5, thinning=5,
            n_chains=pt_n_chains, max_temp=pt_max_temp, use_ancestral_seed=True
        )
        return {
            'mean': float(np.mean(ests)),
            'var':  float(np.var(ests)),
            'avg_time': float(np.mean(times)),
            'mem_py': mem_py, 'mem_rss': mem_rss,
            'success': True,
        }
 
    # ──────────────────────────────────────────────────────────────────
    # Main loop
    # ──────────────────────────────────────────────────────────────────
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
                batch_size=1000,
                max_batches=2,
                max_tensor_entries=max_tensor_entries,
            )
            harvested_data = harvested['buckets']
            intractable = harvested['attempts_ok'] == 0 and harvested['wall_hits'] > 0
 
            # ==========================================================
            # CASE 1 — Exact SDP intractable: both samplers benchmark
            # ==========================================================
            if intractable:
                print(f"\n[!] Network intractable for exact SDP at H_RATIO={H_RATIO}. "
                      f"Running sampler-only benchmark (MCMC + PT).")
 
                patient = generate_random_patient(bn, target, n_evidence)
                hidden_vars = [n for n in bn.nodes() if n not in patient and n != target]
 
                # Plain MCMC
                print(f"       -> Running plain MCMC (Trials: {MCMC_TRIALS})...")
                mcmc = benchmark_plain_mcmc(bn, target, target_value, patient)
                if mcmc['success']:
                    print(f"          MCMC: {mcmc['mean']:.4f} | "
                          f"Time: {mcmc['avg_time']:.4f}s | "
                          f"Mem: {mcmc['mem_py']:.2f} MB")
                else:
                    print(f"          [!] MCMC failed.")
 
                # Parallel tempering
                print(f"       -> Running PT MCMC (Trials: {MCMC_TRIALS}, "
                      f"n_chains={pt_n_chains}, max_temp={pt_max_temp})...")
                pt = benchmark_pt(bn, target, target_value, patient)
                if pt['success']:
                    print(f"          PT  : {pt['mean']:.4f} | "
                          f"Time: {pt['avg_time']:.4f}s | "
                          f"Mem: {pt['mem_py']:.2f} MB")
                else:
                    print(f"          [!] PT failed.")
 
                if mcmc['success'] or pt['success']:
                    status = 'SDP_INTRACTABLE_MCMC_ONLY'
                else:
                    print(f"          [!] Both samplers failed — fully intractable.")
                    status = 'INTRACTABLE'
 
                # One row per bucket — same sampler results across buckets
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
 
                        'MCMC_Mean_SDP':       mcmc['mean'],
                        'MCMC_Variance':       mcmc['var'],
                        'MCMC_Avg_Time_sec':   mcmc['avg_time'],
                        'MCMC_Mem_MB_Python':  mcmc['mem_py'],
                        'MCMC_Mem_MB_RSS':     mcmc['mem_rss'],
 
                        'PT_Mean_SDP':         pt['mean'],
                        'PT_Variance':         pt['var'],
                        'PT_Avg_Time_sec':     pt['avg_time'],
                        'PT_Mem_MB_Python':    pt['mem_py'],
                        'PT_Mem_MB_RSS':       pt['mem_rss'],
                        'PT_N_Chains':         pt_n_chains,
                        'PT_Max_Temp':         pt_max_temp,
                    })
                    results.append(row)
 
                pd.DataFrame(results).to_csv(output_csv, index=False)
                continue  # next H_RATIO
 
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
 
                # ────────── EXACT ──────────
                print(f"       -> Running Exact SDP...")
                exact_sdp, exact_time, exact_success = run_for_time(
                    fast_broadcast_sdp, bn, target, target_value, patient,
                    DECISION_THRESHOLD, partitions
                )
                exact_mem_mb_python, exact_mem_mb_rss = run_for_memory(
                    fast_broadcast_sdp, bn, target, target_value, patient,
                    DECISION_THRESHOLD, partitions
                )
                if exact_success:
                    print(f"          Time: {exact_time:.4f}s | "
                          f"Mem Python: {exact_mem_mb_python:.2f} MB | "
                          f"Mem RSS: {exact_mem_mb_rss:.2f} MB")
                else:
                    print(f"          [FAILED]: Crashed at {exact_mem_mb_python:.2f} MB")
 
                # ────────── PLAIN MCMC ──────────
                print(f"       -> Running plain MCMC (Trials: {MCMC_TRIALS})...")
                mcmc = benchmark_plain_mcmc(bn, target, target_value, patient)
                if mcmc['success']:
                    print(f"          MCMC: {mcmc['mean']:.4f} | "
                          f"Time: {mcmc['avg_time']:.4f}s | "
                          f"Mem: {mcmc['mem_py']:.2f} MB")
                else:
                    print(f"          [!] MCMC failed.")
 
                # ────────── PARALLEL TEMPERING ──────────
                print(f"       -> Running PT MCMC (Trials: {MCMC_TRIALS}, "
                      f"n_chains={pt_n_chains}, max_temp={pt_max_temp})...")
                pt = benchmark_pt(bn, target, target_value, patient)
                if pt['success']:
                    print(f"          PT  : {pt['mean']:.4f} | "
                          f"Time: {pt['avg_time']:.4f}s | "
                          f"Mem: {pt['mem_py']:.2f} MB")
                else:
                    print(f"          [!] PT failed.")
 
                print(patient)
 
                absolute_error    = abs(exact_sdp - mcmc['mean']) \
                                    if exact_success and mcmc['success'] else None
                pt_absolute_error = abs(exact_sdp - pt['mean']) \
                                    if exact_success and pt['success'] else None
 
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
                    'Wall_Hits':    harvested['wall_hits'],
                    'Attempts_OK':  harvested['attempts_ok'],
                    'SDP_Failures': harvested['sdp_failures'],
 
                    'Exact_SDP':           exact_sdp if exact_success else None,
                    'Exact_Time_sec':      exact_time if exact_success else None,
                    'Exact_Mem_MB_Python': exact_mem_mb_python,
                    'Exact_Mem_MB_RSS':    exact_mem_mb_rss,
                    'Max_Partition_Size':  max_partition_size,
                    'Max_Tensor_Size':     max_tensor_size,
 
                    'MCMC_Mean_SDP':       mcmc['mean'],
                    'MCMC_Variance':       mcmc['var'],
                    'MCMC_Avg_Time_sec':   mcmc['avg_time'],
                    'MCMC_Mem_MB_Python':  mcmc['mem_py'],
                    'MCMC_Mem_MB_RSS':     mcmc['mem_rss'],
                    'Absolute_Error':      absolute_error,
 
                    'PT_Mean_SDP':         pt['mean'],
                    'PT_Variance':         pt['var'],
                    'PT_Avg_Time_sec':     pt['avg_time'],
                    'PT_Mem_MB_Python':    pt['mem_py'],
                    'PT_Mem_MB_RSS':       pt['mem_rss'],
                    'PT_Absolute_Error':   pt_absolute_error,
                    'PT_N_Chains':         pt_n_chains,
                    'PT_Max_Temp':         pt_max_temp,
                })
                results.append(row)
                pd.DataFrame(results).to_csv(output_csv, index=False)
 
    print(f"\nExperiment Complete! Results saved to {output_csv}")
    return pd.DataFrame(results)


if __name__ == "__main__":

    #alarm_model = get_example_model('alarm')
    #alarm_model.name = 'alarm'
    #alarm_debug = run_targeted_sdp_experiment(output_csv="alarm_debug.csv", models_to_run=[alarm_model])
    
    #win95pts_model = get_example_model('win95pts')
    #win95pts_model.name = 'win95pts'
    #win95pts_debug = run_targeted_sdp_experiment(output_csv="win95pts_debug.csv", models_to_run=[win95pts_model])

    #hailfinder_model = get_example_model('hailfinder')
    #hailfinder_model.name = 'hailfinder'
    #hailfinder_model_debug = run_3_method_targeted_sdp(output_csv="debug.csv", models_to_run=[hailfinder_model])

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
    
    models_to_run = [andes_model, link_model, pathfinder_model]
    
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
    #models_to_run.sort(key=lambda m: m.number_of_nodes())

    print("\nModel order for experiments (sorted by number of nodes):")
    for model in models:
        print(f" - {model.name} ({model.number_of_nodes()} nodes)")

    
    toy_models = models[:2]

    results = run_3_method_targeted_sdp(output_csv="targeted_sdp_benchmark_all_methods_BIG_MEMORY.csv", models_to_run=models_to_run)
