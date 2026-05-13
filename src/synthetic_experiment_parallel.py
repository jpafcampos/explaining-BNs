import os
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
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
import glob
import time
import tracemalloc
import gc
import threading
from multiprocessing import Pool

MAX_TENSOR_ALLOWED = 90_000_000

def parse_bn_filename(filename):
    base = os.path.basename(filename).replace('.bif', '').replace('bn_', '')
    parts = base.split('_')
    n_nodes = int(parts[0].replace('n', ''))
    density = int(parts[1].replace('w', ''))
    # Take every remaining part as type_CPT, joined by underscores
    type_CPT = '_'.join(parts[2:])
    return n_nodes, density, type_CPT

def estimate_exact_inference_memory(bn):
    """
    Simulates Variable Elimination using the Min-Degree heuristic to find the 
    maximum clique size (Treewidth + 1). Returns the estimated RAM required in GB.
    """
    # 1. Create undirected skeleton
    G = bn.to_undirected()
    
    # 2. Moralize the graph (marry all parents)
    for node in bn.nodes():
        parents = list(bn.get_parents(node))
        for i in range(len(parents)):
            for j in range(i+1, len(parents)):
                G.add_edge(parents[i], parents[j])
                
    # 3. Simulate elimination to find the largest tensor (clique)
    max_clique_size = 0
    nodes = list(G.nodes())
    
    while nodes:
        # Find node with minimum degree
        degrees = dict(G.degree(nodes))
        min_node = min(degrees, key=degrees.get)
        
        # Calculate the size of the tensor formed by this node and its neighbors
        neighbors = list(G.neighbors(min_node))
        clique_size = len(neighbors) + 1 # +1 for the node itself
        
        if clique_size > max_clique_size:
            max_clique_size = clique_size
            
        # Connect all neighbors to each other (fill-in edges)
        for i in range(len(neighbors)):
            for j in range(i+1, len(neighbors)):
                G.add_edge(neighbors[i], neighbors[j])
                
        # Remove the node
        G.remove_node(min_node)
        nodes.remove(min_node)
        
    # Calculate Memory: 2^W states * 8 bytes per float
    estimated_bytes = (2 ** max_clique_size) * 8
    estimated_gb = estimated_bytes / (1024 ** 3)
    
    return max_clique_size, estimated_gb


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
        print(f"\n[!] run_for_time: {func.__name__} failed with "
              f"{type(e).__name__}: {e}")
        return None, (time.time() - start_time), False
    
def run_for_memory(func, *args, **kwargs):
    """
    Measures peak memory using BOTH tracemalloc (Python-level) and
    thread-sampled RSS (OS-level, catches numpy/C allocations).
    Both are measured as deltas from baseline.
    Returns (python_peak_mb, rss_peak_mb).
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
            time.sleep(0.01)

    tracemalloc.start()
    baseline_traced, _ = tracemalloc.get_traced_memory()  # snapshot before func

    sampler_thread = threading.Thread(target=sampler, daemon=True)
    sampler_thread.start()

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

    python_peak_mb = (python_peak_bytes - baseline_traced) / (1024 * 1024)
    rss_peak_mb = (peak_rss[0] - baseline_rss) / (1024 * 1024)

    return python_peak_mb, rss_peak_mb

from math import prod
def compute_tensor_size(bn, partition):
    return prod(len(bn.get_cpds(v).state_names[v]) for v in partition)


def compute_max_tensor_size(bn, partitions):
    if not partitions:
        return 0
    return max(compute_tensor_size(bn, p) for p in partitions)

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


import psutil
def process_single_file(args):
    """Process one BIF file and return results as a list of dicts."""
     # Tell joblib/pgmpy to stay single-threaded inside this worker
    os.environ["LOKY_MAX_CPU_COUNT"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    def log_mem(label):
        rss = process.memory_info().rss / (1024 ** 3)
        print(f"[PID {os.getpid()}] {label} | RSS: {rss:.2f} GB", flush=True)

    file, H_RATIOS, DECISION_THRESHOLD, TARGET_BUCKETS, SIZES_TO_RUN, DENSITIES_TO_RUN, MCMC_TRIALS = args
    results = []
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 ** 3)
    print(f"[PID {os.getpid()}] START {os.path.basename(file)} | RSS: {mem_before:.2f} GB", flush=True)

    n_nodes, density, rigidity = parse_bn_filename(file)
    if n_nodes not in SIZES_TO_RUN:
        return [] 
    
    if density not in DENSITIES_TO_RUN:
        return []

    print(f"\n========================================")
    print(f"Loading: {os.path.basename(file)}")
    
    bn = BIFReader(file).get_model()
    log_mem(f"After loading BN: {os.path.basename(file)}")
    all_nodes = list(bn.nodes())

    
    target = select_optimal_target_node(bn)
    target_states = bn.get_cpds(target).state_names[target]
    target_value = target_states[1] if len(target_states) > 1 else target_states[0]
    
    available_nodes = [n for n in all_nodes if n != target]

    for H_RATIO in H_RATIOS:
        print(f"\n--- Hidden Ratio: {H_RATIO:.0%} ---")
        
        n_hidden = max(1, int(len(available_nodes) * H_RATIO))

        # can not process more than 100 hidden variables
        if n_nodes == 200 and n_hidden > 100:
            continue

        n_evidence = len(available_nodes) - n_hidden
        #evidence_vars = [n for n in available_nodes if n not in hidden_vars]
        
        # Run the Harvester 
        #harvested_data = harvest_patients_for_all_buckets(
        #    bn, target, target_value, DECISION_THRESHOLD, evidence_vars, TARGET_BUCKETS
        #)
        gc.collect() # Clean up before the harvest, which can be memory-intensive
        #harvested_data = find_exact_experimental_patients_random(bn, target, target_value, DECISION_THRESHOLD,
        #                                                        n_evidence, buckets=TARGET_BUCKETS, batch_size=2_000)
        harvested = memory_aware_random_harvester(
            bn, target, target_value, DECISION_THRESHOLD,
            n_evidence,
            buckets=TARGET_BUCKETS,
            batch_size=400,
            max_batches=2,
            max_tensor_entries=MAX_TENSOR_ALLOWED,
        )
        harvested_data = harvested['buckets']
        
        gc.collect() # Clean up after the harvest, which can be memory-intensive
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
            print(f"       -> Biggest partition size: {max(len(p) for p in partitions)} hidden variables")
            print(f"       -> Running Exact SDP...")
            
            # Pass 1: Time
            exact_sdp, exact_time, exact_success = run_for_time(
                fast_broadcast_sdp, bn, target, target_value, patient, DECISION_THRESHOLD, partitions
            )
            #log_mem(f"After Exact SDP Time Test: {os.path.basename(file)}")
            
            # Pass 2: Memory
            gc.collect() # Clean up before the memory test
            exact_mem_mb_python, exact_mem_mb_rss = run_for_memory(
                fast_broadcast_sdp, bn, target, target_value, patient, DECISION_THRESHOLD, partitions
            )
            #log_mem(f"After Exact SDP Memory Test: {os.path.basename(file)}")
            gc.collect() # Clean up after the memory test

            if exact_success:
                print(f"          Time: {exact_time:.4f} sec | Peak Memory: {exact_mem_mb_python:.2f} MB (Python) / {exact_mem_mb_rss:.2f} MB (RSS)")
            else:
                print(f"          [FAILED]: Crashed at {exact_mem_mb_python:.2f} MB (Python) / {exact_mem_mb_rss:.2f} MB (RSS)")

            # ========================================================
            # MCMC EVALUATION
            # ========================================================
            mcmc_estimates = []
            mcmc_times = []
            
            print(f"       -> Running MCMC SDP (Trials: {MCMC_TRIALS})...")
            
            # Pass 1: Pure Time (across all trials)
            for trial in range(MCMC_TRIALS):
                est_sdp, t_time, _ = run_for_time(
                    fast_mcmc_sdp_estimation_new, bn, target, target_value, patient, DECISION_THRESHOLD,
                    n_samples=1000, burn_in=5000, thinning=100, use_lw_seed = True
                )
                mcmc_estimates.append(est_sdp)
                mcmc_times.append(t_time)
                #log_mem(f"After MCMC Trial {trial+1}/{MCMC_TRIALS} Time Test: {os.path.basename(file)}")
                
            #log_mem(f"After MCMC Time Test: {os.path.basename(file)}")
            mcmc_mean = np.mean(mcmc_estimates)
            mcmc_avg_time = np.mean(mcmc_times)
            mcmc_variance = np.var(mcmc_estimates)
            print(f"               -> Mean MCMC: {mcmc_mean}")

            # Pass 2: Peak Memory
            gc.collect() # Clean up before the memory test
            mcmc_mem_mb_python, mcmc_mem_mb_rss = run_for_memory(
                fast_mcmc_sdp_estimation_new, bn, target, target_value, patient, DECISION_THRESHOLD,
                n_samples=10, burn_in=5, thinning=5
            )
            #log_mem(f"After MCMC Memory Test: {os.path.basename(file)}")
            gc.collect() # Clean up after the memory test

            print(f"          Avg Time: {mcmc_avg_time:.4f} sec | Peak Memory: {mcmc_mem_mb_python:.2f} MB (Python) / {mcmc_mem_mb_rss:.2f} MB (RSS)")

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
                    vectorized_pt_mcmc_sdp_estimation, bn, target, target_value, patient, DECISION_THRESHOLD,
                    n_samples=1000, burn_in=5000, thinning=100, n_chains=4, max_temp=40.0, use_ancestral_seed = True
                )
                pt_mcmc_estimates.append(est_sdp)
                pt_mcmc_times.append(t_time)
                #log_mem(f"After PT-MCMC Trial {trial+1}/{MCMC_TRIALS} Time Test: {os.path.basename(file)}")
            
            #log_mem(f"After PT-MCMC Time Test: {os.path.basename(file)}")
            pt_mcmc_mean = np.mean(pt_mcmc_estimates)
            pt_mcmc_avg_time = np.mean(pt_mcmc_times)
            pt_mcmc_variance = np.var(pt_mcmc_estimates)
            print(f"               -> Mean PT: {pt_mcmc_mean}")
            # Pass 2: Peak Memory
            gc.collect() # Clean up before the memory test
            pt_mcmc_mem_mb_python, pt_mcmc_mem_mb_rss = run_for_memory(
                vectorized_pt_mcmc_sdp_estimation, bn, target, target_value, patient, DECISION_THRESHOLD,
                n_samples=10, burn_in=5, thinning=5, n_chains=4, max_temp=10.0
            )
            #log_mem(f"After PT-MCMC Memory Test: {os.path.basename(file)}")
            gc.collect() # Clean up after the memory test
            print(f"          Avg Time: {pt_mcmc_avg_time:.4f} sec | Peak Memory: {pt_mcmc_mem_mb_python:.2f} MB (Python) / {pt_mcmc_mem_mb_rss:.2f} MB (RSS)")

            absolute_error_pt = abs(exact_sdp - pt_mcmc_mean)
            
        
            # Record everything to the dataset
            results.append({
                'Network': os.path.basename(file),
                'N_Nodes': n_nodes,
                'Density': density,
                'Rigidity': rigidity,
                'Hidden_Ratio': H_RATIO,
                'Target_Bucket': target_sdp,
                'Target_Node': target,
                'Target_Value': target_value,
                'Exact_SDP': exact_sdp,
                'Exact_Time_sec': exact_time,
                'Exact_Peak_Memory_MB_Python': exact_mem_mb_python,
                'Exact_Peak_Memory_MB_RSS': exact_mem_mb_rss,
                'MCMC_Mean_SDP': mcmc_mean,
                'MCMC_Variance': mcmc_variance,
                'MCMC_Avg_Time_sec': mcmc_avg_time,
                'MCMC_Peak_Memory_MB_Python': mcmc_mem_mb_python,
                'MCMC_Peak_Memory_MB_RSS': mcmc_mem_mb_rss,
                'Absolute_Error': absolute_error,
                'PT_MCMC_Mean_SDP': pt_mcmc_mean,
                'PT_MCMC_Variance': pt_mcmc_variance,
                'PT_MCMC_Avg_Time_sec': pt_mcmc_avg_time,
                'PT_Peak_Memory_MB_Python': pt_mcmc_mem_mb_python,
                'PT_Peak_Memory_MB_RSS': pt_mcmc_mem_mb_rss,
                'PT_Absolute_Error': absolute_error_pt
            })
    
    mem_after = process.memory_info().rss / (1024 ** 3)
    print(f"[PID {os.getpid()}] END {os.path.basename(file)} | RSS: {mem_after:.2f} GB | Delta: {mem_after - mem_before:.2f} GB", flush=True)
    
    return results

def run_targeted_sdp_experiment(bif_directory, output_csv="targeted_sdp_random_bns.csv", n_workers=4):
    bif_files = sorted(glob.glob(os.path.join(bif_directory, "*.bif")))
    
    H_RATIOS = [0.25, 0.50] # Hidden variable ratios to test
    DECISION_THRESHOLD = 0.5
    TARGET_BUCKETS = [0.6, 0.8]
    MCMC_TRIALS = 10

    SIZES_TO_RUN = [20, 50]
    DENSITIES_TO_RUN = [2, 6]

    # Build args list for each file
    args_list = [
        (file, H_RATIOS, DECISION_THRESHOLD, TARGET_BUCKETS, SIZES_TO_RUN, DENSITIES_TO_RUN, MCMC_TRIALS)
        for file in bif_files
    ]

    all_results = []

    files_done = 0
    with Pool(processes=n_workers) as pool:
        for file_results in pool.imap_unordered(process_single_file, args_list):
            files_done += 1
            if file_results:
                all_results.extend(file_results)
                pd.DataFrame(all_results).to_csv(output_csv, index=False)
                print(f"[{files_done}/{len(bif_files)}] Checkpoint saved — {len(all_results)} total rows")
            else:
                print(f"[{files_done}/{len(bif_files)}] No results for this file")


    #pd.DataFrame(all_results).to_csv(output_csv, index=False)
    print(f"\nExperiment Complete! Results saved to {output_csv}")
    return pd.DataFrame(all_results)



import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bif-dir', type=str, default='./generated_bif_files/')
    parser.add_argument('--output', type=str, default='results/parallel_output.csv')
    parser.add_argument('--n-workers', type=int, default=1)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    run_targeted_sdp_experiment(args.bif_dir, args.output, args.n_workers)