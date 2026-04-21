# benchmark_memory.py
import tracemalloc
import gc
from pgmpy.readwrite import BIFReader

from same_decision_probability_calculation import *
from monte_carlo_sdp import fast_mcmc_sdp_estimation

import random
import time 

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



def benchmark_hidden_vars(bif_file, n_trials=20, n_hidden=80):
    bn = BIFReader(bif_file).get_model()
    all_nodes = list(bn.nodes())
    target = select_optimal_target_node(bn)
    target_states = bn.get_cpds(target).state_names[target]
    target_value = target_states[1] if len(target_states) > 1 else target_states[0]
    available_nodes = [n for n in all_nodes if n != target]

    print(f"Nodes: {len(all_nodes)} | Running {n_trials} trials with {n_hidden} random hidden vars each\n")

    times = []
    memories = []

    for trial in range(n_trials):
        # Mirror exactly what the harvester does
        evidence_vars = random.sample(available_nodes, len(available_nodes) - n_hidden)
        hidden_vars = [n for n in all_nodes if n not in evidence_vars and n != target]
        patient = {var: random.choice(bn.get_cpds(var).state_names[var]) 
                   for var in evidence_vars}

        gc.collect()
        tracemalloc.start()

        try:
            partitions = get_partitions(bn, hidden_vars, target, patient)

            start = time.perf_counter()
            result = fast_broadcast_sdp(bn, target, target_value, patient, 0.5, partitions)
            elapsed = time.perf_counter() - start

            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            peak_mb = peak / 1024 / 1024
            times.append(elapsed)
            memories.append(peak_mb)

            print(f"Trial {trial+1:3d}/{n_trials} | "
                  f"Hidden edges: {bn.subgraph(hidden_vars).number_of_edges():3d} | "
                  f"Time: {elapsed:.4f} sec | "
                  f"Peak memory: {peak_mb:.2f} MB | OK")

        except MemoryError:
            tracemalloc.stop()
            print(f"Trial {trial+1:3d}/{n_trials} | OUT OF MEMORY")
        except Exception as e:
            tracemalloc.stop()
            print(f"Trial {trial+1:3d}/{n_trials} | ERROR: {e}")

    if times:
        print(f"\n--- Summary ---")
        print(f"Time    — min: {min(times):.4f}s | max: {max(times):.4f}s | avg: {sum(times)/len(times):.4f}s")
        print(f"Memory  — min: {min(memories):.2f}MB | max: {max(memories):.2f}MB | avg: {sum(memories)/len(memories):.2f}MB")
        print(f"Hidden edges — this tells us if low memory correlates with sparse hidden subgraphs")

import pandas as pd



def build_growing_partition_benchmark(bn, target, evidence_vars_pool, patient_states, max_steps=30):
    """
    Progressively adds hidden variables to force the biggest partition to grow by exactly 1.
    
    Starting with all available_nodes as evidence and 0 hidden variables,
    at each step we move one evidence variable to the hidden set such that
    it joins the current biggest partition.
    
    Returns a list of (hidden_vars, evidence_patient) tuples to benchmark.
    """
    all_nodes = set(bn.nodes())
    available_nodes = [n for n in all_nodes if n != target]
    
    # Start with ALL available as evidence (0 hidden variables)
    evidence = {v: patient_states[v] for v in available_nodes}
    hidden = []
    
    configurations = []
    
    for step in range(max_steps):
        partitions = get_partitions(bn, hidden, target, evidence)
        
        if not partitions:
            # No hidden variables yet — pick any node connected to target
            # via active paths given current evidence
            candidates = list(evidence.keys())
        else:
            # Find the current biggest partition
            biggest_partition = max(partitions, key=len)
            biggest_size = len(biggest_partition)
            
            # We want to find an evidence variable that, when moved to hidden,
            # will be d-connected to biggest_partition (joining it)
            candidates = []
            for candidate in list(evidence.keys()):
                # Trial: move this candidate to hidden, see what happens
                trial_hidden = hidden + [candidate]
                trial_evidence = {k: v for k, v in evidence.items() if k != candidate}
                
                trial_partitions = get_partitions(bn, trial_hidden, target, trial_evidence)
                if not trial_partitions:
                    continue
                
                new_biggest = max(trial_partitions, key=len)
                new_biggest_size = len(new_biggest)
                
                # We want the biggest partition to grow by exactly 1
                if new_biggest_size == biggest_size + 1:
                    candidates.append(candidate)
        
        if not candidates:
            print(f"Step {step}: no candidate variable grows the biggest partition. Stopping.")
            break
        
        # Pick the first valid candidate
        chosen = candidates[0]
        hidden.append(chosen)
        del evidence[chosen]
        
        new_partitions = get_partitions(bn, hidden, target, evidence)
        new_biggest_size = max(len(p) for p in new_partitions) if new_partitions else 0
        
        print(f"Step {step+1:3d}: added '{chosen}' → "
              f"n_hidden={len(hidden)}, biggest_partition={new_biggest_size}")
        
        configurations.append({
            'step': step + 1,
            'n_hidden': len(hidden),
            'biggest_partition_size': new_biggest_size,
            'hidden_vars': list(hidden),
            'patient': dict(evidence),
        })
    
    return configurations

import threading
import psutil
import os
import tracemalloc
import linecache
import gc
import time
from math import prod

def compute_tensor_size(bn, partition):
    return prod(len(bn.get_cpds(v).state_names[v]) for v in partition)

def compute_max_tensor_size(bn, partitions):
    if not partitions:
        return 0
    return max(compute_tensor_size(bn, p) for p in partitions)

def profile_sdp_allocations(bn, target, target_value, patient, threshold, partitions, top_n=15):
    """
    Properly profiles fast_broadcast_sdp by sampling memory during execution
    to catch transient numpy tensor allocations that get freed before return.
    """
    process = psutil.Process(os.getpid())

    gc.collect()
    tracemalloc.start(25)
    baseline_rss = process.memory_info().rss
    baseline_traced = tracemalloc.get_traced_memory()[0]

    peak_rss = [baseline_rss]
    peak_traced = [0]
    peak_snapshot = [None]
    stop_event = threading.Event()

    def sampler():
        """Samples memory every 1ms, captures snapshot when new peak found."""
        while not stop_event.is_set():
            try:
                current_rss = process.memory_info().rss
                if current_rss > peak_rss[0]:
                    peak_rss[0] = current_rss

                _, peak = tracemalloc.get_traced_memory()
                if peak > peak_traced[0]:
                    peak_traced[0] = peak
                    peak_snapshot[0] = tracemalloc.take_snapshot()
            except Exception:
                pass
            time.sleep(0.001)

    sampler_thread = threading.Thread(target=sampler, daemon=True)
    sampler_thread.start()

    try:
        result = fast_broadcast_sdp(bn, target, target_value, patient, threshold, partitions)
    finally:
        stop_event.set()
        sampler_thread.join(timeout=2.0)

    final_peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    peak_tracked_mb = max(peak_traced[0], final_peak) / 1024 / 1024
    peak_rss_mb = (peak_rss[0] - baseline_rss) / 1024 / 1024

    print(f"\n{'='*60}")
    print(f"MEMORY PROFILE — partition={max(len(p) for p in partitions)}")
    print(f"{'='*60}")
    print(f"Peak tracemalloc (Python+numpy):  {peak_tracked_mb:.2f} MB")
    print(f"Peak RSS delta (OS-level):        {peak_rss_mb:.2f} MB")
    print(f"Snapshots captured at new peaks:  {'yes' if peak_snapshot[0] else 'no'}")

    if peak_snapshot[0] is not None:
        stats = peak_snapshot[0].filter_traces([
            tracemalloc.Filter(inclusive=True, filename_pattern="*same_decision*"),
        ]).statistics('lineno')

        print(f"\nTop {top_n} allocations AT PEAK inside fast_broadcast_sdp:")
        for i, stat in enumerate(stats[:top_n]):
            if stat.size < 1024:
                continue
            frame = stat.traceback[0]
            line = linecache.getline(frame.filename, frame.lineno).strip()
            print(f"  #{i+1:2d} {stat.size / 1024 / 1024:7.3f} MB — "
                  f"line {frame.lineno}: {line[:80]}")

    return result, peak_tracked_mb, peak_rss_mb

def benchmark_growing_partition(bif_file, max_steps=30, mcmc_trials=1):
    """
    Runs the exact SDP and MCMC benchmark on configurations where the 
    biggest partition grows by exactly 1 at each step.
    """
    import tracemalloc
    import gc
    import time
    
    bn = BIFReader(bif_file).get_model()
    all_nodes = list(bn.nodes())
    target = select_optimal_target_node(bn)
    target_states = bn.get_cpds(target).state_names[target]
    target_value = target_states[1] if len(target_states) > 1 else target_states[0]
    available_nodes = [n for n in all_nodes if n != target]
    
    # Fix a patient state for all evidence variables upfront
    patient_states = {
        n: bn.get_cpds(n).state_names[n][0] 
        for n in available_nodes
    }
    
    configurations = build_growing_partition_benchmark(
        bn, target, available_nodes, patient_states, max_steps=max_steps
    )
    
    results = []
    
    for config in configurations:
        hidden_vars = config['hidden_vars']
        patient = config['patient']
        partitions = get_partitions(bn, hidden_vars, target, patient)
        max_partition = max(len(p) for p in partitions)
        max_partition_tensor_size = compute_max_tensor_size(bn, partitions)
        
        row = {
            'step': config['step'],
            'n_hidden': config['n_hidden'],
            'max_partition_size': max_partition,
            'max_partition_tensor_size': max_partition_tensor_size,
            'exact_time_sec': None,
            'exact_peak_memory_mb': None,
            'exact_success': False,
            'mcmc_avg_time_sec': None,
            'mcmc_peak_memory_mb': None,
            'mcmc_avg_estimate': None,
            'mcmc_success': False,
        }
        
        # Exact SDP
        gc.collect()
        try:
            start = time.perf_counter()
            real_sdp, peak_traced_mb, peak_rss_mb = profile_sdp_allocations(
                bn, target, target_value, patient, 0.5, partitions
            )
            exact_time = time.perf_counter() - start

            row['exact_time_sec'] = exact_time
            row['exact_peak_memory_mb'] = peak_traced_mb
            row['exact_peak_rss_mb'] = peak_rss_mb
            row['exact_success'] = True
            row['exact_sdp_result'] = real_sdp

            print(f"Step {config['step']:3d} | partition={max_partition:3d} | "
                  f"Max partition tensor size: {max_partition_tensor_size} entries | "
                f"Time: {exact_time:.4f}s | Traced: {peak_traced_mb:.2f}MB | RSS: {peak_rss_mb:.2f}MB")
        except Exception as e:
            print(f"Step {config['step']:3d} | partition={max_partition:3d} | FAILED: {e}")
        
        # MCMC
        gc.collect()
        tracemalloc.start()
        try:
            mcmc_times = []
            mcmc_estimates = []
            for _ in range(mcmc_trials):
                start = time.perf_counter()
                est = fast_mcmc_sdp_estimation(bn, target, target_value, patient, 0.5,
                                              n_samples=1000, burn_in=200, thinning=10)
                mcmc_times.append(time.perf_counter() - start)
                mcmc_estimates.append(est)
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            print(f"Step {config['step']:3d} | partition={max_partition:3d} | "
                  f"MCMC Time/Memory: {np.mean(mcmc_times):.4f}s / {peak/1024/1024:.2f}MB | Estimate: {np.mean(mcmc_estimates):.4f}")
            
            row['mcmc_avg_time_sec'] = np.mean(mcmc_times)
            row['mcmc_peak_memory_mb'] = peak / 1024 / 1024
            row['mcmc_avg_estimate'] = np.mean(mcmc_estimates)
            row['mcmc_error'] = abs(np.mean(mcmc_estimates) - real_sdp) if row['exact_success'] else None
            row['mcmc_success'] = True
        except Exception as e:
            tracemalloc.stop()
            print(f"Step {config['step']:3d} | MCMC FAILED: {e}")
        
        results.append(row)
        pd.DataFrame(results).to_csv("results/growing_partition_benchmark.csv", index=False)
    
    return results



if __name__ == "__main__":

    #benchmark_hidden_vars("./generated_bif_files/bn_n200_w2_uncertain_strong.bif", n_hidden=150)
   results = benchmark_growing_partition("./generated_bif_files/bn_n50_w2_uncertain_strong.bif", max_steps=20, mcmc_trials=2)