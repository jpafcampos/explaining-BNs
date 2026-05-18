# benchmark_memory.py
import tracemalloc
import gc
from pgmpy.readwrite import BIFReader

from same_decision_probability_calculation import *
from monte_carlo_sdp import *

import random
import time 

import networkx as nx

MCMC_TRIALS = 1

def _moral_graph(bn):
    G = nx.Graph()
    G.add_nodes_from(bn.nodes())
    for node in bn.nodes():
        parents = list(bn.predecessors(node))
        for p in parents:
            G.add_edge(node, p)
        for i, p1 in enumerate(parents):
            for p2 in parents[i+1:]:
                G.add_edge(p1, p2)
    return G


def constrained_treewidth(bn, eliminate_last, evidence_vars=None):
    """
    Min-fill upper bound on the constrained treewidth: width of an elimination
    order in which the variables in `eliminate_last` are eliminated AFTER
    everything else. This is the quantity that bounds the cost of Algorithm 1
    in Choi, Xue, Darwiche (2012) — cost is O(N * 2^tw) for the VE phase.

    Pass `evidence_vars` to strip instantiated nodes before elimination
    (they're cost-free, so leaving them out gives a tighter bound).
    """
    G = _moral_graph(bn)
    last = set(eliminate_last)

    # Strip evidence: connect its neighbours, then drop it.
    if evidence_vars:
        for v in list(evidence_vars):
            if v in G:
                nbrs = list(G.neighbors(v))
                for i, a in enumerate(nbrs):
                    for b in nbrs[i+1:]:
                        G.add_edge(a, b)
                G.remove_node(v)

    def fill_in(graph, v):
        nbrs = list(graph.neighbors(v))
        return sum(
            1 for i, a in enumerate(nbrs)
            for b in nbrs[i+1:]
            if not graph.has_edge(a, b)
        )

    def eliminate(graph, v):
        nbrs = list(graph.neighbors(v))
        for i, a in enumerate(nbrs):
            for b in nbrs[i+1:]:
                graph.add_edge(a, b)
        graph.remove_node(v)
        return len(nbrs) + 1  # clique size including v

    max_clique = 0

    # Phase 1: eliminate the "free" vars first (anything not in eliminate_last).
    free = [v for v in G.nodes() if v not in last]
    while free:
        v = min(free, key=lambda x: fill_in(G, x))
        max_clique = max(max_clique, eliminate(G, v))
        free.remove(v)

    # Phase 2: eliminate the constrained set last.
    remaining = [v for v in eliminate_last if v in G]
    while remaining:
        v = min(remaining, key=lambda x: fill_in(G, x))
        max_clique = max(max_clique, eliminate(G, v))
        remaining.remove(v)

    return max_clique - 1   # treewidth = max clique size - 1

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


import pandas as pd

import numpy as np
from pgmpy.factors.discrete import TabularCPD

import numpy as np
from pgmpy.factors.discrete import TabularCPD

def inject_determinism(bn, sparsity=0.4):
    """
    Randomly injects zeros into the CPTs of a BayesianNetwork to simulate 
    the determinism found in real-world networks.
    """
    for cpd in bn.get_cpds():
        vals = cpd.values.copy()
        var_card = cpd.variable_card
        
        # Calculate how many parent instantiations exist
        num_parent_inst = np.prod(cpd.cardinality[1:]) if len(cpd.cardinality) > 1 else 1
        
        # Flatten into a 2D array: (states_of_variable, parent_instantiations)
        vals_2d = vals.reshape(var_card, num_parent_inst)
        
        for col in range(num_parent_inst):
            # Try to inject zeros based on the sparsity threshold
            for row in range(var_card):
                if np.random.rand() < sparsity:
                    # CRITICAL: Only zero it out if it's NOT the last remaining non-zero probability
                    if np.sum(vals_2d[:, col] > 0) > 1:
                        vals_2d[row, col] = 0.0
            
            # Re-normalize the column so it sums to 1.0
            col_sum = np.sum(vals_2d[:, col])
            vals_2d[:, col] /= col_sum
            
        # FIX: Check if the node actually has parents before setting evidence
        has_parents = len(cpd.variables) > 1
        
        new_cpd = TabularCPD(
            variable=cpd.variable,
            variable_card=cpd.variable_card,
            values=vals_2d, 
            evidence=cpd.variables[1:] if has_parents else None,
            evidence_card=cpd.cardinality[1:] if has_parents else None,
            state_names=cpd.state_names
        )
        
        bn.add_cpds(new_cpd)
        
    print(f"Injected ~{sparsity*100}% sparsity into the network CPTs.")
    return bn

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
        new_biggest = max(new_partitions, key=len) if new_partitions else []
        new_biggest_size = len(new_biggest)
        
        # Constrained treewidth: H eliminated last (bounds Algorithm 1 globally)
        tw_all_H = constrained_treewidth(
            bn, eliminate_last=hidden, evidence_vars=set(evidence.keys())
        )
        # Constrained treewidth: only the biggest partition last (per-bucket cost)
        tw_biggest = constrained_treewidth(
            bn, eliminate_last=new_biggest, evidence_vars=set(evidence.keys())
        )


        print(f"Step {step+1:3d}: added '{chosen}' → "
              f"n_hidden={len(hidden)}, biggest_partition={new_biggest_size}, "
              f"tw(H)={tw_all_H}, tw(biggest)={tw_biggest}")
        
        configurations.append({
            'step': step + 1,
            'n_hidden': len(hidden),
            'biggest_partition_size': new_biggest_size,
            'constrained_treewidth_H': tw_all_H,
            'constrained_treewidth_biggest': tw_biggest,
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

from pgmpy.utils import get_example_model



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

def benchmark_plain_mcmc(bn, target, target_value, patient):
    """Run MCMC_TRIALS times and measure memory once. Returns dict."""
    ests, times = [], []
    success = True
    for _ in range(MCMC_TRIALS):
        est, t, ok = run_for_time(
            fast_mcmc_sdp_estimation_new, bn, target, target_value, patient,
            0.5, n_samples=1000, burn_in=5000,
            thinning=100, use_lw_seed=True
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
        0.5, n_samples=10, burn_in=5, thinning=5,
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
            0.5, n_samples=1000, burn_in=5000,
            thinning=100, n_chains=4, max_temp=40,
            use_ancestral_seed=True
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
        0.5, n_samples=10, burn_in=5, thinning=5,
        n_chains=4, max_temp=40, use_ancestral_seed=True
    )
    return {
        'mean': float(np.mean(ests)),
        'var':  float(np.var(ests)),
        'avg_time': float(np.mean(times)),
        'mem_py': mem_py, 'mem_rss': mem_rss,
        'success': True,
    }

def benchmark_growing_partition(bif_file, max_steps=30):
    """
    Runs the exact SDP and MCMC benchmark on configurations where the 
    biggest partition grows by exactly 1 at each step.
    """
    import tracemalloc
    import gc
    import time
    
    bn = BIFReader(bif_file).get_model()
    #bn = inject_determinism(bn, sparsity=0.4)
    #bn = get_example_model('win95pts')
    
    all_nodes = list(bn.nodes())
    target = select_optimal_target_node(bn)
    #target = 'PrtMem'
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
            'constrained_treewidth_H': config['constrained_treewidth_H'],
            'constrained_treewidth_biggest': config['constrained_treewidth_biggest'],
            'exact_time_sec': None,
            'exact_peak_memory_mb': None,
            'exact_success': False,
            'mcmc_avg_time_sec': None,
            'mcmc_peak_memory_mb': None,
            'mcmc_avg_estimate': None,
            'mcmc_success': False,
            'pt_avg_time_sec': None,
            'pt_peak_memory_mb': None,
            'pt_avg_estimate': None,
            'pt_success': False
        }
        
        # Exact SDP - Run for memory
        if config['step'] < 28:
            gc.collect()
            try:
                #start = time.perf_counter()
                #real_sdp, peak_traced_mb, peak_rss_mb = profile_sdp_allocations(
                #    bn, target, target_value, patient, 0.5, partitions
                #)
                #exact_time = time.perf_counter() - start

                real_sdp, exact_time, exact_success = run_for_time(
                    fast_broadcast_sdp, bn, target, target_value, patient,
                    0.5, partitions
                )
                peak_traced_mb, peak_rss_mb = run_for_memory(
                    fast_broadcast_sdp, bn, target, target_value, patient,
                    0.5, partitions)

                row['exact_time_sec'] = exact_time
                row['exact_peak_memory_mb'] = peak_traced_mb
                row['exact_peak_rss_mb'] = peak_rss_mb
                row['exact_success'] = exact_success
                row['exact_sdp_result'] = real_sdp

                print(f"Step {config['step']:3d} | partition={max_partition:3d} | "
                    f"Max partition tensor size: {max_partition_tensor_size} entries | "
                    f"Time: {exact_time:.4f}s | Traced: {peak_traced_mb:.2f}MB | RSS: {peak_rss_mb:.2f}MB")
            except Exception as e:
                print(f"Step {config['step']:3d} | partition={max_partition:3d} | FAILED: {e}")

            #Exact SDP 2 - Accurate Chen paper version
            gc.collect()
            try:
                real_sdp_chen, exact_time_chen, exact_success_chen = run_for_time(
                    chen_sdp_exact, bn, target, target_value, patient,
                    0.5, partitions
                )
                peak_traced_mb_chen, peak_rss_mb_chen = run_for_memory(
                    chen_sdp_exact, bn, target, target_value, patient,
                    0.5, partitions)

                row['exact_time_sec_chen'] = exact_time_chen
                row['exact_peak_memory_mb_chen'] = peak_traced_mb_chen
                row['exact_peak_rss_mb_chen'] = peak_rss_mb_chen
                row['exact_success_chen'] = exact_success_chen
                row['exact_sdp_result_chen'] = real_sdp_chen

                print("Accurate Chen paper version:")
                print(f"Step {config['step']:3d} | partition={max_partition:3d} | "
                    f"Max partition tensor size: {max_partition_tensor_size} entries | "
                    f"Time: {exact_time_chen:.4f}s | Traced: {peak_traced_mb_chen:.2f}MB | RSS: {peak_rss_mb_chen:.2f}MB")
            except Exception as e:
                print(f"Step {config['step']:3d} | partition={max_partition:3d} | FAILED: {e}")
        
        else:
            # SDP cannot run, fill data with N/A
            #print(f"Step {config['step']:3d} | partition={max_partition:3d} | SDP SKIPPED (too large)")
            row['exact_time_sec'] = None
            row['exact_peak_memory_mb'] = None
            row['exact_peak_rss_mb'] = None
            row['exact_success'] = False
            row['exact_time_sec_chen'] = None
            row['exact_peak_memory_mb_chen'] = None
            row['exact_peak_rss_mb_chen'] = None
            row['exact_success_chen'] = False

        

        # MCMC
        gc.collect()
        try:
            mcmc_result = benchmark_plain_mcmc(bn, target, target_value, patient)
            if mcmc_result['success']:
                print(f"Step {config['step']:3d} | partition={max_partition:3d} | "
                      f"MCMC Time/Memory: {mcmc_result['avg_time']:.4f}s / {mcmc_result['mem_py']:.2f}MB | Estimate: {mcmc_result['mean']:.4f}")
                print(f"MCMC error: {abs(mcmc_result['mean'] - real_sdp) if row['exact_success'] else 'N/A'}")
                row['mcmc_avg_time_sec'] = mcmc_result['avg_time']
                row['mcmc_peak_memory_mb'] = mcmc_result['mem_py']
                row['mcmc_avg_estimate'] = mcmc_result['mean']
                row['mcmc_error'] = abs(mcmc_result['mean'] - real_sdp) if row['exact_success'] else None
                row['mcmc_success'] = True
            else:
                print(f"Step {config['step']:3d} | MCMC FAILED")
        except Exception as e:
            print(f"Step {config['step']:3d} | MCMC FAILED: {e}")
        
        # PT
        try:
            pt_result = benchmark_pt(bn, target, target_value, patient)
            if pt_result['success']:
                print(f"Step {config['step']:3d} | partition={max_partition:3d} | "
                      f"PT Time/Memory: {pt_result['avg_time']:.4f}s / {pt_result['mem_py']:.2f}MB | Estimate: {pt_result['mean']:.4f}")
                print(f"PT error: {abs(pt_result['mean'] - real_sdp) if row['exact_success'] else 'N/A'}")
                row['pt_avg_time_sec'] = pt_result['avg_time']
                row['pt_peak_memory_mb'] = pt_result['mem_py']
                row['pt_avg_estimate'] = pt_result['mean']
                row['pt_error'] = abs(pt_result['mean'] - real_sdp) if row['exact_success'] else None
                row['pt_success'] = True
            else:
                print(f"Step {config['step']:3d} | PT FAILED")
        except Exception as e:
            print(f"Step {config['step']:3d} | PT FAILED: {e}")
            
        results.append(row)
        pd.DataFrame(results).to_csv("results/growing_partition_benchmark_all_methods.csv", index=False)
    
    return results



if __name__ == "__main__":

   results = benchmark_growing_partition("./generated_bif_files/bn_n50_w2_uncertain_strong.bif", max_steps=27)