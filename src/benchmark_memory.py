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

def benchmark_hidden_vars_until_max(bif_file, max_hidden=20):
    results = []
    bn = BIFReader(bif_file).get_model()
    all_nodes = list(bn.nodes())
    target = select_optimal_target_node(bn)
    target_states = bn.get_cpds(target).state_names[target]
    target_value = target_states[1] if len(target_states) > 1 else target_states[0]
    available_nodes = [n for n in all_nodes if n != target]

    for n_hidden in range(1, min(max_hidden, len(available_nodes))):
        hidden_vars = available_nodes[:n_hidden]
        patient = {n: bn.get_cpds(n).state_names[n][0] 
                   for n in available_nodes if n not in hidden_vars}

        gc.collect()
        tracemalloc.start()

        try:
            partitions = get_partitions(bn, hidden_vars, target, patient)
            
            start = time.perf_counter()
            result = fast_broadcast_sdp(bn, target, target_value, patient, 0.5, partitions)
            elapsed = time.perf_counter() - start

            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            print(f"Hidden vars: {n_hidden:3d} | "
                  f"Size of biggest partition: {max(len(p) for p in partitions):3d} | "
                  f"Peak memory: {peak / 1024 / 1024:.2f} MB | "
                  f"Time: {elapsed:.4f} sec | OK")
            
            # create results to be saved as csv
            results.append({
                "n_hidden": n_hidden,
                "hidden_edges": bn.subgraph(hidden_vars).number_of_edges(),
                "max_partition_size": max(len(p) for p in partitions),
                "time_sec": elapsed,
                "peak_memory_mb": peak / 1024 / 1024
            })

            #save results to csv after each iteration
            df = pd.DataFrame(results)
            df.to_csv("results/benchmark_memory_results.csv", index=False)

        except MemoryError:
            tracemalloc.stop()
            print(f"Hidden vars: {n_hidden:3d} | OUT OF MEMORY")
            break
        except Exception as e:
            tracemalloc.stop()
            print(f"Hidden vars: {n_hidden:3d} | ERROR: {e}")
            break

    return results

def benchmark_hidden_vars_sdp_vs_mcmc(bif_file, max_hidden=20, mcmc_trials=1):
    results = []
    bn = BIFReader(bif_file).get_model()
    all_nodes = list(bn.nodes())
    target = select_optimal_target_node(bn)
    target_states = bn.get_cpds(target).state_names[target]
    target_value = target_states[1] if len(target_states) > 1 else target_states[0]
    available_nodes = [n for n in all_nodes if n != target]

    for n_hidden in range(1, min(max_hidden, len(available_nodes))):
        hidden_vars = available_nodes[:n_hidden]
        patient = {n: bn.get_cpds(n).state_names[n][0] 
                   for n in available_nodes if n not in hidden_vars}

        partitions = get_partitions(bn, hidden_vars, target, patient)
        max_partition = max(len(p) for p in partitions)
        hidden_edges = bn.subgraph(hidden_vars).number_of_edges()

        row = {
            "n_hidden": n_hidden,
            "hidden_edges": hidden_edges,
            "max_partition_size": max_partition,
            # Exact SDP fields
            "exact_time_sec": None,
            "exact_peak_memory_mb": None,
            "exact_success": False,
            # MCMC fields
            "mcmc_avg_time_sec": None,
            "mcmc_peak_memory_mb": None,
            "mcmc_avg_estimate": None,
            "mcmc_success": False,
        }

        # ========================================================
        # EXACT SDP
        # ========================================================
        gc.collect()
        tracemalloc.start()
        try:
            start = time.perf_counter()
            exact_result = fast_broadcast_sdp(bn, target, target_value, patient, 0.5, partitions)
            exact_time = time.perf_counter() - start
            _, exact_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            row["exact_time_sec"] = exact_time
            row["exact_peak_memory_mb"] = exact_peak / 1024 / 1024
            row["exact_success"] = True

            print(f"Hidden vars: {n_hidden:3d} | "
                  f"Biggest partition: {max_partition:3d} | "
                  f"Exact — Time: {exact_time:.4f}s | "
                  f"Memory: {exact_peak / 1024 / 1024:.2f} MB | OK")

        except MemoryError:
            tracemalloc.stop()
            print(f"Hidden vars: {n_hidden:3d} | Biggest partition: {max_partition:3d} | Exact — OUT OF MEMORY")
        except Exception as e:
            tracemalloc.stop()
            print(f"Hidden vars: {n_hidden:3d} | Biggest partition: {max_partition:3d} | Exact — ERROR: {e}")

        # ========================================================
        # MCMC
        # ========================================================
        gc.collect()
        mcmc_times = []
        mcmc_estimates = []

        tracemalloc.start()
        try:
            for trial in range(mcmc_trials):
                start = time.perf_counter()
                est = fast_mcmc_sdp_estimation(bn, target, target_value, patient, 0.5,
                                               n_samples=1000, burn_in=200, thinning=10)
                mcmc_times.append(time.perf_counter() - start)
                mcmc_estimates.append(est)

            _, mcmc_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            row["mcmc_avg_time_sec"] = np.mean(mcmc_times)
            row["mcmc_peak_memory_mb"] = mcmc_peak / 1024 / 1024
            row["mcmc_avg_estimate"] = np.mean(mcmc_estimates)
            row["mcmc_success"] = True

            print(f"Hidden vars: {n_hidden:3d} | "
                  f"Size of biggest partition: {max(len(p) for p in partitions):3d} | "
                  f"Peak memory: {mcmc_peak / 1024 / 1024:.2f} MB | "
                  f"Time: {np.mean(mcmc_times):.4f} sec | OK")

        except MemoryError:
            tracemalloc.stop()
            print(f"Hidden vars: {n_hidden:3d} | Biggest partition: {max_partition:3d} | MCMC  — OUT OF MEMORY")
        except Exception as e:
            tracemalloc.stop()
            print(f"{"":>14} Biggest partition: {max_partition:3d} | MCMC  — ERROR: {e}")

        # ========================================================
        # SAVE & CONTINUE
        # ========================================================
        results.append(row)
        pd.DataFrame(results).to_csv("results/benchmark_memory_results.csv", index=False)

        # Stop if exact SDP already failed — no point continuing
        if not row["exact_success"]:
            print("Exact SDP failed — stopping benchmark.")
            break

    return results

if __name__ == "__main__":

    #benchmark_hidden_vars("./generated_bif_files/bn_n200_w2_uncertain_strong.bif", n_hidden=150)
    #results = benchmark_hidden_vars_until_max("./generated_bif_files/bn_n50_w2_uncertain_strong.bif", max_hidden=40)
    results = benchmark_hidden_vars_sdp_vs_mcmc("./generated_bif_files/bn_n50_w2_uncertain_strong.bif", max_hidden=40, mcmc_trials=1)