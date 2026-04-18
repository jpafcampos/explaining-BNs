# benchmark_memory.py
import tracemalloc
import gc
from pgmpy.readwrite import BIFReader

from same_decision_probability_calculation import *

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

if __name__ == "__main__":

    benchmark_hidden_vars("./generated_bif_files/bn_n200_w2_uncertain_strong.bif", n_hidden=150)
    #benchmark_hidden_vars_until_max("./generated_bif_files/bn_n200_w2_uncertain_strong.bif", max_hidden=38)