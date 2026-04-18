# benchmark_memory.py
import tracemalloc
import gc
from pgmpy.readwrite import BIFReader

from same_decision_probability_calculation import *

import random

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



def benchmark_hidden_vars(bif_file, max_hidden=20):
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
            result = fast_broadcast_sdp(bn, target, target_value, patient, 0.5, partitions)
            
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            print(f"Hidden vars: {n_hidden:3d} | Peak memory: {peak / 1024 / 1024:.2f} MB | OK")

        except MemoryError:
            tracemalloc.stop()
            print(f"Hidden vars: {n_hidden:3d} | OUT OF MEMORY")
            break
        except Exception as e:
            tracemalloc.stop()
            print(f"Hidden vars: {n_hidden:3d} | ERROR: {e}")
            break

if __name__ == "__main__":

    benchmark_hidden_vars("./generated_bif_files/bn_n200_w12_uncertain_strong.bif", max_hidden=190)