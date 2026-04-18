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
import os
import glob
import time
import tracemalloc
from multiprocessing import Pool

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

def select_optimal_target_node_old(bn):
    """
    Selects a target node deeply embedded in the network (highest degree).
    """
    best_node = None
    max_degree = -1
    
    for node in bn.nodes():
        # Ensure it's a binary node
        if len(bn.get_cpds(node).state_names[node]) != 2:
            continue
            
        degree = len(bn.get_parents(node)) + len(bn.get_children(node))
        if degree > max_degree:
            max_degree = degree
            best_node = node
            
    # Fallback if no binary nodes exist (rare)
    if best_node is None:
        return random.choice(list(bn.nodes()))
        
    return best_node

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
        return None, np.nan, False # Failed

def run_for_memory(func, *args, **kwargs):
    """Runs with tracemalloc to record peak memory. Ignores execution time."""
    tracemalloc.start()
    try:
        func(*args, **kwargs)
    except Exception:
        pass # We just want to see how high memory got before it crashed/finished
        
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak_mem / (1024 * 1024) # Return MB

def estimate_exact_inference_memory_accurate(bn, target, evidence_dict):
    """
    Simulates Variable Elimination accurately by accounting for Barren Node Pruning
    and Tensor Slicing caused by observed evidence and target partitions.
    """
    evidence_vars = list(evidence_dict.keys())
    relevant_nodes = evidence_vars + [target]
    
    # 1. Prune Barren Nodes (Get Ancestral Subgraph)
    # We use pgmpy's built-in DiGraph extraction for the ancestral graph
    ancestral_graph = bn.get_ancestral_graph(relevant_nodes)
    
    # 2. Moralize the pruned graph (Marry all parents)
    # We MUST do this before removing evidence to capture v-structure dependencies
    G = nx.Graph(ancestral_graph.edges()) # Convert to undirected
    G.add_nodes_from(ancestral_graph.nodes())
    
    for node in ancestral_graph.nodes():
        # Get parents in the directed ancestral graph
        parents = list(ancestral_graph.predecessors(node))
        for i in range(len(parents)):
            for j in range(i+1, len(parents)):
                 G.add_edge(parents[i], parents[j])
                 
    # 3. Apply Tensor Slicing (Remove Observed Variables)
    # Because they are known constants, they do not add dimensions to the VE tensors.
    # Removing them breaks paths and drastically lowers the effective treewidth!
    nodes_to_remove = set(evidence_vars) | {target}
    for node in nodes_to_remove:
        if node in G:
            G.remove_node(node)
            
    # 4. Simulate Min-Degree Elimination on the remaining Hidden Variables
    max_clique_size = 0
    nodes = list(G.nodes())
    
    while nodes:
        # Find node with minimum degree
        degrees = dict(G.degree(nodes))
        min_node = min(degrees, key=degrees.get)
        
        # Calculate the tensor size (Current Node + its uneliminated neighbors)
        neighbors = list(G.neighbors(min_node))
        clique_size = len(neighbors) + 1 
        
        if clique_size > max_clique_size:
            max_clique_size = clique_size
            
        # Connect all neighbors to each other (fill-in edges)
        for i in range(len(neighbors)):
            for j in range(i+1, len(neighbors)):
                G.add_edge(neighbors[i], neighbors[j])
                
        # Remove the node
        G.remove_node(min_node)
        nodes.remove(min_node)
        
    # If all variables were observed/pruned, memory is practically zero
    if max_clique_size == 0:
        return 0, 0.0

    # Calculate Memory: 2^W states * 8 bytes per float
    estimated_bytes = (2 ** max_clique_size) * 8
    estimated_gb = estimated_bytes / (1024 ** 3)
    
    return max_clique_size, estimated_gb

def find_exact_experimental_patients_slurm(bn, target_node, target_value, decision_threshold, n_evidence, buckets=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], tolerance=0.05, batch_size=8000, max_batches=2, mem_limit_gb=8.0):
    """
    Brute-force searches for patients by generating massive random batches.
    Includes a real-time pre-flight memory check to prevent Slurm OOM kills.
    """
    all_nodes = list(bn.nodes())
    available_nodes = [n for n in all_nodes if n != target_node]
    
    unfilled_buckets = {b: None for b in buckets}
    batch_count = 0
    oom_skips = 0  # Track how many patients we had to skip due to RAM limits
    
    while any(v is None for v in unfilled_buckets.values()) and batch_count < max_batches:
        batch_count += 1
        print(f"Generating batch {batch_count}/{max_batches} of {batch_size} random realities...")
        
        for _ in range(batch_size):
            # 1. Pick random evidence variables and states
            evidence_vars = random.sample(available_nodes, n_evidence)
            hidden_vars = [n for n in available_nodes if n not in evidence_vars]
            
            temp_patient = {}
            for var in evidence_vars:
                states = bn.get_cpds(var).state_names[var]
                temp_patient[var] = random.choice(states)
            
            # ==========================================
            # 2. THE SLURM SAFETY VALVE
            # Check the exact constrained memory of this specific patient
            # ==========================================
            max_tensor, est_gb = estimate_exact_inference_memory_accurate(bn, target_node, temp_patient)
            
            if est_gb > mem_limit_gb:
                oom_skips += 1
                # If the network is so dense that 10 random patients in a row exceed 8GB, 
                # we are wasting CPU time. Bail out of the network entirely.
                if oom_skips >= 10:
                    print(f"    [!] HARVEST ABORTED: Network fundamentally exceeds {mem_limit_gb}GB limit.")
                    return unfilled_buckets 
                continue # Skip this patient and try another random configuration
            else:
                oom_skips = 0 # Reset counter if we find a safe configuration!
            # ==========================================

            # 3. Create a fast, pruned sub-model for the base decision
            relevant_nodes = list(evidence_vars) + [target_node]
            ancestral_structure = bn.get_ancestral_graph(relevant_nodes)
            sub_model = BayesianNetwork(ancestral_structure.edges())
            sub_model.add_nodes_from(ancestral_structure.nodes())
            for node in sub_model.nodes():
                sub_model.add_cpds(bn.get_cpds(node))
            base_inference = VariableElimination(sub_model)
            
            # 4. Check base decision (Must be >= threshold)
            try:
                base_dist = base_inference.query(variables=[target_node], evidence=temp_patient, show_progress=False)
                if base_dist.get_value(**{target_node: target_value}) < decision_threshold:
                    continue 
            except Exception:
                continue
                
            # 5. Calculate Exact SDP (We know it is RAM-safe now!)
            partitions = get_partitions(bn, hidden_vars, target_node, temp_patient)
            try:
                exact_sdp = fast_broadcast_sdp(bn, target_node, target_value, temp_patient, decision_threshold, partitions)
            except Exception:
                continue
                
            # 6. Check if it fits into any empty bucket
            empty_targets = [b for b, v in unfilled_buckets.items() if v is None]
            for b in empty_targets:
                if abs(exact_sdp - b) <= tolerance:
                    unfilled_buckets[b] = (temp_patient.copy(), exact_sdp)
                    print(f"--> Filled bucket {b} with Exact SDP: {exact_sdp:.4f} (Peak RAM predicted: {est_gb:.4f} GB)")
                    break
                    
            if not any(v is None for v in unfilled_buckets.values()):
                break
                
    if any(v is None for v in unfilled_buckets.values()):
        missing = [b for b, v in unfilled_buckets.items() if v is None]
        print(f"Finished searching. Could not find patients for buckets: {missing}")
    else:
        print("All buckets filled successfully!")
        
    return unfilled_buckets

def process_single_file(args):
    """Process one BIF file and return results as a list of dicts."""
 
    file, H_RATIOS, DECISION_THRESHOLD, TARGET_BUCKETS, SIZES_TO_RUN, MCMC_TRIALS = args
    results = []

    n_nodes, density, rigidity = parse_bn_filename(file)
    if n_nodes not in SIZES_TO_RUN:
        return [] 

    print(f"\n========================================")
    print(f"Loading: {os.path.basename(file)}")
    
    bn = BIFReader(file).get_model()
    all_nodes = list(bn.nodes())

    ## --- SAFETY VALVE: Check Memory Limits Before Harvesting ---
    #max_tensor, est_gb = estimate_exact_inference_memory(bn)
    #print(f"  -> Max Tensor Size: {max_tensor} variables")
    #print(f"  -> Estimated Peak RAM: {est_gb:.2f} GB")
    #
    #if est_gb > 32.0: 
    #    print(f"  [!] DANGER: Network exceeds memory safety limits. Skipping entirely.")
    #    return [] # Return empty results so the worker survives and moves to the next file
    ## -----------------------------------------------------------
    
    target = select_optimal_target_node(bn)
    target_states = bn.get_cpds(target).state_names[target]
    target_value = target_states[1] if len(target_states) > 1 else target_states[0]
    
    available_nodes = [n for n in all_nodes if n != target]

    for H_RATIO in H_RATIOS:
        print(f"\n--- Hidden Ratio: {H_RATIO:.0%} ---")
        
        n_hidden = max(1, int(len(available_nodes) * H_RATIO))

        # can not process more than 100 hidden variables
        n_hidden = min(n_hidden, 100)

        n_evidence = len(available_nodes) - n_hidden
        #evidence_vars = [n for n in available_nodes if n not in hidden_vars]
        
        # Run the Harvester 
        #harvested_data = harvest_patients_for_all_buckets(
        #    bn, target, target_value, DECISION_THRESHOLD, evidence_vars, TARGET_BUCKETS
        #)
        harvested_data = find_exact_experimental_patients_slurm(bn, target, target_value, DECISION_THRESHOLD,
                                                                n_evidence, buckets=TARGET_BUCKETS, batch_size=8_000, mem_limit_gb=32.0)
        
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
            print(f"       -> Running Exact SDP...")
            
            # Pass 1: Time
            exact_sdp, exact_time, exact_success = run_for_time(
                fast_broadcast_sdp, bn, target, target_value, patient, DECISION_THRESHOLD, partitions
            )
            
            # Pass 2: Memory
            exact_mem_mb = run_for_memory(
                fast_broadcast_sdp, bn, target, target_value, patient, DECISION_THRESHOLD, partitions
            )
            
            if exact_success:
                print(f"          Time: {exact_time:.4f} sec | Peak Memory: {exact_mem_mb:.2f} MB")
            else:
                print(f"          [FAILED]: Crashed at {exact_mem_mb:.2f} MB")

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
            mcmc_mem_mb = run_for_memory(
                fast_mcmc_sdp_estimation, bn, target, target_value, patient, DECISION_THRESHOLD,
                n_samples=100, burn_in=50, thinning=5
            )
            
            print(f"          Avg Time: {mcmc_avg_time:.4f} sec | Peak Memory: {mcmc_mem_mb:.2f} MB")
            
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
                    pt_mcmc_sdp_estimation, bn, target, target_value, patient, DECISION_THRESHOLD,
                    n_samples=1000, burn_in=2000, thinning=50, n_chains=4, max_temp=40.0
                )
                pt_mcmc_estimates.append(est_sdp)
                pt_mcmc_times.append(t_time)

            pt_mcmc_mean = np.mean(pt_mcmc_estimates)
            pt_mcmc_avg_time = np.mean(pt_mcmc_times)
            pt_mcmc_variance = np.var(pt_mcmc_estimates)

            # Pass 2: Peak Memory
            pt_mcmc_mem_mb = run_for_memory(
                pt_mcmc_sdp_estimation, bn, target, target_value, patient, DECISION_THRESHOLD,
                n_samples=100, burn_in=50, thinning=5, n_chains=4, max_temp=10.0
            )

            print(f"          Avg Time: {pt_mcmc_avg_time:.4f} sec | Peak Memory: {pt_mcmc_mem_mb:.2f} MB")

            absolute_error_pt = abs(exact_sdp - pt_mcmc_mean)
            
        
            # Record everything to the dataset
            results.append({
                'Network': os.path.basename(file),
                'N_Nodes': n_nodes,
                'Density': density,
                'Rigidity': rigidity,
                'Target_Bucket': target_sdp,
                'Target_Node': target,
                'Target_Value': target_value,
                'Exact_SDP': exact_sdp,
                'Exact_Time_sec': exact_time,
                'MCMC_Mean_SDP': mcmc_mean,
                'MCMC_Variance': mcmc_variance,
                'MCMC_Avg_Time_sec': mcmc_avg_time,
                'Absolute_Error': absolute_error,
                'PT_MCMC_Mean_SDP': pt_mcmc_mean,
                'PT_MCMC_Variance': pt_mcmc_variance,
                'PT_MCMC_Avg_Time_sec': pt_mcmc_avg_time,
                'PT_Absolute_Error': absolute_error_pt
            })
    
    
    return results

def run_targeted_sdp_experiment(bif_directory, output_csv="targeted_sdp_random_bns.csv", n_workers=4):
    bif_files = sorted(glob.glob(os.path.join(bif_directory, "*.bif")))
    
    H_RATIOS = [0.1, 0.25, 0.5, 0.75, 0.9]
    #H_RATIOS = [0.3]
    DECISION_THRESHOLD = 0.5
    TARGET_BUCKETS = [0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0]
    #TARGET_BUCKETS = [0.5, 0.9]
    MCMC_TRIALS = 10

    SIZES_TO_RUN = [20, 50, 100, 200]
    #SIZES_TO_RUN = [200]
    # Build args list for each file
    args_list = [
        (file, H_RATIOS, DECISION_THRESHOLD, TARGET_BUCKETS, SIZES_TO_RUN, MCMC_TRIALS)
        for file in bif_files
    ]

    all_results = []
    with Pool(processes=n_workers) as pool:
        for file_results in pool.imap_unordered(process_single_file, args_list):
            all_results.extend(file_results)
            # Save progressively after each file completes
            #pd.DataFrame(all_results).to_csv(output_csv, index=False)

    pd.DataFrame(all_results).to_csv(output_csv, index=False)
    print(f"\nExperiment Complete! Results saved to {output_csv}")
    return pd.DataFrame(all_results)



import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bif-dir', type=str, default='./generated_bif_files/')
    parser.add_argument('--output', type=str, default='results/parallel_output.csv')
    parser.add_argument('--n-workers', type=int, default=4)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    run_targeted_sdp_experiment(args.bif_dir, args.output, args.n_workers)