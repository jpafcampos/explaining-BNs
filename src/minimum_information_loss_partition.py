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
import bnlearn as bn
import itertools
import math
import random

'''
This module implements the Minimum Information Partition (MIP) for a set of variables in a Bayesian
Main References:

Minimizing symmetric submodular functions, Queyranne, M. (1998). Mathematical Programming, 82(1-2), 3-12.
Fast and exact search for the partition with minimal information loss, Hidaka & Oizumi. (2018). Plos One.
'''

def get_submodular_score(A_indices, cov_matrix):
    """
    Evaluates the mutual information (information loss) of a partition.
    Uses the determinant of the covariance matrices of the split subsets.
    """
    n = cov_matrix.shape[0]
    A = list(A_indices)
    A_comp = [i for i in range(n) if i not in A]
    
    # Add a tiny ridge (epsilon) to the diagonal to prevent singular matrix errors
    eps = 1e-9
    
    # If one side of the partition is empty, the information loss is mathematically zero
    if len(A) == 0 or len(A_comp) == 0:
        return 0.0
        
    # Use slogdet for numerical stability (prevents overflow/underflow)
    _, logdet_A = np.linalg.slogdet(cov_matrix[np.ix_(A, A)] + np.eye(len(A)) * eps)
    _, logdet_Ac = np.linalg.slogdet(cov_matrix[np.ix_(A_comp, A_comp)] + np.eye(len(A_comp)) * eps)
    
    # We ignore the logdet of the full matrix because it is constant across all partitions
    return logdet_A + logdet_Ac

def queyranne_mip(variables, cov_matrix):
    """
    Queyranne's Algorithm (1998) for symmetric submodular function minimization.
    Finds the exact Minimum Information Partition in O(N^3) time.
    """
    n = len(variables)
    
    # Start with every variable in its own individual set
    V = [{i} for i in range(n)]
    
    best_val = float('inf')
    best_partition_indices = None
    
    # The algorithm reduces the system by 1 element in each phase
    while len(V) > 1:
        # Phase Initialization
        W = [V[0]]
        unvisited = V[1:]
        
        # Pendant Pair Search (Maximum Cardinality Search analogue)
        while unvisited:
            best_x = None
            min_diff = float('inf')
            
            # Find the next cluster that minimizes the marginal increase in mutual info
            W_indices = set().union(*W)
            for x in unvisited:
                val_W_x = get_submodular_score(W_indices.union(x), cov_matrix)
                val_x = get_submodular_score(x, cov_matrix)
                
                # Formula: minimize f(W U {x}) - f({x})
                diff = val_W_x - val_x 
                if diff < min_diff:
                    min_diff = diff
                    best_x = x
                    
            W.append(best_x)
            unvisited.remove(best_x)
            
        # The last two elements added are the "pendant pair" (t, u)
        t = W[-2]
        u = W[-1]
        
        # Evaluate the cut separating {u} from the rest of the network
        score = get_submodular_score(u, cov_matrix)
        if score < best_val:
            best_val = score
            best_partition_indices = u
            
        # Merge t and u into a single element for the next phase
        V.remove(t)
        V.remove(u)
        V.append(t.union(u))
        
    # Map the numerical indices back to your actual pgmpy variable names
    part1 = [variables[i] for i in best_partition_indices]
    part2 = [var for var in variables if var not in part1]
    
    return part1, part2

def recursive_queyranne(variables, cov_matrix):
    """
    Recursively applies Queyranne's algorithm to partition a set of variables
    until no resulting cluster is larger than half the size of the original set.
    """
    n_original = len(variables)
    max_allowed_size = n_original // 2
    
    # We will store the final valid clusters here
    final_clusters = []
    
    # A queue to hold clusters that still need to be evaluated
    clusters_to_process = [variables]
    
    while clusters_to_process:
        current_cluster = clusters_to_process.pop(0)
        
        # If the cluster is already small enough, save it and move on
        if len(current_cluster) <= max_allowed_size:
            final_clusters.append(current_cluster)
            continue
            
        # Otherwise, we need to split it using Queyranne's algorithm.
        # First, we must extract the sub-covariance matrix for just these variables.
        # We find the indices of the current_cluster variables in the ORIGINAL list.
        subset_indices = [variables.index(v) for v in current_cluster]
        sub_cov_matrix = cov_matrix[np.ix_(subset_indices, subset_indices)]
        
        # Call the original queyranne_mip on this specific subset
        part1, part2 = queyranne_mip(current_cluster, sub_cov_matrix)
        
        # Add the two newly split partitions back into the queue for checking
        clusters_to_process.append(part1)
        clusters_to_process.append(part2)
        
    return final_clusters

def balanced_mip_swap_heuristic(variables, cov_matrix, max_iter=100):
    """
    Finds a balanced partition (|A| = |B| = N/2) minimizing information loss 
    using a Kernighan-Lin style greedy swap heuristic.
    """
    n = len(variables)
    #if n % 2 != 0:
    #    raise ValueError("The number of variables must be even to split into two equal halves.")
        
    half_n = n // 2
    
    # 1. Start with an arbitrary balanced split
    A_indices = set(range(half_n))
    B_indices = set(range(half_n, n))
    
    current_score = get_submodular_score(A_indices, cov_matrix)
    
    # 2. Iteratively improve the partition
    for iteration in range(max_iter):
        best_swap = None
        best_score_improvement = 0
        
        # Test swapping every element in A with every element in B
        for a in A_indices:
            for b in B_indices:
                # Propose the swap
                proposed_A = (A_indices - {a}).union({b})
                proposed_score = get_submodular_score(proposed_A, cov_matrix)
                
                improvement = current_score - proposed_score
                
                if improvement > best_score_improvement:
                    best_score_improvement = improvement
                    best_swap = (a, b)
                    
        # 3. If we found a swap that lowers information loss, apply it
        if best_swap:
            a, b = best_swap
            A_indices.remove(a)
            A_indices.add(b)
            B_indices.remove(b)
            B_indices.add(a)
            current_score -= best_score_improvement
        else:
            # Local minimum reached; no single swap improves the score
            break 
            
    # Map indices back to variable names
    part1 = [variables[i] for i in A_indices]
    part2 = [variables[i] for i in B_indices]
    
    return part1, part2

def balanced_mip_search(variables, cov_matrix, alpha=2, max_samples=5000):
    """
    Finds a balanced Minimum Information Partition by trading off 
    information loss with a size-imbalance penalty.
    
    alpha: The trade-off parameter. 
           0.0 = Pure Information Loss (Unbalanced)
           Higher values = Forces more balanced splits (e.g., 6 vs 7)
    """
    n = len(variables)
    best_cost = float('inf')
    best_part_indices = None
    
    # We only need to check sizes up to n // 2 (since partitions are symmetric)
    target_sizes = range(1, (n // 2) + 1)
    
    for size in target_sizes:
        # If the number of combinations is small, evaluate all of them exactly
        # e.g., for N=13, size 6 has 1716 combinations (instant)
        num_combinations = math.comb(n, size)
        
        if num_combinations <= max_samples:
            candidates = itertools.combinations(range(n), size)
        else:
            # If N is larger (e.g., N=30), randomly sample to keep it lightning fast
            candidates = (random.sample(range(n), size) for _ in range(max_samples))
            
        for A_indices in candidates:
            A = list(A_indices)
            A_comp = [i for i in range(n) if i not in A]
            
            # 1. Compute Information Loss (Submodular Score)
            eps = 1e-9
            _, logdet_A = np.linalg.slogdet(cov_matrix[np.ix_(A, A)] + np.eye(len(A)) * eps)
            _, logdet_Ac = np.linalg.slogdet(cov_matrix[np.ix_(A_comp, A_comp)] + np.eye(len(A_comp)) * eps)
            info_loss = logdet_A + logdet_Ac
            
            # 2. Compute Imbalance Penalty
            size_difference = abs(len(A) - len(A_comp))
            
            # 3. The Trade-off Cost Function
            cost = info_loss + (alpha * size_difference)
            
            if cost < best_cost:
                best_cost = cost
                best_part_indices = A
                
    # Map the numerical indices back to your actual pgmpy variable names
    part1 = [variables[i] for i in best_part_indices]
    part2 = [var for var in variables if var not in part1]
    
    return part1, part2

def get_partition_covariance(model, partition_vars, evidence):
    """
    Builds a covariance matrix for discrete variables by computing 
    pairwise expected values via exact inference.
    """
    inference = VariableElimination(model)
    n = len(partition_vars)
    cov_matrix = np.zeros((n, n))
    
    # Get the marginal probabilities (Expected Value) for each variable
    E_x = []
    for var in partition_vars:
        dist = inference.query(variables=[var], evidence=evidence, show_progress=False)
        # We assume binary/ordinal states mapped to indices for the continuous proxy
        expected_val = sum(i * dist.values[i] for i in range(len(dist.values)))
        E_x.append(expected_val)
        
    # Calculate the Pairwise Covariance E[XY] - E[X]E[Y]
    for i in range(n):
        for j in range(i, n):
            if i == j:
                # Variance: E[X^2] - (E[X])^2
                var_i = partition_vars[i]
                dist = inference.query(variables=[var_i], evidence=evidence, show_progress=False)
                e_x2 = sum((k**2) * dist.values[k] for k in range(len(dist.values)))
                cov_matrix[i, i] = e_x2 - (E_x[i]**2)
            else:
                var_i = partition_vars[i]
                var_j = partition_vars[j]
                
                # Pairwise joint distribution
                dist = inference.query(variables=[var_i, var_j], evidence=evidence, show_progress=False)
                
                # E[XY]
                e_xy = 0.0
                state_names_i = dist.state_names[var_i]
                state_names_j = dist.state_names[var_j]
                
                for idx_i in range(len(state_names_i)):
                    for idx_j in range(len(state_names_j)):
                        prob = dist.get_value(**{var_i: state_names_i[idx_i], var_j: state_names_j[idx_j]})
                        e_xy += (idx_i * idx_j) * prob
                        
                covariance = e_xy - (E_x[i] * E_x[j])
                cov_matrix[i, j] = covariance
                cov_matrix[j, i] = covariance # Matrix is symmetric
                
    return cov_matrix