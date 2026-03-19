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
import itertools
import math
from utils import *

'''
This module implements the exact Same-Decision Probability (SDP) calculation.
Main References:

Algorithms and Applications for the
Same-Decision Probability, Choi, Chen, & Darwiche (2014). JAIR 2014.

Same-decision probability: A confidence measure
for threshold-based decisions. Choi, Xue & Darwiche (2012). International Journal of Approximate Reasoning.
'''

def exact_sdp_bruteforce(model, D, d_value, evidence, threshold):
    """
    Exact brute-force computation of Same-Decision Probability (SDP).
    
    Parameters:
        model      : pgmpy Bayesian model
        D          : decision variable name (string)
        d_value    : state of D to test (e.g., 1 or 'yes')
        evidence   : dict of observed variables {var: value}
        threshold  : decision threshold T
        
    Returns:
        sdp (float)
    """
    # 1. Identify hidden variables (H)
    all_vars = set(model.nodes())
    observed_vars = set(evidence.keys())
    H = list(all_vars - observed_vars - {D})
    
    inference = VariableElimination(model)

    # 2. Determine the CURRENT decision (F(Pr(D|e)))
    # We must know if we are currently above or below the threshold
    current_dist = inference.query(variables=[D], evidence=evidence, show_progress=False)
    d_index = model.get_cpds(D).state_names[D].index(d_value)
    p_d_initial = current_dist.values[d_index]
    
    # The decision function F returns 1 if >= threshold, else 0
    current_decision = p_d_initial >= threshold

    # 3. Setup hidden variable state space
    state_spaces = [model.get_cpds(var).state_names[var] for var in H]
    all_assignments = list(itertools.product(*state_spaces))

    # 4. Pre-compute P(H | e) to get the weights for each scenario
    p_h_dist = inference.query(variables=H, evidence=evidence, show_progress=False)

    sdp = 0.0

    # 5. Iterate through all possible hidden variable instantiations (h)
    for assignment in all_assignments:
        h_dict = dict(zip(H, assignment))

        # Get the probability of this specific scenario: Pr(h | e)
        p_h_given_e = p_h_dist.get_value(**h_dict)

        if p_h_given_e == 0:
            continue

        # 6. Compute the NEW probability: Pr(d | e, h)
        query_e_h = {**evidence, **h_dict}
        p_d_given_e_h_dist = inference.query(
            variables=[D],
            evidence=query_e_h,
            show_progress=False
        )
        p_d_given_e_h = p_d_given_e_h_dist.values[d_index]

        # 7. Check if the decision is the SAME: [F(Pr(D|e,h)) == F(Pr(D|e))]
        new_decision = p_d_given_e_h >= threshold
        
        if new_decision == current_decision:
            sdp += p_h_given_e

    return sdp



def optimized_tree_search_sdp(model, D, d_value, evidence, threshold, partitions):
    """
    Highly optimized exact SDP computation using NumPy Vectorization. 
    Removes iterative dictionary loops to achieve C-level speeds.
    """
    inference = VariableElimination(model)
    #approximate inference
    #inference = ApproxInference(model)
    
    # Get opposite state for D
    d_states = model.get_cpds(D).state_names[D]
    d_index = d_states.index(d_value)
    not_d_value = d_states[1] if d_index == 0 else d_states[0]

    # 1. Compute Initial Log-Odds and Lambda
    initial_dist = inference.query(variables=[D], evidence=evidence, show_progress=False)
    p_d_e = initial_dist.get_value(**{D: d_value})
    p_not_d_e = initial_dist.get_value(**{D: not_d_value})
    
    print(f"Initial P(D|e): {p_d_e:.4f}, P(not D|e): {p_not_d_e:.4f}")
    
    log_O_d_e = math.log(p_d_e / p_not_d_e) if p_not_d_e > 0 else float('inf')
    lambda_threshold = math.log(threshold / (1 - threshold))
    current_decision_positive = (log_O_d_e >= lambda_threshold)
    
    # 2. Vectorized Precomputation of probabilities and weights
    partitions_data = []
    
    for s_i in partitions:
        # Query joint distribution of S_i given D and e
        # This computes the entire joint distribution table at once
        print(f"Precomputing for partition: {s_i}")
        dist_given_d = inference.query(variables=s_i, evidence={**evidence, D: d_value}, show_progress=False)
        print("computed query distribution given D")
        dist_given_not_d = inference.query(variables=s_i, evidence={**evidence, D: not_d_value}, show_progress=False)
        print("computed query distribution given not D")
        
        # Extract raw NumPy arrays and apply a small epsilon to prevent log(0)
        p_d_vals = np.maximum(dist_given_d.values.astype(np.float32), 1e-12)
        p_not_d_vals = np.maximum(dist_given_not_d.values.astype(np.float32), 1e-12)

        #mask = (p_d_vals > 1e-8) | (p_not_d_vals > 1e-8)
        #p_d_vals = p_d_vals[mask]
        #p_not_d_vals = p_not_d_vals[mask]
        
        # Vectorized calculation of weights for ALL states simultaneously
        w_vals = np.log(p_d_vals / p_not_d_vals)
        
        # Extract max and min bounds instantly
        max_w = np.max(w_vals)
        min_w = np.min(w_vals)
        
        # Flatten the arrays to 1D lists for lightning-fast iteration in the DFS
        partitions_data.append({
            'w_flat': w_vals.flatten(),
            'p_d_flat': p_d_vals.flatten(),
            'p_not_d_flat': p_not_d_vals.flatten(),
            'max_w': max_w,
            'min_w': min_w
        })

    # Sort partitions by max variance for optimal early pruning
    partitions_data.sort(key=lambda x: x['max_w'] - x['min_w'], reverse=True)

    # 3. Precompute Suffix Sums for O(1) bound lookups
    n_parts = len(partitions_data)
    suffix_max = [0.0] * (n_parts + 1)
    suffix_min = [0.0] * (n_parts + 1)
    
    for i in range(n_parts - 1, -1, -1):
        suffix_max[i] = suffix_max[i+1] + partitions_data[i]['max_w']
        suffix_min[i] = suffix_min[i+1] + partitions_data[i]['min_w']

    # 4. Lightning-Fast DFS
    def dfs(depth, current_log_odds, prob_cond_d, prob_cond_not_d):
        # O(1) Bound calculations
        upper_bound = current_log_odds + suffix_max[depth]
        lower_bound = current_log_odds + suffix_min[depth]
        
        # O(1) Probability calculation leveraging conditional independence
        def get_prob_q():
            return (p_d_e * prob_cond_d) + (p_not_d_e * prob_cond_not_d)

        # --- PRUNING LOGIC ---
        if current_decision_positive:
            if lower_bound >= lambda_threshold: return get_prob_q()
            if upper_bound < lambda_threshold: return 0.0
        else:
            if upper_bound < lambda_threshold: return get_prob_q()
            if lower_bound >= lambda_threshold: return 0.0
        
        # Leaf node evaluation
        if depth == n_parts:
            is_positive = current_log_odds >= lambda_threshold
            if is_positive == current_decision_positive:
                return get_prob_q()
            return 0.0
            
        # Recursive Step passing state as arguments
        total_sdp = 0.0
        part_data = partitions_data[depth]
        
        # Zip through the flat arrays. This replaces the slow dictionary mapping.
        for w, p_d, p_not_d in zip(part_data['w_flat'], part_data['p_d_flat'], part_data['p_not_d_flat']):
            # If the probability is effectively zero, skip the branch to save time
            if p_d < 1e-10 and p_not_d < 1e-10:
                continue
                
            total_sdp += dfs(
                depth + 1,
                current_log_odds + w,
                prob_cond_d * p_d,
                prob_cond_not_d * p_not_d
            )
            
        return total_sdp

    # Start DFS with base probabilities of 1.0 (multiplicative identity)
    return dfs(0, log_O_d_e, 1.0, 1.0)

def fast_broadcast_sdp_old_Wrong(model, D, d_value, evidence, threshold, partitions):
    inference = VariableElimination(model)
    
    d_states = model.get_cpds(D).state_names[D]
    d_index = d_states.index(d_value)
    not_d_value = d_states[1] if d_index == 0 else d_states[0]

    # 1. Compute Initial Log-Odds
    initial_dist = inference.query(variables=[D], evidence=evidence, show_progress=False)
    p_d_e = initial_dist.get_value(**{D: d_value})
    p_not_d_e = initial_dist.get_value(**{D: not_d_value})
    
    log_O_d_e = math.log(p_d_e / p_not_d_e) if p_not_d_e > 0 else float('inf')
    lambda_threshold = math.log(threshold / (1 - threshold))
    current_decision_positive = (log_O_d_e >= lambda_threshold)

    partitions_data = []
    
    for s_i in partitions:
        s_i_list = list(s_i)  # Lock the axis order for this partition
        
        # Get all CPDs that belong to this partition (contain any var in s_i)
        relevant_cpds = [cpd for cpd in model.get_cpds() if any(v in s_i_list for v in cpd.variables)]
        
        def get_joint_tensor(target_evidence):
            # 1. Grab all original factors from the network
            factors = [cpd.to_factor() for cpd in model.get_cpds()]
            
            # 2. Reduce by target evidence (D and E)
            for f in factors:
                overlap = [(v, target_evidence[v]) for v in f.variables if v in target_evidence]
                if overlap:
                    f.reduce(overlap, inplace=True)
                    
            # 3. Identify all "foreign" variables 
            vars_in_factors = set(v for f in factors for v in f.variables)
            foreign_vars = vars_in_factors - set(s_i_list)
            
            # 4. eliminate foreign variables using Exact Sum-Product VE
            for var in foreign_vars:
                f_with = [f for f in factors if var in f.variables]
                f_without = [f for f in factors if var not in f.variables]
                
                if f_with:
                    # Use .copy() and the '*' operator to safely multiply factors 
                    # without risking in-place NoneType returns.
                    prod = f_with[0].copy()
                    for f in f_with[1:]:
                        prod = prod * f
                    
                    prod.marginalize([var], inplace=True)
                    
                    # Unconditionally append.
                    f_without.append(prod)
                        
                factors = f_without
                
            # 5. Broadcast the remaining clean factors
            joint_prob = 1.0
            for factor in factors:
                if not factor.variables:
                    # Safely extract the scalar multiplier (np.sum perfectly handles 0-d arrays)
                    joint_prob *= float(np.sum(factor.values))
                    continue
                    
                f_vars = factor.variables
                expanded_vals = factor.values
                
                # Expand missing dimensions
                for _ in range(len(s_i_list) - len(f_vars)):
                    expanded_vals = np.expand_dims(expanded_vals, -1)
                    
                # Align axes to the master s_i_list order
                transpose_order = []
                none_idx = len(f_vars)
                for var in s_i_list:
                    if var in f_vars:
                        transpose_order.append(f_vars.index(var))
                    else:
                        transpose_order.append(none_idx)
                        none_idx += 1
                        
                aligned_vals = np.transpose(expanded_vals, transpose_order)
                joint_prob = joint_prob * aligned_vals
                #print(f"Processed factor with variables {f_vars}, current joint shape: {joint_prob.shape}")
            #print(joint_prob)
            return joint_prob

        # --- Evaluate all states simultaneously ---
        joint_d = get_joint_tensor({**evidence, D: d_value})
        joint_not_d = get_joint_tensor({**evidence, D: not_d_value})
        
        # Normalize to convert joint probabilities to conditional probabilities P(S_i | D, e)
        sum_d = np.sum(joint_d)
        sum_not_d = np.sum(joint_not_d)
        
        sum_d = sum_d if sum_d > 0 else 1.0
        sum_not_d = sum_not_d if sum_not_d > 0 else 1.0
        
        p_d_tensor = np.maximum(joint_d / sum_d, 1e-12)
        p_not_d_tensor = np.maximum(joint_not_d / sum_not_d, 1e-12)
        
        # Calculate Log-Odds Weights
        w_tensor = np.log(p_d_tensor / p_not_d_tensor)
        
        # .flatten() unravels the N-dimensional tensor into a 1D list in the exact 
        # same order that itertools.product would have generated.
        partitions_data.append({
            'w_flat': w_tensor.flatten().tolist(),
            'p_d_flat': p_d_tensor.flatten().tolist(),
            'p_not_d_flat': p_not_d_tensor.flatten().tolist(),
            'max_w': np.max(w_tensor),
            'min_w': np.min(w_tensor)
        })

    # Sort partitions by max variance for optimal early pruning
    partitions_data.sort(key=lambda x: x['max_w'] - x['min_w'], reverse=True)

    # Precompute Suffix Sums
    n_parts = len(partitions_data)
    suffix_max = [0.0] * (n_parts + 1)
    suffix_min = [0.0] * (n_parts + 1)
    for i in range(n_parts - 1, -1, -1):
        suffix_max[i] = suffix_max[i+1] + partitions_data[i]['max_w']
        suffix_min[i] = suffix_min[i+1] + partitions_data[i]['min_w']

    # DFS Loop
    def dfs(depth, current_log_odds, prob_cond_d, prob_cond_not_d):
        upper_bound = current_log_odds + suffix_max[depth]
        lower_bound = current_log_odds + suffix_min[depth]
        
        def get_prob_q():
            return (p_d_e * prob_cond_d) + (p_not_d_e * prob_cond_not_d)

        if current_decision_positive:
            if lower_bound >= lambda_threshold: return get_prob_q()
            if upper_bound < lambda_threshold: return 0.0
        else:
            if upper_bound < lambda_threshold: return get_prob_q()
            if lower_bound >= lambda_threshold: return 0.0
        
        if depth == n_parts:
            is_positive = current_log_odds >= lambda_threshold
            if is_positive == current_decision_positive:
                return get_prob_q()
            return 0.0
            
        total_sdp = 0.0
        part_data = partitions_data[depth]
        
        for w, p_d, p_not_d in zip(part_data['w_flat'], part_data['p_d_flat'], part_data['p_not_d_flat']):
            if p_d < 1e-10 and p_not_d < 1e-10: continue
            total_sdp += dfs(depth + 1, current_log_odds + w, prob_cond_d * p_d, prob_cond_not_d * p_not_d)
            
        return total_sdp

    return dfs(0, log_O_d_e, 1.0, 1.0)

def fast_broadcast_sdp(model, D, d_value, evidence, threshold, partitions):
    inference = VariableElimination(model)
    
    d_states = model.get_cpds(D).state_names[D]
    d_index = d_states.index(d_value)
    not_d_value = d_states[1] if d_index == 0 else d_states[0]

    # 1. Compute Initial Log-Odds
    initial_dist = inference.query(variables=[D], evidence=evidence, show_progress=False)
    p_d_e = initial_dist.get_value(**{D: d_value})
    p_not_d_e = initial_dist.get_value(**{D: not_d_value})
    
    log_O_d_e = math.log(p_d_e / p_not_d_e) if p_not_d_e > 0 else float('inf')
    lambda_threshold = math.log(threshold / (1 - threshold))
    current_decision_positive = (log_O_d_e >= lambda_threshold)

    partitions_data = []
    
    for s_i in partitions:
        s_i_list = list(s_i)  # Lock the axis order for this partition
        
        # Get all CPDs that belong to this partition (contain any var in s_i)
        #relevant_cpds = [cpd for cpd in model.get_cpds() if any(v in s_i_list for v in cpd.variables)]
        
        def get_joint_tensor(target_evidence):
            # 1. Grab all original factors from the network
            factors = [cpd.to_factor() for cpd in model.get_cpds() if any(v in s_i_list for v in cpd.variables)]
            
            # 2. Reduce by target evidence (D and E)
            for f in factors:
                overlap = [(v, target_evidence[v]) for v in f.variables if v in target_evidence]
                if overlap:
                    f.reduce(overlap, inplace=True)
                    
            # 3. Identify all "foreign" variables 
            vars_in_factors = set(v for f in factors for v in f.variables)
            foreign_vars = vars_in_factors - set(s_i_list)
            
            # 4. Eliminate foreign variables using Exact Sum-Product VE
            for var in foreign_vars:
                f_with = [f for f in factors if var in f.variables]
                f_without = [f for f in factors if var not in f.variables]
                
                if f_with:
                    # Use .copy() and the '*' operator to safely multiply factors 
                    # without risking in-place NoneType returns.
                    prod = f_with[0].copy()
                    for f in f_with[1:]:
                        prod = prod * f
                    
                    prod.marginalize([var], inplace=True)
                    
                    # Unconditionally append.
                    f_without.append(prod)
                        
                factors = f_without
                
            # 5. Broadcast the remaining clean factors
            joint_prob = 1.0
            for factor in factors:
                if not factor.variables:
                    # Safely extract the scalar multiplier (np.sum perfectly handles 0-d arrays)
                    joint_prob *= float(np.sum(factor.values))
                    continue
                    
                f_vars = factor.variables
                expanded_vals = factor.values
                
                # Expand missing dimensions
                for _ in range(len(s_i_list) - len(f_vars)):
                    expanded_vals = np.expand_dims(expanded_vals, -1)
                    
                # Align axes to the master s_i_list order
                transpose_order = []
                none_idx = len(f_vars)
                for var in s_i_list:
                    if var in f_vars:
                        transpose_order.append(f_vars.index(var))
                    else:
                        transpose_order.append(none_idx)
                        none_idx += 1
                        
                aligned_vals = np.transpose(expanded_vals, transpose_order)
                joint_prob = joint_prob * aligned_vals
                
            return joint_prob

        # --- Evaluate all states simultaneously ---
        joint_d = get_joint_tensor({**evidence, D: d_value})
        joint_not_d = get_joint_tensor({**evidence, D: not_d_value})
        
        # Normalize to convert joint probabilities to conditional probabilities P(S_i | D, e)
        sum_d = np.sum(joint_d)
        sum_not_d = np.sum(joint_not_d)
        
        sum_d = sum_d if sum_d > 0 else 1.0
        sum_not_d = sum_not_d if sum_not_d > 0 else 1.0
        
        p_d_tensor = np.maximum(joint_d / sum_d, 1e-12)
        p_not_d_tensor = np.maximum(joint_not_d / sum_not_d, 1e-12)
        
        # Calculate Log-Odds Weights
        w_tensor = np.log(p_d_tensor / p_not_d_tensor)
        
        # .flatten() unravels the N-dimensional tensor into a 1D list in the exact 
        # same order that itertools.product would have generated.
        partitions_data.append({
            'w_flat': w_tensor.flatten().tolist(),
            'p_d_flat': p_d_tensor.flatten().tolist(),
            'p_not_d_flat': p_not_d_tensor.flatten().tolist(),
            'max_w': np.max(w_tensor),
            'min_w': np.min(w_tensor)
        })

    # Sort partitions by max variance for optimal early pruning
    partitions_data.sort(key=lambda x: x['max_w'] - x['min_w'], reverse=True)

    # Precompute Suffix Sums
    n_parts = len(partitions_data)
    suffix_max = [0.0] * (n_parts + 1)
    suffix_min = [0.0] * (n_parts + 1)
    for i in range(n_parts - 1, -1, -1):
        suffix_max[i] = suffix_max[i+1] + partitions_data[i]['max_w']
        suffix_min[i] = suffix_min[i+1] + partitions_data[i]['min_w']

    # DFS Loop
    def dfs(depth, current_log_odds, prob_cond_d, prob_cond_not_d):
        upper_bound = current_log_odds + suffix_max[depth]
        lower_bound = current_log_odds + suffix_min[depth]
        
        def get_prob_q():
            return (p_d_e * prob_cond_d) + (p_not_d_e * prob_cond_not_d)

        if current_decision_positive:
            if lower_bound >= lambda_threshold: return get_prob_q()
            if upper_bound < lambda_threshold: return 0.0
        else:
            if upper_bound < lambda_threshold: return get_prob_q()
            if lower_bound >= lambda_threshold: return 0.0
        
        if depth == n_parts:
            is_positive = current_log_odds >= lambda_threshold
            if is_positive == current_decision_positive:
                return get_prob_q()
            return 0.0
            
        total_sdp = 0.0
        part_data = partitions_data[depth]
        
        for w, p_d, p_not_d in zip(part_data['w_flat'], part_data['p_d_flat'], part_data['p_not_d_flat']):
            if p_d < 1e-10 and p_not_d < 1e-10: continue
            total_sdp += dfs(depth + 1, current_log_odds + w, prob_cond_d * p_d, prob_cond_not_d * p_not_d)
            
        return total_sdp

    return dfs(0, log_O_d_e, 1.0, 1.0)