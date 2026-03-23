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


def generate_patient_for_target_sdp(bn, target_node, target_value, decision_threshold, target_sdp, evidence_vars, tolerance=0.05, max_steps=1000):
    """
    Uses Stochastic Hill Climbing to mutate a patient's symptoms until their exact SDP matches the target.
    Includes memory/einsum safety nets and ancestral graph pruning for massive speedups.
    """
    print(f"\n--- Hunting for Patient with SDP ≈ {target_sdp} ---")
    
    # ==========================================
    # OPTIMIZATION: Barren Node Pruning
    # Create a tiny, ultra-fast sub-model for checking the Base Decision anchor
    # ==========================================
    relevant_nodes = list(evidence_vars) + [target_node]
    ancestral_structure = bn.get_ancestral_graph(relevant_nodes)
    
    sub_model = BayesianNetwork(ancestral_structure.edges())
    sub_model.add_nodes_from(ancestral_structure.nodes())
    
    for node in sub_model.nodes():
        sub_model.add_cpds(bn.get_cpds(node))
        
    inference = VariableElimination(sub_model)
    # ==========================================
    
    # 1. Start with a random patient THAT MEETS THE BASE DECISION THRESHOLD
    current_patient = None
    attempts = 0
    while current_patient is None and attempts < 1000:
        temp_patient = {}
        for var in evidence_vars:
            states = sub_model.get_cpds(var).state_names[var]
            temp_patient[var] = random.choice(states)
            
        # --- SAFETY NET 1: Catch Memory Limits on Base Decision ---
        try:
            base_dist = inference.query(variables=[target_node], evidence=temp_patient, show_progress=False)
        except (ValueError, MemoryError):
            print(f"    [!] EXACT INFERENCE IMPOSSIBLE: Sub-network treewidth exceeds hardware limits. Skipping network.")
            return None, None
            
        if base_dist.get_value(**{target_node: target_value}) >= decision_threshold:
            current_patient = temp_patient
        attempts += 1
        
    if current_patient is None:
        print("    [!] Could not find a valid positive starting seed.")
        return None, None
        
    # Get initial SDP and Error (Pass the FULL 'bn' to fast_broadcast_sdp!)
    hidden_vars = [v for v in bn.nodes() if v not in current_patient and v != target_node]
    partitions = get_partitions(bn, hidden_vars, target_node, current_patient)
    
    # --- SAFETY NET 2: Catch Tensor Explosions in Exact SDP ---
    try:
        current_sdp = fast_broadcast_sdp(bn, target_node, target_value, current_patient, decision_threshold, partitions)
    except (ValueError, MemoryError):
        print(f"    [!] EXACT SDP IMPOSSIBLE: Tensor exploded during exact calculation. Skipping network.")
        return None, None
        
    current_error = abs(current_sdp - target_sdp)
    print(f"Starting random patient SDP: {current_sdp:.4f} (Error: {current_error:.4f})")
    
    # 2. Begin Hill Climbing
    step = 0
    while current_error > tolerance and step < max_steps:
        step += 1
        
        var_to_mutate = random.choice(evidence_vars)
        possible_states = sub_model.get_cpds(var_to_mutate).state_names[var_to_mutate]
        old_state = current_patient[var_to_mutate]
        
        new_state = random.choice([s for s in possible_states if s != old_state])
        
        proposed_patient = current_patient.copy()
        proposed_patient[var_to_mutate] = new_state

        # --- STRICT ANCHOR CHECK (with Safety Net) ---
        # Ensure the mutation didn't flip the base decision to negative!
        try:
            base_dist = inference.query(variables=[target_node], evidence=proposed_patient, show_progress=False)
        except (ValueError, MemoryError):
            continue # If a specific mutation somehow triggers a memory explosion, just reject it
            
        if base_dist.get_value(**{target_node: target_value}) < decision_threshold:
            continue 
        
        # --- Calculate new Exact SDP (with Safety Net) ---
        partitions = get_partitions(bn, hidden_vars, target_node, proposed_patient)
        try:
            proposed_sdp = fast_broadcast_sdp(bn, target_node, target_value, proposed_patient, decision_threshold, partitions)
        except (ValueError, MemoryError):
            continue # Reject mutations that cause tensor explosions
            
        proposed_error = abs(proposed_sdp - target_sdp)
        
        # 3. Acceptance Logic: Keep if it moves us closer to the target SDP
        if proposed_error < current_error:
            current_patient = proposed_patient
            current_sdp = proposed_sdp
            current_error = proposed_error
            #print(f"Step {step}: Mutated '{var_to_mutate}' -> SDP improved to {current_sdp:.4f} (Error: {current_error:.4f})")
            
    if current_error <= tolerance:
        print(f"SUCCESS! Found patient matching target {target_sdp} (Actual: {current_sdp:.4f})")
        return current_patient, current_sdp
    else:
        print(f"Failed to converge within {max_steps} steps. Closest was {current_sdp:.4f}.")
        return None, None

def build_experimental_dataset(bn, target_node, target_value, decision_threshold, evidence_vars, max_attempts=5, buckets=[0.6, 0.75, 0.85, 1]):
    all_vars = list(bn.nodes())
    all_vars.remove(target_node)
    
    
    found_patients = {}
    
    for target in buckets:
        print(f"\n=== Generating patient for target SDP bucket: {target} ===")
        patient, sdp, final_target_val = None, None, None
        attempts = 0
        
        # Keep trying until the hill climber succeeds (re-seeds if it gets stuck in a local minimum)
        while patient is None and attempts < max_attempts:
            patient, sdp = generate_patient_for_target_sdp(bn, target_node, target_value, decision_threshold, target, evidence_vars)
            attempts += 1
            
        if patient is not None:
            # Save the final target_val so MCMC knows what to test against!
            found_patients[target] = {
                'evidence': patient, 
                'true_sdp': sdp
            }
            
    return found_patients

def perfect_monte_carlo_sdp_estimation(bn, target, target_value, patient, threshold, n_samples=1000):
    '''
    Function used for debugging purposes only. It computes the exact generator distribution, and thus does
    not function as a real monte carlo
    '''
    inference = VariableElimination(bn)
    hidden_vars = [node for node in bn.nodes() if node not in patient and node != target]
    
    # 1. Get the EXACT joint distribution of H given e
    h_dist = inference.query(variables=hidden_vars, evidence=patient, show_progress=False)
    
    # 2. Extract all valid realities and their true probabilities
    h_states_lists = [h_dist.state_names[var] for var in hidden_vars]
    all_h_combos = list(itertools.product(*h_states_lists))
    
    valid_combos = []
    probs = []
    
    for combo in all_h_combos:
        combo_dict = dict(zip(hidden_vars, combo))
        p = h_dist.get_value(**combo_dict)
        if p > 0:  # Ignore impossible realities!
            valid_combos.append(combo_dict)
            probs.append(p)
            
    # Normalize probabilities for the numpy sampler
    probs = np.array(probs)
    probs /= probs.sum()
    
    # 3. Draw N PERFECT samples using the true distribution
    sampled_indices = np.random.choice(len(valid_combos), size=n_samples, p=probs, replace=True)
    
    # 4. Evaluate the Decision Boundary with a Cache
    count_same_decision = 0
    decision_cache = {}
    
    for idx in sampled_indices:
        sample_h = valid_combos[idx]
        patient_id = tuple(sample_h.items())
        
        # Check cache first
        if patient_id in decision_cache:
            makes_same = decision_cache[patient_id]
        else:
            sample_evidence = patient.copy()
            sample_evidence.update(sample_h)
            prob_dist = inference.query(variables=[target], evidence=sample_evidence, show_progress=False)
            makes_same = prob_dist.get_value(**{target: target_value}) >= threshold
            decision_cache[patient_id] = makes_same
            
        if makes_same:
            count_same_decision += 1
            
    # Pure, unweighted mean!
    return count_same_decision / n_samples

from pgmpy.sampling import BayesianModelSampling
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import State

def monte_carlo_sdp_estimation(bn, target, target_value, patient, threshold, n_samples=1000):
    '''
    Uses the likelihood sampler from pgmpy to draw samples from the distribution. Has shown to be biased.
    '''
    sampler = BayesianModelSampling(bn)
    
    # 1. pgmpy requires evidence to be a list of State objects
    evidence_states = [State(var, state) for var, state in patient.items()]
    
    # 2. Draw the weighted samples
    print(f"Drawing {n_samples} samples...")
    samples = sampler.likelihood_weighted_sample(size=n_samples, 
                                                 evidence=evidence_states, 
                                                 show_progress=False)

    inference = VariableElimination(bn)
    
    weighted_same_decision = 0.0
    total_weight = 0.0
    
    print("Evaluating decision boundary for samples...")
    num_agreements = 0
    for _, sample in samples.iterrows():
        # 3. Extract the likelihood weight
        weight = sample['_weight']
        total_weight += weight
        
        # 4. Remove '_weight' so pgmpy doesn't crash during the query
        sample_evidence = {k: v for k, v in sample.to_dict().items() if k != '_weight'}
        # remove target variable from evidence
        if target in sample_evidence:
            del sample_evidence[target]

        # 5. O(1) exact inference (very fast because all nodes are observed!)
        prob_dist = inference.query(variables=[target], evidence=sample_evidence, show_progress=False)
        
        # 6. Add the weight if the decision threshold is met
        if prob_dist.get_value(**{target: target_value}) >= threshold:
            weighted_same_decision += weight
            num_agreements += 1
            
    # 7. The final SDP is the weighted fraction of samples that kept the same decision
    estimated_sdp = weighted_same_decision / total_weight
    
    # not weighted version (just for sanity check)
    #estimated_sdp = num_agreements / len(samples)

    return estimated_sdp



'''
Markov Chain Monte Carlo (Metropolis Hastings). This works better!
'''

def calculate_log_joint(bn, full_state):
    """Calculates log P(H, E, Target) instantly by multiplying CPDs."""
    log_p = 0.0
    for cpd in bn.get_cpds():
        # Extract only the variables needed for this specific CPD lookup
        cpd_args = {v: full_state[v] for v in cpd.variables}
        p = cpd.get_value(**cpd_args)
        if p == 0:
            return float('-inf')  # This is a physically impossible patient reality
        log_p += math.log(p)
    return log_p

def calculate_unnormalized_posterior(bn, h_dict, e_dict, target):
    """Calculates log P(H, E) by summing the joint across the possible Target states."""
    target_states = bn.get_cpds(target).state_names[target]
    total_p = 0.0
    
    for t_state in target_states:
        full_state = {**h_dict, **e_dict, target: t_state}
        log_j = calculate_log_joint(bn, full_state)
        if log_j != float('-inf'):
            total_p += math.exp(log_j)
            
    return math.log(total_p) if total_p > 0 else float('-inf')

def mcmc_sdp_estimation(bn, target, target_value, patient, threshold, n_samples=11000, burn_in=1000, thinning=10):
    hidden_vars = [node for node in bn.nodes() if node not in patient and node != target]
    
    # 1. SEED THE CHAIN: Use Likelihood Weighting just to find ONE physically possible patient.
    sampler = BayesianModelSampling(bn)
    evidence_states = [State(var, state) for var, state in patient.items()]
    valid_seed_found = False
    
    while not valid_seed_found:
        seed_samples = sampler.likelihood_weighted_sample(size=100, evidence=evidence_states, show_progress=False)
        valid_seeds = seed_samples[seed_samples['_weight'] > 0] # Filter out impossible realities
        if not valid_seeds.empty:
            best_seed = valid_seeds.sort_values('_weight', ascending=False).iloc[0]
            current_h = {var: best_seed[var] for var in hidden_vars}
            valid_seed_found = True
            
    # Calculate starting probability
    current_log_p = calculate_unnormalized_posterior(bn, current_h, patient, target)
    
    # 2. RUN THE METROPOLIS-HASTINGS CHAIN
    total_iterations = burn_in + (n_samples * thinning)
    accepted_samples = []
    
    for i in range(total_iterations):
        # Propose a new reality by flipping ONE random hidden variable
        var_to_flip = random.choice(hidden_vars)
        possible_states = bn.get_cpds(var_to_flip).state_names[var_to_flip]
        current_state_val = current_h[var_to_flip]
        
        new_state_val = random.choice([s for s in possible_states if s != current_state_val])
        
        proposed_h = current_h.copy()
        proposed_h[var_to_flip] = new_state_val
        
        # Evaluate proposed reality
        proposed_log_p = calculate_unnormalized_posterior(bn, proposed_h, patient, target)
        
        # Metropolis-Hastings Acceptance Criterion: log(alpha) = log_P(new) - log_P(old)
        log_alpha = proposed_log_p - current_log_p
        
        accept = False
        if log_alpha >= 0:
            accept = True
        elif proposed_log_p != float('-inf'):
            if math.log(random.uniform(0, 1)) < log_alpha:
                accept = True
                
        if accept:
            current_h = proposed_h
            current_log_p = proposed_log_p
            
        # Save independent samples
        if i >= burn_in and (i - burn_in) % thinning == 0:
            accepted_samples.append(current_h.copy())
            
    # 3. EVALUATE THE DECISION BOUNDARY
    inference = VariableElimination(bn)
    count_same_decision = 0
    decision_cache = {}
    
    for sample_h in accepted_samples:
        patient_id = tuple(sample_h.items())
        
        if patient_id in decision_cache:
            makes_same = decision_cache[patient_id]
        else:
            full_evidence = {**patient, **sample_h}
            prob_dist = inference.query(variables=[target], evidence=full_evidence, show_progress=False)
            makes_same = prob_dist.get_value(**{target: target_value}) >= threshold
            decision_cache[patient_id] = makes_same
            
        if makes_same:
            count_same_decision += 1
            
    return count_same_decision / len(accepted_samples)


def get_exact_target_posterior_O1(bn, target, target_value, full_state):
    """
    Calculates P(Target | All Other Nodes) in O(1) time without Variable Elimination.
    Relies purely on the Target's Markov Blanket (its own CPD and its children's CPDs).
    100% immune to treewidth and einsum dimension limits.
    """
    target_states = bn.get_cpds(target).state_names[target]
    log_probs = {}
    
    # The ONLY CPDs that change depending on the Target's state:
    relevant_nodes = [target] + list(bn.get_children(target))
    
    for state in target_states:
        # Test what happens if the Target takes this state
        test_state = {**full_state, target: state}
        log_p = 0.0
        possible = True
        
        for node in relevant_nodes:
            cpd = bn.get_cpds(node)
            # Extract only the variables this specific CPD needs
            cpd_args = {v: test_state[v] for v in cpd.variables}
            prob = cpd.get_value(**cpd_args)
            
            if prob == 0.0:
                possible = False
                break
            log_p += math.log(prob)
            
        if possible:
            log_probs[state] = log_p
        else:
            log_probs[state] = float('-inf')
            
    # Log-Sum-Exp to normalize and get the exact probability
    valid_log_probs = [lp for lp in log_probs.values() if lp != float('-inf')]
    if not valid_log_probs:
        return 0.0 # Mathematically impossible state
        
    max_log = max(valid_log_probs)
    total_p = sum(math.exp(lp - max_log) for lp in valid_log_probs)
    
    # Return the normalized probability for the specific target_value
    target_lp = log_probs.get(target_value, float('-inf'))
    if target_lp == float('-inf'):
        return 0.0
        
    return math.exp(target_lp - max_log) / total_p

def fast_mcmc_sdp_estimation(bn, target, target_value, patient, threshold, n_samples=11000, burn_in=1000, thinning=10):
    hidden_vars = [node for node in bn.nodes() if node not in patient and node != target]
    
    # 1. SEED THE CHAIN: Use Likelihood Weighting just to find ONE physically possible patient.
    sampler = BayesianModelSampling(bn)
    evidence_states = [State(var, state) for var, state in patient.items()]
    valid_seed_found = False
    
    while not valid_seed_found:
        seed_samples = sampler.likelihood_weighted_sample(size=100, evidence=evidence_states, show_progress=False)
        valid_seeds = seed_samples[seed_samples['_weight'] > 0] # Filter out impossible realities
        if not valid_seeds.empty:
            best_seed = valid_seeds.sort_values('_weight', ascending=False).iloc[0]
            current_h = {var: best_seed[var] for var in hidden_vars}
            valid_seed_found = True
            
    # Calculate starting probability
    current_log_p = calculate_unnormalized_posterior(bn, current_h, patient, target)
    
    # 2. RUN THE METROPOLIS-HASTINGS CHAIN
    total_iterations = burn_in + (n_samples * thinning)
    accepted_samples = []
    
    for i in range(total_iterations):
        # Propose a new reality by flipping ONE random hidden variable
        var_to_flip = random.choice(hidden_vars)
        possible_states = bn.get_cpds(var_to_flip).state_names[var_to_flip]
        current_state_val = current_h[var_to_flip]
        
        new_state_val = random.choice([s for s in possible_states if s != current_state_val])
        
        proposed_h = current_h.copy()
        proposed_h[var_to_flip] = new_state_val
        
        # Evaluate proposed reality
        proposed_log_p = calculate_unnormalized_posterior(bn, proposed_h, patient, target)
        
        # Metropolis-Hastings Acceptance Criterion: log(alpha) = log_P(new) - log_P(old)
        log_alpha = proposed_log_p - current_log_p
        
        accept = False
        if log_alpha >= 0:
            accept = True
        elif proposed_log_p != float('-inf'):
            if math.log(random.uniform(0, 1)) < log_alpha:
                accept = True
                
        if accept:
            current_h = proposed_h
            current_log_p = proposed_log_p
            
        # Save independent samples
        if i >= burn_in and (i - burn_in) % thinning == 0:
            accepted_samples.append(current_h.copy())
            
    # 3. EVALUATE THE DECISION BOUNDARY
    inference = VariableElimination(bn)
    count_same_decision = 0
    decision_cache = {}
    
    for sample_h in accepted_samples:
        patient_id = tuple(sample_h.items())
        if patient_id in decision_cache:
            makes_same = decision_cache[patient_id]
        else:
            # Combine evidence and the MCMC hidden sample
            full_evidence = {**patient, **sample_h}
            
            # Use the O(1) Markov Blanket evaluator!
            p_target = get_exact_target_posterior_O1(bn, target, target_value, full_evidence)
            makes_same = p_target >= threshold
            
            decision_cache[patient_id] = makes_same
            
        if makes_same:
            count_same_decision += 1
            
    return count_same_decision / len(accepted_samples)