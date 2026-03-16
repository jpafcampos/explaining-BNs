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

def find_exact_experimental_patients(bn, target, target_value, threshold, hidden_size=10, buckets=[0.15, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]):
    sampler = BayesianModelSampling(bn)
    inference = VariableElimination(bn)
    
    found_patients = {}
    all_vars = list(bn.nodes())
    all_vars.remove(target)
    
    evidence_size = len(all_vars) - hidden_size
    print(f"Hunting for patients... (Locking {evidence_size} variables as evidence)")
    
    batch_size = 5000  # Generate 5000 patients at a time to bypass pgmpy overhead
    
    while len(found_patients) < len(buckets):
        print(f"Generating a batch of {batch_size} random realities...")
        batch_samples = sampler.forward_sample(size=batch_size, show_progress=False)
        
        for _, random_state in batch_samples.iterrows():
            if len(found_patients) == len(buckets):
                break  # Break out if we filled the last bucket!
            
            # Lock a large chunk of the network as evidence
            evidence_vars = random.sample(all_vars, evidence_size)
            patient_evidence = {var: random_state[var] for var in evidence_vars}
            
            # Ensure the base decision is positive
            base_dist = inference.query(variables=[target], evidence=patient_evidence, show_progress=False)
            if base_dist.get_value(**{target: target_value}) < threshold:
                continue
                
            # Get partitions
            hidden_vars = [v for v in bn.nodes() if v not in patient_evidence and v != target]
            partitions = get_partitions(bn, hidden_vars, target, patient_evidence)    
            
            # Calculate the Absolute Exact SDP
            true_sdp = fast_broadcast_sdp(bn, target, target_value, patient_evidence, threshold, partitions)
            print(f"--> Found patient with Exact SDP: {true_sdp:.4f}")
            # Check if it fits an EMPTY bucket (+/- 0.05 tolerance)
            for bucket in buckets:
                if bucket not in found_patients and abs(true_sdp - bucket) <= 0.05:
                    print(f"--> Filled bucket {bucket} with Exact SDP: {true_sdp:.4f}")
                    found_patients[bucket] = {
                        'evidence': patient_evidence,
                        'true_sdp': true_sdp
                    }
                    break
                    
    print("All buckets filled successfully!")
    return found_patients


def perfect_monte_carlo_sdp_estimation(bn, target, target_value, patient, threshold, n_samples=1000):
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


def monte_carlo_sdp_rejection_sampling(bn,target, target_value,patient,threshold,n_samples=10000):

    H = [v for v in bn.nodes() if v not in patient and v != target]

    sampler = BayesianModelSampling(bn)
    inference = VariableElimination(bn)

    evidence_states = [State(v, s) for v, s in patient.items()]

    samples = sampler.rejection_sample(
        size=n_samples,
        evidence=evidence_states,
        show_progress=False
    )

    agreements = 0

    for _, sample in samples.iterrows():

        h_dict = {var: sample[var] for var in H}
        evidence = {**patient, **h_dict}

        prob = inference.query(
            variables=[target],
            evidence=evidence,
            show_progress=False
        ).get_value(**{target: target_value})

        if prob >= threshold:
            agreements += 1

    return agreements / len(samples)



'''
Markov Chain Monte Carlo (Metropolis Hastings)
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

def mcmc_sdp_estimation(bn, target, target_value, patient, threshold, n_samples=2000, burn_in=500, thinning=5):
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
    # These are true samples, so we just use the normal, unweighted mean!
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