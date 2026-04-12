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

def harvest_patients_for_all_buckets(bn, target_node, target_value, decision_threshold, evidence_vars, target_buckets, tolerance=0.05, max_restarts=100, max_steps_per_restart=800):
    """
    Wanders the probability landscape using Stochastic Hill Climbing.
    If it gets trapped in a local minimum, it triggers a Random Restart.
    Harvests any patient that fits an empty bucket along the way.
    """
    print(f"\n--- Starting Hill Climbing Harvest for Buckets: {target_buckets} ---")
    
    # 1. Setup Pruned Sub-model for fast base-decision checks
    relevant_nodes = list(evidence_vars) + [target_node]
    ancestral_structure = bn.get_ancestral_graph(relevant_nodes)
    sub_model = BayesianNetwork(ancestral_structure.edges())
    sub_model.add_nodes_from(ancestral_structure.nodes())
    for node in sub_model.nodes():
        sub_model.add_cpds(bn.get_cpds(node))
    inference = VariableElimination(sub_model)
    
    unfilled_buckets = {bucket: None for bucket in target_buckets}
    
    # ========================================================
    # RANDOM RESTART LOOP
    # ========================================================
    for restart in range(max_restarts):
        empty_targets = [b for b, v in unfilled_buckets.items() if v is None]
        if not empty_targets:
            break # We filled them all!
            
        # 1. Find a Valid Starting Seed for this climb
        current_patient = None
        attempts = 0
        while current_patient is None and attempts < 1000:
            temp_patient = {var: random.choice(sub_model.get_cpds(var).state_names[var]) for var in evidence_vars}
            try:
                base_dist = inference.query(variables=[target_node], evidence=temp_patient, show_progress=False)
                if base_dist.get_value(**{target_node: target_value}) >= decision_threshold:
                    current_patient = temp_patient
            except (ValueError, MemoryError):
                return unfilled_buckets # Fast fail on impossible networks
            attempts += 1
            
        if current_patient is None:
            continue
            
        hidden_vars = [v for v in bn.nodes() if v not in current_patient and v != target_node]
        partitions = get_partitions(bn, hidden_vars, target_node, current_patient)
        
        try:
            current_sdp = fast_broadcast_sdp(bn, target_node, target_value, current_patient, decision_threshold, partitions)
        except (ValueError, MemoryError):
            return unfilled_buckets
            
        # Check if the random seed filled anything!
        for bucket in empty_targets:
            if abs(current_sdp - bucket) <= tolerance:
                unfilled_buckets[bucket] = (current_patient.copy(), current_sdp)
                print(f"    [+] INSTANT HARVEST (Restart {restart}): Filled bucket {bucket} with SDP {current_sdp:.4f}!")
                empty_targets = [b for b, v in unfilled_buckets.items() if v is None]
                
        if not empty_targets:
            break
            
        # ========================================================
        # HILL CLIMBING LOOP
        # ========================================================
        # We track patience. If we reject X mutations in a row, we are stuck.
        patience = len(evidence_vars) 
        stuck_counter = 0
        
        for step in range(max_steps_per_restart):
            if not empty_targets:
                break
                
            # Gravity: Pull toward the nearest empty bucket
            active_target = min(empty_targets, key=lambda b: abs(current_sdp - b))
            current_error = abs(current_sdp - active_target)
            
            # Mutate 1 random variable (Stochastic HC)
            var_to_mutate = random.choice(evidence_vars)
            possible_states = sub_model.get_cpds(var_to_mutate).state_names[var_to_mutate]
            
            proposed_patient = current_patient.copy()
            proposed_patient[var_to_mutate] = random.choice([s for s in possible_states if s != proposed_patient[var_to_mutate]])

            # Base Anchor Check
            try:
                base_dist = inference.query(variables=[target_node], evidence=proposed_patient, show_progress=False)
                if base_dist.get_value(**{target_node: target_value}) < decision_threshold:
                    stuck_counter += 1
                    if stuck_counter >= patience: break # Local minimum reached
                    continue 
            except (ValueError, MemoryError):
                stuck_counter += 1
                continue 
                
            # Evaluate Exact SDP
            partitions = get_partitions(bn, hidden_vars, target_node, proposed_patient)
            try:
                proposed_sdp = fast_broadcast_sdp(bn, target_node, target_value, proposed_patient, decision_threshold, partitions)
            except (ValueError, MemoryError):
                stuck_counter += 1
                continue 
                
            # --- THE HARVEST CHECK ---
            for bucket in empty_targets:
                if abs(proposed_sdp - bucket) <= tolerance:
                    unfilled_buckets[bucket] = (proposed_patient.copy(), proposed_sdp)
                    print(f"    [+] HARVEST SUCCESS (Restart {restart}, Step {step}): Filled bucket {bucket} (Exact SDP: {proposed_sdp:.4f})!")
                    empty_targets = [b for b, v in unfilled_buckets.items() if v is None]
            
            if not empty_targets:
                break
                
            # --- STRICT GREEDY ACCEPTANCE ---
            proposed_error = abs(proposed_sdp - active_target)
            
            if proposed_error < current_error:
                # We moved closer! Accept and reset the stuck counter.
                current_patient = proposed_patient
                current_sdp = proposed_sdp
                stuck_counter = 0
            else:
                # We got worse (or hit a flat plateau). Reject it.
                stuck_counter += 1
                
            # If we reject too many times in a row, we are trapped
            # Break the inner loop to trigger a Random Restart!
            if stuck_counter >= patience:
                # print(f"      -> Trapped in local minimum at SDP {current_sdp:.4f}. Restarting...")
                break 
                
    remaining = [b for b, v in unfilled_buckets.items() if v is None]
    if remaining:
        print(f"--- Harvest Complete. Could not fill buckets: {remaining} ---")
    else:
        print(f"--- Harvest Complete. All buckets filled successfully! ---")
        
    return unfilled_buckets


def find_exact_experimental_patients(bn, target_node, target_value, decision_threshold, evidence_vars, buckets=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], tolerance=0.05, batch_size=8000, max_batches=2):
    """
    Brute-force searches for patients with locked evidence set
    Evaluates each random reality and assigns it to a bucket if the exact SDP matches.
    Returns a dictionary mapping buckets to a tuple: (patient_evidence_dict, exact_sdp)
    """
    all_nodes = list(bn.nodes())
    hidden_vars = [n for n in all_nodes if n not in evidence_vars and n != target_node]
    
    print(f"\nHunting for patients... (Locking {len(evidence_vars)} variables as evidence)")
    
    # ==========================================
    # OPTIMIZATION: Pruned Sub-model for Base Decisions
    # ==========================================
    relevant_nodes = list(evidence_vars) + [target_node]
    ancestral_structure = bn.get_ancestral_graph(relevant_nodes)
    
    sub_model = BayesianNetwork(ancestral_structure.edges())
    sub_model.add_nodes_from(ancestral_structure.nodes())
    for node in sub_model.nodes():
        sub_model.add_cpds(bn.get_cpds(node))
        
    base_inference = VariableElimination(sub_model)
    # ==========================================
    
    unfilled_buckets = {b: None for b in buckets}
    batch_count = 0
    
    while any(v is None for v in unfilled_buckets.values()) and batch_count < max_batches:
        batch_count += 1
        print(f"Generating batch {batch_count}/{max_batches} of {batch_size} random realities...")
        
        for _ in range(batch_size):
            # 1. Generate random patient
            temp_patient = {}
            for var in evidence_vars:
                states = sub_model.get_cpds(var).state_names[var]
                temp_patient[var] = random.choice(states)
            
            # 2. Check base decision (Must be >= threshold)
            try:
                base_dist = base_inference.query(variables=[target_node], evidence=temp_patient, show_progress=False)
                if base_dist.get_value(**{target_node: target_value}) < decision_threshold:
                    continue # Reject and generate a new random patient
            except (ValueError, MemoryError):
                print(f"    [!] EXACT INFERENCE IMPOSSIBLE: Sub-network exceeded hardware limits.")
                return unfilled_buckets # Bail out safely
                
            # 3. Calculate Exact SDP
            partitions = get_partitions(bn, hidden_vars, target_node, temp_patient)
            #print(partitions)
            try:
                exact_sdp = fast_broadcast_sdp(bn, target_node, target_value, temp_patient, decision_threshold, partitions)
                #print(exact_sdp)
            except (ValueError, MemoryError):
                print(f"    [!] EXACT SDP IMPOSSIBLE: Tensor exploded during calculation.")
                return unfilled_buckets # Bail out safely
                
            # 4. Check if it fits into any empty bucket!
            empty_targets = [b for b, v in unfilled_buckets.items() if v is None]
            for b in empty_targets:
                if abs(exact_sdp - b) <= tolerance:
                    # Save the result as a tuple
                    unfilled_buckets[b] = (temp_patient.copy(), exact_sdp)
                    print(f"--> Filled bucket {b} with Exact SDP: {exact_sdp:.4f}")
                    break # Only fill one bucket per patient
                    
            # Break the batch loop early if we filled everything
            if not any(v is None for v in unfilled_buckets.values()):
                break
                
    if any(v is None for v in unfilled_buckets.values()):
        missing = [b for b, v in unfilled_buckets.items() if v is None]
        print(f"Finished searching. Could not find patients for buckets: {missing}")
    else:
        print("All buckets filled successfully!")
        
    return unfilled_buckets

'''
Complete randomized version
'''
def find_exact_experimental_patients_random(bn, target_node, target_value, decision_threshold, n_evidence, buckets=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], tolerance=0.05, batch_size=8000, max_batches=2):
    """
    Changes the set of evidence variables for each random patient, instead of keeping them fixed.
    """
    all_nodes = list(bn.nodes())
    available_nodes = [n for n in all_nodes if n != target_node]
    # pick random n_evidence variables to lock as evidence
    evidence_vars = random.sample(available_nodes, min(n_evidence, len(available_nodes)))

    hidden_vars = [n for n in all_nodes if n not in evidence_vars and n != target_node]
    
    print(f"\nHunting for patients... (Locking {len(evidence_vars)} variables as evidence)")
    
    base_inference = VariableElimination(bn)
    
    unfilled_buckets = {b: None for b in buckets}
    batch_count = 0
    
    while any(v is None for v in unfilled_buckets.values()) and batch_count < max_batches:
        batch_count += 1
        print(f"Generating batch {batch_count}/{max_batches} of {batch_size} random realities...")
        
        for _ in range(batch_size):
            # 1. Generate random patient
            temp_patient = {}
            evidence_vars = random.sample(available_nodes, min(n_evidence, len(available_nodes)))
            #print(f"  Evidence vars for this patient: {evidence_vars}")
            for var in evidence_vars:
                states = bn.get_cpds(var).state_names[var]
                temp_patient[var] = random.choice(states)
            
            # 2. Check base decision (Must be >= threshold)
            try:
                base_dist = base_inference.query(variables=[target_node], evidence=temp_patient, show_progress=False)
                if base_dist.get_value(**{target_node: target_value}) < decision_threshold:
                    continue # Reject and generate a new random patient
            except (ValueError, MemoryError):
                print(f"    [!] EXACT INFERENCE IMPOSSIBLE: Sub-network exceeded hardware limits.")
                return unfilled_buckets # Bail out safely
                
            # 3. Calculate Exact SDP
            partitions = get_partitions(bn, hidden_vars, target_node, temp_patient)
            #print(partitions)
            try:
                exact_sdp = fast_broadcast_sdp(bn, target_node, target_value, temp_patient, decision_threshold, partitions)
                #print(exact_sdp)
            except (ValueError, MemoryError):
                print(f"    [!] EXACT SDP IMPOSSIBLE: Tensor exploded during calculation.")
                return unfilled_buckets # Bail out safely
                
            # 4. Check if it fits into any empty bucket!
            empty_targets = [b for b, v in unfilled_buckets.items() if v is None]
            for b in empty_targets:
                if abs(exact_sdp - b) <= tolerance:
                    # Save the result as a tuple
                    unfilled_buckets[b] = (temp_patient.copy(), exact_sdp)
                    print(f"--> Filled bucket {b} with Exact SDP: {exact_sdp:.4f}")
                    break # Only fill one bucket per patient
                    
            # Break the batch loop early if we filled everything
            if not any(v is None for v in unfilled_buckets.values()):
                break
                
    if any(v is None for v in unfilled_buckets.values()):
        missing = [b for b, v in unfilled_buckets.items() if v is None]
        print(f"Finished searching. Could not find patients for buckets: {missing}")
    else:
        print("All buckets filled successfully!")
        
    return unfilled_buckets

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



def get_exact_target_posterior_O1(bn, target, target_value, full_state):
    """
    Calculates P(Target | All Other Nodes) in O(1) time without Variable Elimination.
    Relies purely on the Target's Markov Blanket (its own CPD and its children's CPDs).
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


def fast_mcmc_sdp_estimation(bn, target, target_value, patient, threshold,
                              n_samples=11000, burn_in=1000, thinning=10):
    """
    Estimates the Same-Decision Probability via Metropolis-Hastings MCMC.

    Improvements over previous version:
      1. FIX — seed is drawn proportionally to likelihood weight instead of
         always taking the mode, preventing systematic chain trapping in
         networks where the "flip decision" region is isolated from the mode.
      2. SPEED — MH acceptance ratio is computed via a local CPD update:
         only the CPDs that involve the flipped variable are re-evaluated,
         reducing each iteration from O(all CPDs) to O(Markov blanket).
    """

    hidden_vars = [n for n in bn.nodes() if n not in patient and n != target]
    target_states = bn.get_cpds(target).state_names[target]
    evidence_states = [State(var, val) for var, val in patient.items()]

    # ── Precompute structures used on every MH step ───────────────────────────
    cpd_cache      = {n: bn.get_cpds(n) for n in bn.nodes()}
    children_cache = {v: list(bn.get_children(v)) for v in hidden_vars}

    # For each hidden var, the "affected nodes" when it is flipped are itself
    # plus its children — these are the only CPD terms whose value changes.
    affected_cache = {v: [v] + children_cache[v] for v in hidden_vars}

    # ── Seed: sample proportionally to weight (FIX) ───────────────────────────
    sampler = BayesianModelSampling(bn)
    valid_seed_found = False
    while not valid_seed_found:
        seed_df = sampler.likelihood_weighted_sample(
            size=100, evidence=evidence_states, show_progress=False
        )
        valid_seeds = seed_df[seed_df['_weight'] > 0]
        if not valid_seeds.empty:
            weights = valid_seeds['_weight'].values.astype(float)
            weights /= weights.sum()
            seed_row = valid_seeds.iloc[np.random.choice(len(valid_seeds), p=weights)]
            current_h = {v: seed_row[v] for v in hidden_vars}
            valid_seed_found = True

    # ── Helper: log P(H, E, target=t) for a full state ───────────────────────
    def full_log_joint(h_dict, t_state):
        full = {**h_dict, **patient, target: t_state}
        lp = 0.0
        for cpd in cpd_cache.values():
            p = cpd.get_value(**{v: full[v] for v in cpd.variables})
            if p == 0.0:
                return float('-inf')
            lp += math.log(p)
        return lp

    # log P(H, E) = log Σ_t P(H, E, t)  — stored as dict over target states
    def log_sum_joints(lj_dict):
        vals = [v for v in lj_dict.values() if v != float('-inf')]
        if not vals:
            return float('-inf')
        m = max(vals)
        return m + math.log(sum(math.exp(v - m) for v in vals))

    # ── Local update (SPEED): recompute only affected CPD terms ──────────────
    # When hidden var V is flipped old→new, the log joint changes by:
    #   Δ(t) = Σ_{node ∈ affected(V)} [ log P(node|pa, new) - log P(node|pa, old) ]
    # This is O(|affected(V)|) instead of O(all CPDs).
    def local_log_delta(var, old_val, new_val, h_dict, t_state):
        full_old = {**h_dict, **patient, target: t_state}
        full_new = {**full_old, var: new_val}
        delta = 0.0
        for node in affected_cache[var]:
            cpd   = cpd_cache[node]
            cvars = cpd.variables
            p_old = cpd.get_value(**{v: full_old[v] for v in cvars})
            p_new = cpd.get_value(**{v: full_new[v] for v in cvars})
            if p_new == 0.0:
                return float('-inf')
            if p_old == 0.0:
                # current joint was -inf; caller will recompute from scratch
                return float('inf')
            delta += math.log(p_new) - math.log(p_old)
        return delta

    # ── Initialise running log joints ─────────────────────────────────────────
    current_lj  = {t: full_log_joint(current_h, t) for t in target_states}
    current_log_p = log_sum_joints(current_lj)

    # ── Metropolis-Hastings loop ───────────────────────────────────────────────
    total_iters    = burn_in + n_samples * thinning
    accepted_samples = []

    for i in range(total_iters):
        var      = random.choice(hidden_vars)
        cur_val  = current_h[var]
        others   = [s for s in cpd_cache[var].state_names[var] if s != cur_val]
        if not others:
            continue
        new_val  = random.choice(others)

        # Proposed log joints via local delta (fast path)
        proposed_lj = {}
        recompute   = False
        for t in target_states:
            if current_lj[t] == float('-inf'):
                recompute = True
                break
            delta = local_log_delta(var, cur_val, new_val, current_h, t)
            if delta == float('inf'):       # old CPD was zero → need full pass
                recompute = True
                break
            proposed_lj[t] = current_lj[t] + delta

        if recompute:
            tmp_h = {**current_h, var: new_val}
            proposed_lj = {t: full_log_joint(tmp_h, t) for t in target_states}

        proposed_log_p = log_sum_joints(proposed_lj)

        # MH acceptance
        log_alpha = proposed_log_p - current_log_p
        if log_alpha >= 0 or (
            proposed_log_p != float('-inf') and
            math.log(random.random()) < log_alpha
        ):
            current_h[var]  = new_val
            current_lj      = proposed_lj
            current_log_p   = proposed_log_p

        if i >= burn_in and (i - burn_in) % thinning == 0:
            accepted_samples.append(current_h.copy())

    # ── Evaluate decision boundary ────────────────────────────────────────────
    count_same = 0
    decision_cache = {}

    for sample_h in accepted_samples:
        key = tuple(sorted(sample_h.items()))
        if key not in decision_cache:
            full_ev = {**patient, **sample_h}
            p = get_exact_target_posterior_O1(bn, target, target_value, full_ev)
            decision_cache[key] = (p >= threshold)
        if decision_cache[key]:
            count_same += 1

    return count_same / len(accepted_samples)

def fast_mcmc_sdp_estimation_old(bn, target, target_value, patient, threshold, n_samples=11000, burn_in=1000, thinning=10):
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
            
            # Use the O(1) Markov Blanket evaluator
            p_target = get_exact_target_posterior_O1(bn, target, target_value, full_evidence)
            makes_same = p_target >= threshold
            
            decision_cache[patient_id] = makes_same
            
        if makes_same:
            count_same_decision += 1
            
    return count_same_decision / len(accepted_samples)
