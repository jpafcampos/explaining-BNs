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
import gc

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
def find_exact_experimental_patients_random(bn, target_node, target_value, decision_threshold, 
                                            n_evidence, buckets=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 
                                            tolerance=0.05, batch_size=8000, max_batches=2, max_partition_size=28):
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
            #print(f"  Generating random patient with {n_evidence} evidence variables, {_+1} / {batch_size} in this batch...")
            temp_patient = {}
            evidence_vars = random.sample(available_nodes, min(n_evidence, len(available_nodes)))
            hidden_vars = [n for n in all_nodes if n not in evidence_vars and n != target_node]
            #print(f"  Evidence vars for this patient: {evidence_vars}")
            for var in evidence_vars:
                states = bn.get_cpds(var).state_names[var]
                temp_patient[var] = random.choice(states)
            
            partitions = get_partitions(bn, hidden_vars, target_node, temp_patient)
            #print(f"       -> Biggest partition size during harvester: {max(len(p) for p in partitions)} hidden variables")
            #print(partitions)
            # Biggest partion can not be over 28 variables
            if max(len(p) for p in partitions) > max_partition_size:
                continue # Skip this patient, it's too big for our hardware to handle

            # 2. Check base decision (Must be >= threshold)
            try:
                base_dist = base_inference.query(variables=[target_node], evidence=temp_patient, show_progress=False)
                if base_dist.get_value(**{target_node: target_value}) < decision_threshold:
                    continue # Reject and generate a new random patient
            except (ValueError, MemoryError):
                print(f"    [!] EXACT INFERENCE IMPOSSIBLE: Sub-network exceeded hardware limits.")
                gc.collect() # Clean up after the explosion
                return unfilled_buckets
                
            # 3. Calculate Exact SDP
            try:
                exact_sdp = fast_broadcast_sdp(bn, target_node, target_value, temp_patient, decision_threshold, partitions)
            except (ValueError, MemoryError):
                print(f"    [!] EXACT SDP IMPOSSIBLE: Tensor exploded during calculation.")
                gc.collect() # Clean up after the explosion
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


import numpy as np
import math
import random



def fast_mcmc_sdp_estimation_new(bn, target, target_value, patient, threshold,
                              n_samples=11000, burn_in=1000, thinning=10,
                              use_lw_seed=True):
    """
    Estimates the Same-Decision Probability via Metropolis-Hastings MCMC.

    NumPy-vectorised speedup.

      1. LIKELIHOOD-WEIGHTED SEED (size=1) — chain starts inside the typical
         set of P(H | E). Single draw is cheap and greatly reduces bias from
         the starting position.
      2. NumPy-indexed inner loop — all CPDs are pre-converted to integer-
         indexed numpy arrays; MH proposals access them via tuple indexing
         instead of pgmpy's string-keyed get_value.
      3. FULL-JOINT FALLBACK — when the local delta encounters a zero-
         probability configuration, fall back to full_log_joint (also
         vectorised) instead of rejecting, preserving the original
         statistical behaviour.
    """

    hidden_vars = [n for n in bn.nodes() if n not in patient and n != target]
    target_states_list = bn.get_cpds(target).state_names[target]
    n_target_states = len(target_states_list)
    target_state_idx = list(range(n_target_states))

    # ──────────────────────────────────────────────────────────────────────
    # 1) Build integer-indexed view of the network
    # ──────────────────────────────────────────────────────────────────────
    all_nodes = list(bn.nodes())
    state_index = {}
    for n in all_nodes:
        cpd = bn.get_cpds(n)
        state_index[n] = {s: i for i, s in enumerate(cpd.state_names[n])}

    cpd_array = {}       # node -> numpy array shape (node_card, pa1_card, ...)
    cpd_vars  = {}       # node -> list of variables in CPD order
    EPSILON = 1e-10      # Tuning parameter for smoothing zero-probabilities

    for n in all_nodes:
        cpd = bn.get_cpds(n)
        arr = np.asarray(cpd.values, dtype=float)
        
        # --- EPSILON SMOOTHING ---
        # 1. Replace absolute zeros with a tiny probability
        arr[arr == 0.0] = EPSILON
        
        # 2. Re-normalize the CPD along the node's state axis (axis=0 in pgmpy)
        # This ensures the probabilities for any given parent configuration still sum to 1.0
        arr = arr / arr.sum(axis=0, keepdims=True)
        
        cpd_array[n] = arr
        cpd_vars[n]  = list(cpd.variables)

    inv_state = {v: {i: s for s, i in state_index[v].items()} for v in bn.nodes()}
    children_cache = {v: list(bn.get_children(v)) for v in hidden_vars}
    affected_cache = {v: [v] + children_cache[v] for v in hidden_vars}

    # Current integer state for patient (fixed)
    patient_idx = {v: state_index[v][val] for v, val in patient.items()}
    current_idx = dict(patient_idx)

    # ──────────────────────────────────────────────────────────────────────
    # 2) Seed hidden vars from a single likelihood-weighted draw
    # ──────────────────────────────────────────────────────────────────────

    # Seed section
    if use_lw_seed:
        try:
            topo_order = list(nx.topological_sort(bn))
            sample = dict(patient)
            for node in topo_order:
                if node in sample or node == target or node not in hidden_vars:
                    continue
                cpd     = bn.get_cpds(node)
                parents = cpd.variables[1:] 
                if not parents:
                    probs = cpd_array[node].flatten()
                else:
                    parent_idx_tuple = tuple(
                        state_index[p][sample[p]] if p in sample else 0
                        for p in parents
                    )
                    probs = cpd_array[node][(slice(None),) + parent_idx_tuple]
                probs = np.array(probs, dtype=float)
                probs = np.maximum(probs, 0)
                total = probs.sum()
                probs = probs / total if total > 0 else np.ones(len(probs)) / len(probs)
                sampled           = np.random.choice(len(probs), p=probs)
                sample[node]      = inv_state[node][sampled]
                current_idx[node] = sampled
        except Exception as e:
            print(f"    [SEED] Ancestral seed failed ({e}) — random fallback")
            for v in hidden_vars:
                current_idx[v] = random.randrange(cpd_array[v].shape[0])
    else:
        for v in hidden_vars:
            current_idx[v] = random.randrange(cpd_array[v].shape[0])

    # ──────────────────────────────────────────────────────────────────────
    # 3) Helper functions — vectorised
    # ──────────────────────────────────────────────────────────────────────
    def full_log_joint(state_idx_dict, t_idx):
        """Log P(H, E, target=t_idx) — iterates all CPDs."""
        lp = 0.0
        for node in all_nodes:
            idx = tuple(
                t_idx if v == target else state_idx_dict[v]
                for v in cpd_vars[node]
            )
            p = cpd_array[node][idx]
            if p == 0.0:
                return float('-inf')
            lp += math.log(p)
        return lp

    def log_sum(values):
        finite = [v for v in values if v != float('-inf')]
        if not finite:
            return float('-inf')
        m = max(finite)
        return m + math.log(sum(math.exp(v - m) for v in finite))

    # Initialise running log joints at seed state
    current_lj = [full_log_joint(current_idx, t) for t in target_state_idx]
    current_log_p = log_sum(current_lj)

    # ──────────────────────────────────────────────────────────────────────
    # 4) Metropolis-Hastings loop
    # ──────────────────────────────────────────────────────────────────────
    total_iters = burn_in + n_samples * thinning
    accepted_samples = []

    for i in range(total_iters):
        var = random.choice(hidden_vars)
        cur_val = current_idx[var]
        cardinality = cpd_array[var].shape[0]
        if cardinality < 2:
            pass
        else:
            # Propose a different state uniformly at random
            new_val = random.randrange(cardinality - 1)
            if new_val >= cur_val:
                new_val += 1

            # Attempt fast local-delta path; fall back to full recompute on
            # zero-probability edge cases.
            proposed_lj = [0.0] * n_target_states
            recompute = False
            for t_idx in target_state_idx:
                if current_lj[t_idx] == float('-inf'):
                    recompute = True
                    break
                delta = 0.0
                failed = False
                for node in affected_cache[var]:
                    order = cpd_vars[node]
                    p_old_args = tuple(
                        t_idx if v == target else current_idx[v]
                        for v in order
                    )
                    p_new_args = tuple(
                        t_idx if v == target
                        else (new_val if v == var else current_idx[v])
                        for v in order
                    )
                    p_old = cpd_array[node][p_old_args]
                    p_new = cpd_array[node][p_new_args]
                    if p_new == 0.0:
                        failed = True
                        break
                    if p_old == 0.0:
                        # Old state was zero-prob — need full pass to recover
                        recompute = True
                        break
                    delta += math.log(p_new) - math.log(p_old)

                if recompute:
                    break
                if failed:
                    # New state is zero-prob under some target assignment — reject
                    proposed_lj[t_idx] = float('-inf')
                    continue
                proposed_lj[t_idx] = current_lj[t_idx] + delta

        # Full-joint fallback
        if recompute:
            tmp_idx = dict(current_idx)
            tmp_idx[var] = new_val
            proposed_lj = [full_log_joint(tmp_idx, t) for t in target_state_idx]

        proposed_log_p = log_sum(proposed_lj)

        # MH acceptance
        log_alpha = proposed_log_p - current_log_p
        if log_alpha >= 0 or (
            proposed_log_p != float('-inf')
            and math.log(random.random()) < log_alpha
        ):
            current_idx[var] = new_val
            current_lj = proposed_lj
            current_log_p = proposed_log_p

        # Record thinned post-burn samples
        if i >= burn_in and (i - burn_in) % thinning == 0:
            accepted_samples.append(tuple(current_idx[v] for v in hidden_vars))

    # ──────────────────────────────────────────────────────────────────────
    # 5) Decision boundary evaluation
    # ──────────────────────────────────────────────────────────────────────
    

    count_same = 0
    decision_cache = {}

    for snapshot in accepted_samples:
        key = snapshot
        if key not in decision_cache:
            sample_h = {
                v: inv_state[v][snapshot[k]]
                for k, v in enumerate(hidden_vars)
            }
            full_ev = {**patient, **sample_h}
            p = get_exact_target_posterior_O1(bn, target, target_value, full_ev)
            decision_cache[key] = (p >= threshold)
        if decision_cache[key]:
            count_same += 1

    return count_same / len(accepted_samples)

def pt_mcmc_sdp_estimation(bn, target, target_value, patient, threshold,
                            n_samples=11000, burn_in=1000, thinning=10,
                            n_chains=4, max_temp=10.0):


    hidden_vars     = [n for n in bn.nodes() if n not in patient and n != target]
    target_states   = bn.get_cpds(target).state_names[target]
    cpd_cache       = {n: bn.get_cpds(n) for n in bn.nodes()}
    children_cache  = {v: list(bn.get_children(v)) for v in hidden_vars}
    affected_cache  = {v: [v] + children_cache[v] for v in hidden_vars}
    evidence_states = [State(var, val) for var, val in patient.items()]
    sampler         = BayesianModelSampling(bn)

    # Geometrically spaced temperature ladder: chain 0 = cold (τ=1), chain k = hot
    temps = np.geomspace(1.0, max_temp, n_chains)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def full_log_joint(h_dict, t_state):
        """log P(H, E, target=t_state) — full joint over all CPDs."""
        full = {**h_dict, **patient, target: t_state}
        lp = 0.0
        for cpd in cpd_cache.values():
            p = cpd.get_value(**{v: full[v] for v in cpd.variables})
            if p == 0.0:
                return float('-inf')
            lp += math.log(p)
        return lp

    def log_sum_joints(lj_dict):
        """log P(H, E) = log Σ_t P(H, E, t) via log-sum-exp."""
        vals = [v for v in lj_dict.values() if v != float('-inf')]
        if not vals:
            return float('-inf')
        m = max(vals)
        return m + math.log(sum(math.exp(v - m) for v in vals))

    def local_log_delta(var, old_val, new_val, h_dict, t_state):
        """
        O(d·k) local update: recompute only the CPDs in the Markov blanket
        of the flipped variable rather than the full joint.
        """
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
                return float('inf')   # triggers full recompute in caller
            delta += math.log(p_new) - math.log(p_old)
        return delta

    def get_seed():
        """
        Proportional-weight seeding: sample starting point weighted by
        likelihood rather than always taking the mode, diversifying chains.
        """
        while True:
            seed_df = sampler.likelihood_weighted_sample(
                size=100, evidence=evidence_states, show_progress=False
            )
            valid = seed_df[seed_df['_weight'] > 0]
            if not valid.empty:
                w = valid['_weight'].values.astype(float)
                w /= w.sum()
                row = valid.iloc[np.random.choice(len(valid), p=w)]
                return {v: row[v] for v in hidden_vars}

    # ── Initialise all chains from diverse seeds ──────────────────────────────
    chains     = [get_seed() for _ in range(n_chains)]
    chain_ljs  = [
        {t: full_log_joint(h, t) for t in target_states}
        for h in chains
    ]
    chain_logp = [log_sum_joints(lj) for lj in chain_ljs]

    n_pairs       = n_chains - 1
    swap_attempts = np.zeros(n_pairs, dtype=int)
    swap_accepts  = np.zeros(n_pairs, dtype=int)

    # ── Main loop ─────────────────────────────────────────────────────────────
    total_iters      = burn_in + n_samples * thinning
    accepted_samples = []

    for i in range(total_iters):

        # ── Step 1: MH update for every chain at its own temperature ──────────
        # Each chain proposes a single-variable flip and accepts via a
        # tempered ratio: α = min(1, [P(H',E) / P(H,E)]^(1/τ))
        # Hot chains (high τ) accept more freely, exploring broadly.
        # Cold chain (τ=1) targets the true posterior.
        for c in range(n_chains):
            var     = random.choice(hidden_vars)
            cur_val = chain[var] if (chain := chains[c]) else None
            cur_val = chains[c][var]
            others  = [s for s in cpd_cache[var].state_names[var] if s != cur_val]
            if not others:
                continue
            new_val = random.choice(others)

            # Local delta for each target state (fast path)
            proposed_lj = {}
            recompute   = False
            for t in target_states:
                if chain_ljs[c][t] == float('-inf'):
                    recompute = True
                    break
                delta = local_log_delta(var, cur_val, new_val, chains[c], t)
                if delta == float('inf'):
                    recompute = True
                    break
                proposed_lj[t] = chain_ljs[c][t] + delta

            if recompute:
                tmp         = {**chains[c], var: new_val}
                proposed_lj = {t: full_log_joint(tmp, t) for t in target_states}

            proposed_log_p = log_sum_joints(proposed_lj)

            # Tempered acceptance: divide log-ratio by τ
            # τ=1  → standard MH (cold, conservative)
            # τ>1  → flattened acceptance (hot, exploratory)
            log_alpha = (proposed_log_p - chain_logp[c]) / temps[c]

            if log_alpha >= 0 or (
                proposed_log_p != float('-inf') and
                math.log(random.random()) < log_alpha
            ):
                chains[c][var]  = new_val
                chain_ljs[c]    = proposed_lj
                chain_logp[c]   = proposed_log_p

        # ── Step 2: Parallel tempering swap between one adjacent pair ─────────
        # Propose swapping the configurations of chains c1 and c2.
        # The MH ratio for this swap is:
        #   log α = (1/τ_c1 - 1/τ_c2) × (log P(H_c2,E) - log P(H_c1,E))
        # Hot-chain configurations that pass the cold chain's threshold
        # get injected into the cold chain, allowing mode crossing.
        if i % 10 == 0 and n_chains > 1:
            c1 = random.randint(0, n_pairs - 1)
            c2 = c1 + 1
            swap_attempts[c1] += 1

            log_alpha_swap = (
                (1.0 / temps[c1] - 1.0 / temps[c2]) *
                (chain_logp[c2] - chain_logp[c1])
            )

            if log_alpha_swap >= 0 or math.log(random.random()) < log_alpha_swap:
                chains[c1],     chains[c2]     = chains[c2],     chains[c1]
                chain_ljs[c1],  chain_ljs[c2]  = chain_ljs[c2],  chain_ljs[c1]
                chain_logp[c1], chain_logp[c2] = chain_logp[c2], chain_logp[c1]
                swap_accepts[c1] += 1

        # ── Step 3: Collect from cold chain only (τ=1, index 0) ───────────────
        if i >= burn_in and (i - burn_in) % thinning == 0:
            accepted_samples.append(chains[0].copy())

    # ── Report swap health ────────────────────────────────────────────────────
    for p_idx in range(n_pairs):
        rate = (swap_accepts[p_idx] / swap_attempts[p_idx]
                if swap_attempts[p_idx] > 0 else 0.0)
        status = '✓' if 0.2 <= rate <= 0.5 else '✗ adjust max_temp'
        #print(f"  Swap τ={temps[p_idx]:.1f}↔τ={temps[p_idx+1]:.1f}: " f"{rate:.3f}  {status}")

    # ── Evaluate decision boundary on cold chain samples ─────────────────────
    count_same    = 0
    decision_cache = {}

    for sample_h in accepted_samples:
        key = tuple(sorted(sample_h.items()))
        if key not in decision_cache:
            full_ev = {**patient, **sample_h}
            p       = get_exact_target_posterior_O1(
                          bn, target, target_value, full_ev)
            decision_cache[key] = (p >= threshold)
        if decision_cache[key]:
            count_same += 1

    return count_same / len(accepted_samples)


import numpy as np
import math
import random
import networkx as nx


def vectorized_pt_mcmc_sdp_estimation(bn, target, target_value, patient, threshold,
                            n_samples=11000, burn_in=1000, thinning=10,
                            n_chains=4, max_temp=10.0,
                            use_ancestral_seed=True):
    """
    Parallel-tempering MCMC for SDP estimation, NumPy-vectorised.

    Optimisations over the original PT version:
      1. Integer-indexed CPD tables — all hot-loop CPD accesses are
         multi-dim numpy lookups instead of pgmpy string-keyed get_value.
      2. Fast ancestral seed per chain — replaces the costly
         BayesianModelSampling.likelihood_weighted_sample for each chain.
      3. Local-delta proposals with full-joint fallback — same statistical
         behaviour as the original; only the cost per inner step changes.
      4. Decision boundary still evaluated via get_exact_target_posterior_O1
         on cold chain samples.

    Statistical structure (unchanged from original):
      - Geometrically spaced temperature ladder, chain 0 cold (τ=1).
      - Tempered MH per chain: log_alpha / τ.
      - Adjacent-pair swap every 10 iterations.
      - Sample collection from cold chain only.
    """

    hidden_vars        = [n for n in bn.nodes() if n not in patient and n != target]
    target_states_list = bn.get_cpds(target).state_names[target]
    n_target_states    = len(target_states_list)
    target_state_idx   = list(range(n_target_states))

    # ──────────────────────────────────────────────────────────────────────
    # 1) Integer-indexed view of the network (same as plain MCMC)
    # ──────────────────────────────────────────────────────────────────────
    all_nodes   = list(bn.nodes())
    state_index = {}
    for n in all_nodes:
        cpd = bn.get_cpds(n)
        state_index[n] = {s: i for i, s in enumerate(cpd.state_names[n])}

    cpd_array = {}
    cpd_vars  = {}
    for n in all_nodes:
        cpd = bn.get_cpds(n)
        cpd_array[n] = np.asarray(cpd.values)
        cpd_vars[n]  = list(cpd.variables)

    children_cache = {v: list(bn.get_children(v)) for v in hidden_vars}
    affected_cache = {v: [v] + children_cache[v] for v in hidden_vars}
    inv_state      = {v: {i: s for s, i in state_index[v].items()} for v in bn.nodes()}

    patient_idx = {v: state_index[v][val] for v, val in patient.items()}

    # Geometric temperature ladder
    temps = np.geomspace(1.0, max_temp, n_chains)

    # ──────────────────────────────────────────────────────────────────────
    # 2) Per-chain helpers
    # ──────────────────────────────────────────────────────────────────────
    def full_log_joint(state_idx_dict, t_idx):
        lp = 0.0
        for node in all_nodes:
            idx = tuple(
                t_idx if v == target else state_idx_dict[v]
                for v in cpd_vars[node]
            )
            p = cpd_array[node][idx]
            if p == 0.0:
                return float('-inf')
            lp += math.log(p)
        return lp

    def log_sum(values):
        finite = [v for v in values if v != float('-inf')]
        if not finite:
            return float('-inf')
        m = max(finite)
        return m + math.log(sum(math.exp(v - m) for v in finite))

    def get_seed_idx():
        """Fast ancestral seed — returns int-indexed dict for one chain."""
        idx = dict(patient_idx)
        if use_ancestral_seed:
            try:
                topo_order = list(nx.topological_sort(bn))
                sample = dict(patient)
                for node in topo_order:
                    if node in sample or node == target or node not in hidden_vars:
                        continue
                    cpd     = bn.get_cpds(node)
                    parents = cpd.variables[1:]
                    if not parents:
                        probs = cpd_array[node].flatten()
                    else:
                        parent_idx_tuple = tuple(
                            state_index[p][sample[p]] if p in sample else 0
                            for p in parents
                        )
                        probs = cpd_array[node][(slice(None),) + parent_idx_tuple]
                    probs = np.asarray(probs, dtype=float)
                    probs = np.maximum(probs, 0)
                    total = probs.sum()
                    probs = probs / total if total > 0 else np.ones(len(probs)) / len(probs)
                    sampled = np.random.choice(len(probs), p=probs)
                    sample[node] = inv_state[node][sampled]
                    idx[node]    = sampled
            except Exception:
                for v in hidden_vars:
                    idx[v] = random.randrange(cpd_array[v].shape[0])
        else:
            for v in hidden_vars:
                idx[v] = random.randrange(cpd_array[v].shape[0])
        return idx

    # ──────────────────────────────────────────────────────────────────────
    # 3) Initialise all chains
    # ──────────────────────────────────────────────────────────────────────
    chains      = [get_seed_idx() for _ in range(n_chains)]
    chain_ljs   = [
        [full_log_joint(c, t) for t in target_state_idx]
        for c in chains
    ]
    chain_logp  = [log_sum(lj) for lj in chain_ljs]

    n_pairs       = n_chains - 1
    swap_attempts = np.zeros(n_pairs, dtype=int)
    swap_accepts  = np.zeros(n_pairs, dtype=int)

    # ──────────────────────────────────────────────────────────────────────
    # 4) Main loop
    # ──────────────────────────────────────────────────────────────────────
    total_iters      = burn_in + n_samples * thinning
    accepted_samples = []

    for i in range(total_iters):

        # ── Step 1: per-chain tempered MH update ──────────────────────────
        for c in range(n_chains):
            var = random.choice(hidden_vars)
            cur_val     = chains[c][var]
            cardinality = cpd_array[var].shape[0]
            if cardinality < 2:
                continue
            new_val = random.randrange(cardinality - 1)
            if new_val >= cur_val:
                new_val += 1

            # Try fast local delta path
            proposed_lj = [0.0] * n_target_states
            recompute   = False
            for t_idx in target_state_idx:
                if chain_ljs[c][t_idx] == float('-inf'):
                    recompute = True
                    break
                delta  = 0.0
                failed = False
                for node in affected_cache[var]:
                    order = cpd_vars[node]
                    p_old_args = tuple(
                        t_idx if v == target else chains[c][v]
                        for v in order
                    )
                    p_new_args = tuple(
                        t_idx if v == target
                        else (new_val if v == var else chains[c][v])
                        for v in order
                    )
                    p_old = cpd_array[node][p_old_args]
                    p_new = cpd_array[node][p_new_args]
                    if p_new == 0.0:
                        failed = True
                        break
                    if p_old == 0.0:
                        recompute = True
                        break
                    delta += math.log(p_new) - math.log(p_old)

                if recompute:
                    break
                if failed:
                    proposed_lj[t_idx] = float('-inf')
                    continue
                proposed_lj[t_idx] = chain_ljs[c][t_idx] + delta

            if recompute:
                tmp = dict(chains[c])
                tmp[var] = new_val
                proposed_lj = [full_log_joint(tmp, t) for t in target_state_idx]

            proposed_log_p = log_sum(proposed_lj)

            # Tempered acceptance: log_alpha / τ
            log_alpha = (proposed_log_p - chain_logp[c]) / temps[c]

            if log_alpha >= 0 or (
                proposed_log_p != float('-inf')
                and math.log(random.random()) < log_alpha
            ):
                chains[c][var]  = new_val
                chain_ljs[c]    = proposed_lj
                chain_logp[c]   = proposed_log_p

        # ── Step 2: adjacent-pair swap every 10 iterations ────────────────
        if i % 10 == 0 and n_chains > 1:
            c1 = random.randint(0, n_pairs - 1)
            c2 = c1 + 1
            swap_attempts[c1] += 1

            log_alpha_swap = (
                (1.0 / temps[c1] - 1.0 / temps[c2])
                * (chain_logp[c2] - chain_logp[c1])
            )

            if log_alpha_swap >= 0 or math.log(random.random()) < log_alpha_swap:
                chains[c1],     chains[c2]     = chains[c2],     chains[c1]
                chain_ljs[c1],  chain_ljs[c2]  = chain_ljs[c2],  chain_ljs[c1]
                chain_logp[c1], chain_logp[c2] = chain_logp[c2], chain_logp[c1]
                swap_accepts[c1] += 1

        # ── Step 3: collect from cold chain only ──────────────────────────
        if i >= burn_in and (i - burn_in) % thinning == 0:
            accepted_samples.append(
                tuple(chains[0][v] for v in hidden_vars)
            )

    # ──────────────────────────────────────────────────────────────────────
    # 5) Decision boundary evaluation (cold chain samples)
    # ──────────────────────────────────────────────────────────────────────
    count_same     = 0
    decision_cache = {}

    for snapshot in accepted_samples:
        key = snapshot
        if key not in decision_cache:
            sample_h = {
                v: inv_state[v][snapshot[k]]
                for k, v in enumerate(hidden_vars)
            }
            full_ev = {**patient, **sample_h}
            p = get_exact_target_posterior_O1(bn, target, target_value, full_ev)
            decision_cache[key] = (p >= threshold)
        if decision_cache[key]:
            count_same += 1

    return count_same / len(accepted_samples)