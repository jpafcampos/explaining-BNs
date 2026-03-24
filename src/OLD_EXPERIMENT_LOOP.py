def harvest_patients_for_all_buckets_simulated_annealing(bn, target_node, target_value, decision_threshold, evidence_vars, target_buckets, tolerance=0.05, max_steps=3000):
    """
    Wanders the probability landscape continuously. If it stumbles across an SDP 
    that fits ANY empty bucket, it saves it. Uses SA to pull toward the nearest empty bucket.
    """
    print(f"\n--- Starting Global Harvest for Buckets: {target_buckets} ---")
    
    # 1. Setup Pruned Sub-model for fast base-decision checks
    relevant_nodes = list(evidence_vars) + [target_node]
    ancestral_structure = bn.get_ancestral_graph(relevant_nodes)
    sub_model = BayesianNetwork(ancestral_structure.edges())
    sub_model.add_nodes_from(ancestral_structure.nodes())
    for node in sub_model.nodes():
        sub_model.add_cpds(bn.get_cpds(node))
    inference = VariableElimination(sub_model)
    
    # 2. Initialize Buckets
    unfilled_buckets = {bucket: None for bucket in target_buckets}
    
    # 3. Find a Valid Starting Seed
    current_patient = None
    attempts = 0
    while current_patient is None and attempts < 1000:
        temp_patient = {var: random.choice(sub_model.get_cpds(var).state_names[var]) for var in evidence_vars}
        try:
            base_dist = inference.query(variables=[target_node], evidence=temp_patient, show_progress=False)
            if base_dist.get_value(**{target_node: target_value}) >= decision_threshold:
                current_patient = temp_patient
        except (ValueError, MemoryError):
            return None # Fast fail on impossible networks
        attempts += 1
        
    if current_patient is None:
        print("    [!] Could not find valid starting seed.")
        return unfilled_buckets # Return empty dict
        
    hidden_vars = [v for v in bn.nodes() if v not in current_patient and v != target_node]
    
    # Calculate starting SDP
    partitions = get_partitions(bn, hidden_vars, target_node, current_patient)
    try:
        current_sdp = fast_broadcast_sdp(bn, target_node, target_value, current_patient, decision_threshold, partitions)
    except (ValueError, MemoryError):
        return unfilled_buckets
        
    # --- CHECK STARTING SEED AGAINST ALL BUCKETS ---
    for bucket in list(unfilled_buckets.keys()):
        if abs(current_sdp - bucket) <= tolerance:
            unfilled_buckets[bucket] = (current_patient.copy(), current_sdp)
            print(f"    [+] INSTANT HARVEST: Filled bucket {bucket} with SDP {current_sdp:.4f}!")
            
    # 4. Global Simulated Annealing Loop
    step = 0
    current_temp = 0.5
    cooling_rate = 0.998 # Slower cooling since we are walking a longer path
    
    while step < max_steps and any(v is None for v in unfilled_buckets.values()):
        step += 1
        current_temp = max(0.005, current_temp * cooling_rate)
        
        # Which empty bucket are we closest to right now? (This is our temporary gravity)
        empty_targets = [b for b, v in unfilled_buckets.items() if v is None]
        active_target = min(empty_targets, key=lambda b: abs(current_sdp - b))
        current_error = abs(current_sdp - active_target)
        
        # Mutate
        num_mutations = random.choices([1, 2], weights=[0.8, 0.2])[0]
        vars_to_mutate = random.sample(evidence_vars, num_mutations)
        proposed_patient = current_patient.copy()
        
        for var in vars_to_mutate:
            possible_states = sub_model.get_cpds(var).state_names[var]
            proposed_patient[var] = random.choice([s for s in possible_states if s != proposed_patient[var]])

        # Safety Check
        try:
            base_dist = inference.query(variables=[target_node], evidence=proposed_patient, show_progress=False)
            if base_dist.get_value(**{target_node: target_value}) < decision_threshold:
                continue 
                
            partitions = get_partitions(bn, hidden_vars, target_node, proposed_patient)
            proposed_sdp = fast_broadcast_sdp(bn, target_node, target_value, proposed_patient, decision_threshold, partitions)
        except (ValueError, MemoryError):
            continue 
            
        # --- THE HARVEST CHECK ---
        # Did we stumble across ANY empty bucket?
        for bucket in empty_targets:
            if abs(proposed_sdp - bucket) <= tolerance:
                unfilled_buckets[bucket] = (proposed_patient.copy(), proposed_sdp)
                print(f"    [+] HARVEST SUCCESS (Step {step}): Filled bucket {bucket} with SDP {proposed_sdp:.4f}!")
                # Re-evaluate active target since we just filled one!
                empty_targets = [b for b, v in unfilled_buckets.items() if v is None]
                if not empty_targets:
                    break
        
        if not empty_targets:
            break # We filled them all!
            
        # SA Acceptance Logic (relative to the active target we picked at the start of the loop)
        proposed_error = abs(proposed_sdp - active_target)
        
        accept = False
        if proposed_error < current_error:
            accept = True
        else:
            prob = math.exp((current_error - proposed_error) / current_temp)
            if random.random() < prob:
                accept = True
                
        if accept:
            current_patient = proposed_patient
            current_sdp = proposed_sdp
            
    print(f"--- Harvest Complete after {step} steps ---")
    remaining = [b for b, v in unfilled_buckets.items() if v is None]
    if remaining:
        print(f"    Could not fill buckets: {remaining}")
        
    return unfilled_buckets


def run_targeted_sdp_experiment(bif_directory, output_csv="targeted_sdp_benchmark.csv"):
    bif_files = glob.glob(os.path.join(bif_directory, "*.bif"))
    results = []
    
    H_RATIO = 0.25
    DECISION_THRESHOLD = 0.5
    TARGET_BUCKETS = [0.55, 0.65, 0.75, 0.85, 0.95, 1.0]
    MCMC_TRIALS = 20
    
    for file in bif_files:
        n_nodes, density, rigidity = parse_bn_filename(file)
        print(f"\n========================================")
        print(f"Loading: {os.path.basename(file)}")
        
        bn = BIFReader(file).get_model()
        all_nodes = list(bn.nodes())
        
        target = select_optimal_target_node(bn)
        target_states = bn.get_cpds(target).state_names[target]
        target_value = target_states[1] if len(target_states) > 1 else target_states[0]
        
        available_nodes = [n for n in all_nodes if n != target]
        n_hidden = max(1, int(len(available_nodes) * H_RATIO))
        hidden_vars = random.sample(available_nodes, n_hidden)
        evidence_vars = [n for n in available_nodes if n not in hidden_vars]
        
        for target_sdp in TARGET_BUCKETS:
            print(f"  -> Hunting for patient with Target SDP: {target_sdp}")
            
            # Allow 6 attempts to find a patient (in case Hill Climber gets stuck in a local minimum)
            patient, exact_sdp = None, None
            for attempt in range(6):
                patient, exact_sdp = generate_patient_for_target_sdp(
                    bn, target, target_value, DECISION_THRESHOLD, target_sdp, evidence_vars
                )
                if patient is not None:
                    break
                    
            if patient is None:
                print(f"     [!] Failed to find patient for SDP {target_sdp} in this network.")
                continue
                
            print(f"     Found patient! Exact SDP: {exact_sdp:.4f}")
            
            # ========================================================
            # RACE TIMING 1: EXACT SDP
            # ========================================================
            partitions = get_partitions(bn, hidden_vars, target, patient)
            exact_time = np.nan
            
            try:
                start_time = time.time()
                # Re-run the exact calculation once just to time it cleanly
                exact_sdp_benchmark = fast_broadcast_sdp(bn, target, target_value, patient, DECISION_THRESHOLD, partitions)
                exact_time = time.time() - start_time
                print(f"       -> Exact Time: {exact_time:.4f} seconds")
            except (ValueError, MemoryError):
                print(f"       -> Exact Time: [FAILED DUE TO MEMORY/EINSUM LIMIT]")
            
            # ========================================================
            # RACE TIMING 2: MCMC SDP
            # ========================================================
            mcmc_estimates = []
            mcmc_times = []
            
            for trial in range(MCMC_TRIALS):
                start_time = time.time()
                est_sdp = fast_mcmc_sdp_estimation(
                    bn, target, target_value, patient, DECISION_THRESHOLD
                )
                mcmc_times.append(time.time() - start_time)
                mcmc_estimates.append(est_sdp)
                
            mcmc_mean = np.mean(mcmc_estimates)
            mcmc_variance = np.var(mcmc_estimates)
            mcmc_avg_time = np.mean(mcmc_times)
            
            print(f"       -> MCMC Avg Time: {mcmc_avg_time:.4f} seconds")
            
            absolute_error = abs(exact_sdp - mcmc_mean)
            
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
                'Absolute_Error': absolute_error
            })
            
            # Save progressively
            pd.DataFrame(results).to_csv(output_csv, index=False)

    print(f"\nExperiment Complete! Results saved to {output_csv}")
    return pd.DataFrame(results)