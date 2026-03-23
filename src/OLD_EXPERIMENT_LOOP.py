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