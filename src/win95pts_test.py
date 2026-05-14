import random
import time
import tracemalloc
import gc
import pandas as pd
from pgmpy.utils import get_example_model
from same_decision_probability_calculation import fast_broadcast_sdp
from utils import get_partitions


def check_d_separation(bn, partition, target, evidence_vars):
    if not partition:
        return True
    
    active_trails = bn.active_trail_nodes(target, observed=evidence_vars)
    reachable_from_target = active_trails[target]
    
    return set(partition).isdisjoint(reachable_from_target)

def harvest_win95_sdp1_tracked(output_csv="win95_sdp1_tracked.csv", max_attempts=10000):
    results = []
    
    print(f"Starting Harvester: win95pts | Target SDP=1.0 | H-Ratio=0.5")
    print(f"{'='*75}")
    
    print("\n[+] Downloading/Loading network: win95pts")
    bn = get_example_model('win95pts')
    all_nodes = list(bn.nodes())
    
    target = 'PrtMem'
    target_states = bn.get_cpds(target).state_names[target]
    target_value = target_states[1] if len(target_states) > 1 else target_states[0]
    
    available_nodes = [n for n in all_nodes if n != target]
    n_hidden = int(len(available_nodes) * 0.5) 
    
    success_count = 0
    
    for attempt in range(max_attempts):
        # 1. Sample Evidence and Hidden Variables
        evidence_vars = random.sample(available_nodes, len(available_nodes) - n_hidden)
        hidden_vars = [n for n in all_nodes if n not in evidence_vars and n != target]
        
        # Randomly choose states to create the logical contradictions that trigger pruning
        patient = {var: random.choice(bn.get_cpds(var).state_names[var]) for var in evidence_vars}
        
        # 2. Get Partitions
        partitions = get_partitions(bn, hidden_vars, target, patient)
        if not partitions:
            continue
            
        largest_partition = max(partitions, key=len)
        max_partition_size = len(largest_partition)
        
        # Skip small partitions to focus on the big space/time trade-offs
        if max_partition_size < 20:
            continue
            
        # 3. Compute Exact SDP with Space/Time Tracking
        gc.collect()
        tracemalloc.start()
        start_time = time.perf_counter()
        
        try:
            sdp_val = fast_broadcast_sdp(bn, target, target_value, patient, 0.5, partitions)
            elapsed_time = time.perf_counter() - start_time
            _, peak_mem = tracemalloc.get_traced_memory()
            peak_mem_mb = peak_mem / 1024 / 1024
        except Exception:
            tracemalloc.stop()
            continue
        finally:
            tracemalloc.stop()
            
        # 4. Check Criteria (SDP == 1.0)
        if abs(sdp_val - 1.0) < 1e-6:
            success_count += 1
            
            # 5. Check D-Separation
            is_d_sep = check_d_separation(bn, largest_partition, target, list(patient.keys()))
            
            print(f"  [HIT] Max Part: {max_partition_size:2d} | D-Sep: {str(is_d_sep):5s} | Time: {elapsed_time:6.2f}s | Mem: {peak_mem_mb:7.2f} MB | Attempt: {attempt+1}")
            
            results.append({
                'Network': 'win95pts',
                'N_Nodes': len(all_nodes),
                'Target': target,
                'Target_Value': target_value,
                'H_Ratio': 0.5,
                'N_Hidden': n_hidden,
                'Max_Partition_Size': max_partition_size,
                'Largest_Part_D_Separated': is_d_sep,
                'Exact_Time_sec': elapsed_time,
                'Exact_Peak_Mem_MB': peak_mem_mb,
                'SDP': sdp_val,
                'Evidence_Dict': str(patient),
                'Hidden_Vars': str(hidden_vars)
            })
            
            # Flush to CSV immediately
            pd.DataFrame(results).to_csv(output_csv, index=False)
            
    print(f"\n[-] Finished: Harvested {success_count} configurations.")

if __name__ == '__main__':
    harvest_win95_sdp1_tracked(
        output_csv="win95_sdp1_tracked_dsep.csv",
        max_attempts=10000 
    )