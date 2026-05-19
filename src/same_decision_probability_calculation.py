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



def get_initial_posterior(model, D, d_value, evidence):
    """
    Computes P(D=d | evidence) using only the Markov blanket of D.
    Marginalises over any parent/co-parent variables not in evidence
    using their marginal priors.
    Falls back to uniform if the computation fails.
    """
    target_states = model.get_cpds(D).state_names[D]
    
    try:
        # Use only nodes in the ancestral graph of {D} ∪ evidence
        # but query via pgmpy's BeliefPropagation which handles
        # missing variables more gracefully than VE on dense nets
        relevant = set(evidence.keys()) | {D}
        sub_ve = VariableElimination(model)
        result = sub_ve.query(
            variables=[D],
            evidence=evidence,
            show_progress=False,
            joint=False
        )
        p_d     = result.get_value(**{D: d_value})
        not_d   = [s for s in target_states if s != d_value][0]
        p_not_d = result.get_value(**{D: not_d})
        
        if np.isnan(p_d) or np.isnan(p_not_d):
            raise ValueError("VE returned NaN")
            
        return p_d, p_not_d
        
    except Exception as e:
        print(f"    [SDP] Initial posterior failed: {e} — using CPD prior")
        # Ultimate fallback: use the target's marginal CPD prior
        # (ignores evidence but avoids crashing)
        cpd = model.get_cpds(D)
        if not model.get_parents(D):
            idx_d     = cpd.state_names[D].index(d_value)
            not_d_val = [s for s in cpd.state_names[D] if s != d_value][0]
            idx_not_d = cpd.state_names[D].index(not_d_val)
            return float(cpd.values[idx_d]), float(cpd.values[idx_not_d])
        return 0.5, 0.5  # truly unknown
    
'''
Fast version, trading-off space for time by materialising joint tensors per partition S_i
'''

def fast_broadcast_sdp(model, D, d_value, evidence, threshold, partitions):
    #print(f"    [SDP] called with {len(partitions)} partitions, "
    #    f"max_size={max(len(p) for p in partitions) if partitions else 0}")
    #inference = VariableElimination(model)
    
    d_states = model.get_cpds(D).state_names[D]
    d_index = d_states.index(d_value)
    not_d_value = d_states[1] if d_index == 0 else d_states[0]

    # 1. Compute Initial Log-Odds
    relevant_nodes = list(evidence.keys()) + [D]
    ancestral_structure = model.get_ancestral_graph(relevant_nodes)
    
    sub_model = BayesianNetwork(ancestral_structure.edges())
    sub_model.add_nodes_from(ancestral_structure.nodes())
    
    for node in sub_model.nodes():
        sub_model.add_cpds(model.get_cpds(node))
        
    sub_inference = VariableElimination(sub_model)
    
    initial_dist = sub_inference.query(variables=[D], evidence=evidence, elimination_order='MinFill', show_progress=False)
    p_d_e = initial_dist.get_value(**{D: d_value})
    p_not_d_e = initial_dist.get_value(**{D: not_d_value})

    # 1. Compute Initial Log-Odds, new version using Markov Blanket

    #p_d_e, p_not_d_e = get_initial_posterior(model, D, d_value, evidence)

    #print(f"    [SDP] p_d_e={p_d_e}, p_not_d_e={p_not_d_e}")

    if p_not_d_e == 0:
        #print(f"    [SDP] p_not_d_e==0, returning 1.0")
        return 1.0
    if p_d_e == 0:
        #print(f"    [SDP] p_d_e==0, returning 0.0")
        return 0.0
    
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
           
            if np.any(np.isnan(joint_prob)):
                print(f"    [SDP] NaN detected in joint_prob for partition {s_i_list}")
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

    # ------ DEBUG
    #print(f"    [SDP] partitions_data built: {len(partitions_data)} entries")
    #for i, pd in enumerate(partitions_data):
    #    print(f"      partition {i}: w_flat[:3]={pd['w_flat'][:3]}, "
    #        f"max_w={pd['max_w']}, min_w={pd['min_w']}")
    # ------ DEBUG

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

    result = dfs(0, log_O_d_e, 1.0, 1.0)
    #print(f"    [SDP] dfs returned: {result}")
    return result


"""
Exact Same-Decision Probability following

    Chen, Choi, Darwiche.
    "Algorithms and Applications for the Same-Decision Probability."
    JAIR 49 (2014) 601-633, Section 5.2.
"""

import math
import itertools
import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination


# ---------------------------------------------------------------------------
# Log-factor helpers (used only for constrained max/min-elimination)
# ---------------------------------------------------------------------------
# A log-factor is a pair (vars: list[str], vals: np.ndarray) where vals.shape
# matches the order of vars. Combination of log-factors that share a variable
# is pointwise ADDITION; elimination of a variable is MAX or MIN over its
# axis. This is the log-space analogue of sum-product VE.

def _log_ratio_factor(f_psi, f_phi, eps=1e-300):
    """log(f_psi / f_phi) element-wise, returned as (vars, vals)."""
    if list(f_psi.variables) != list(f_phi.variables):
        order = [f_phi.variables.index(v) for v in f_psi.variables]
        phi_vals = np.transpose(f_phi.values, order)
    else:
        phi_vals = f_phi.values
    log_r = np.log(np.maximum(f_psi.values, eps)) - np.log(np.maximum(phi_vals, eps))
    return list(f_psi.variables), log_r


def _combine_log_factors(log_factors):
    """Pointwise-add a list of log-factors via broadcasting."""
    all_vars = []
    cards = {}
    for vs, vals in log_factors:
        for i, v in enumerate(vs):
            if v not in all_vars:
                all_vars.append(v)
                cards[v] = vals.shape[i]
    out = np.zeros([cards[v] for v in all_vars])
    for vs, vals in log_factors:
        # Transpose vals so its variables appear in all_vars relative order
        present = [v for v in all_vars if v in vs]
        arr = np.transpose(vals, [vs.index(v) for v in present])
        # Insert size-1 axes for absent variables, at correct positions
        for i, v in enumerate(all_vars):
            if v not in vs:
                arr = np.expand_dims(arr, i)
        out = out + arr
    return all_vars, out


def _maxmin_eliminate_log(log_factors, vars_to_eliminate, mode):
    """Constrained max/min-elimination on a log-factor set. Returns a scalar.

    Intermediate factor size is bounded by the constrained treewidth of the
    elimination order — never exp(|vars_to_eliminate|) all at once.
    """
    assert mode in ("max", "min")
    op = np.max if mode == "max" else np.min
    fl = [(list(vs), vals) for vs, vals in log_factors]
    for var in vars_to_eliminate:
        with_var = [(vs, vals) for vs, vals in fl if var in vs]
        without = [(vs, vals) for vs, vals in fl if var not in vs]
        if with_var:
            cv, cvals = _combine_log_factors(with_var)
            new_vals = op(cvals, axis=cv.index(var))
            new_vars = [v for v in cv if v != var]
            without.append((new_vars, new_vals))
        fl = without
    # Remaining factors are scalars (over the empty variable set). In log
    # space, disconnected log-factors combine by addition.
    total = 0.0
    for vs, vals in fl:
        total += float(vals if vs == [] else np.asarray(vals).flatten()[0])
    return total


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def chen_sdp_exact(model, D, d_value, evidence, threshold, partitions):
    """
    Exact SDP via Chen et al. (2014)

    Parameters
    ----------
    model : pgmpy BayesianNetwork
    D : str   the decision variable
    d_value : str   the value of D defining the hypothesis
    evidence : dict[str, str]   evidence assignment
    threshold : float   probability threshold T
    partitions : iterable of iterable of str
        d-separated partitions of the hidden variable set H given D and E.

    Returns
    -------
    float : the same-decision probability.
    """

    # ── 1. Initial log-odds via VE on the ancestral subgraph ──────────────
    d_states = model.get_cpds(D).state_names[D]
    d_index = d_states.index(d_value)
    not_d_value = d_states[1 - d_index]

    relevant_nodes = list(evidence.keys()) + [D]
    ancestral = model.get_ancestral_graph(relevant_nodes)
    sub_model = BayesianNetwork(ancestral.edges())
    sub_model.add_nodes_from(ancestral.nodes())
    for node in sub_model.nodes():
        sub_model.add_cpds(model.get_cpds(node))
    sub_inf = VariableElimination(sub_model)
    initial = sub_inf.query(
        variables=[D], evidence=evidence,
        elimination_order="MinFill", show_progress=False,
    )
    p_d_e = float(initial.get_value(**{D: d_value}))
    p_not_d_e = float(initial.get_value(**{D: not_d_value}))

    if p_not_d_e == 0:
        return 1.0
    if p_d_e == 0:
        return 0.0

    log_O_d_e = math.log(p_d_e / p_not_d_e)
    lambda_threshold = math.log(threshold / (1 - threshold))
    current_decision_positive = (log_O_d_e >= lambda_threshold)

    # ── 2. Per-partition setup ────────────────────────────────────────────
    # For each partition S_i we build:
    #   psi: factor list with ∏ψ_j = Pr(S_i, d, e^i)
    #   phi: factor list with ∏φ_j = Pr(S_i, ¬d, e^i)
    #   Z_d, Z_nd: normalisers Pr(d, e^i) and Pr(¬d, e^i)
    #   max_w, min_w: constrained max/min log-ratio over S_i, used for bounds

    def _relevant_cpds(s_i_list):
        """CPDs of variables in the ancestral graph of S_i ∪ E ∪ {D}.

        Anything outside this set has CPDs that sum to 1 over the
        variables we'd marginalise, so they can be dropped without
        affecting Pr(S_i, d, e^i)."""
        targets = list(s_i_list) + list(evidence.keys()) + [D]
        anc = model.get_ancestral_graph(targets)
        nodes = set(anc.nodes())
        return [cpd for cpd in model.get_cpds() if cpd.variable in nodes]

    def _ve_to_partition(s_i_list, branch_d_value):
        """Sum-eliminate every variable that isn't in s_i_list, returning the
        remaining factor list. Each factor is bounded by exp(w)."""
        target_evidence = {**evidence, D: branch_d_value}
        factors = [cpd.to_factor() for cpd in _relevant_cpds(s_i_list)]
        for f in factors:
            ov = [(v, target_evidence[v]) for v in f.variables
                  if v in target_evidence]
            if ov:
                f.reduce(ov, inplace=True)
        all_vars = set(v for f in factors for v in f.variables)
        to_eliminate = list(all_vars - set(s_i_list))
        # Min-fill on the elimination order would tighten w; simple order
        # below is correct but possibly looser. Substitute MinFill if needed.
        for var in to_eliminate:
            with_var = [f for f in factors if var in f.variables]
            without = [f for f in factors if var not in f.variables]
            if with_var:
                prod = with_var[0].copy()
                for f in with_var[1:]:
                    prod = prod * f
                prod.marginalize([var], inplace=True)
                without.append(prod)
            factors = without
        return factors

    def _sum_to_scalar(factors, vars_to_elim):
        """Sum-eliminate the given vars from `factors` (kept as factor list)
        and return the product of remaining scalar factor values."""
        fl = [f.copy() for f in factors]
        for var in vars_to_elim:
            with_var = [f for f in fl if var in f.variables]
            without = [f for f in fl if var not in f.variables]
            if with_var:
                prod = with_var[0].copy()
                for f in with_var[1:]:
                    prod = prod * f
                prod.marginalize([var], inplace=True)
                without.append(prod)
            fl = without
        total = 1.0
        for f in fl:
            total *= float(np.sum(f.values))
        return total

    def _factor_product_at(factors, instantiation):
        """Π factors evaluated at the given full instantiation of S_i."""
        prod = 1.0
        for f in factors:
            if not f.variables:
                prod *= float(np.sum(f.values))
                continue
            local = {v: instantiation[v] for v in f.variables}
            prod *= float(f.get_value(**local))
        return prod

    partitions_data = []
    for s_i in partitions:
        s_i_list = list(s_i)
        states = {v: model.get_cpds(v).state_names[v] for v in s_i_list}

        # 2a. Decomposed joints (factor lists, never multiplied together)
        psi = _ve_to_partition(s_i_list, d_value)
        phi = _ve_to_partition(s_i_list, not_d_value)

        # 2b. Normalisers via constrained sum-elimination of S_i
        Z_d = _sum_to_scalar(psi, s_i_list)
        Z_nd = _sum_to_scalar(phi, s_i_list)
        if Z_d <= 0 or Z_nd <= 0:
            # Degenerate partition; skip (shouldn't happen on well-formed input)
            continue

        # 2c. max_w, min_w via constrained max/min-elimination of S_i on log-
        # ratio factor set. Intermediate log-factors stay bounded by exp(w).
        log_chi = [_log_ratio_factor(fp, fq) for fp, fq in zip(psi, phi)]
        # log w_{s_i} = log[∏ψ/∏φ] + log(Z_nd/Z_d); the second term is a
        # constant scalar applied uniformly to every s_i, so we subtract it
        # AFTER the constrained elimination.
        zlog = math.log(Z_d) - math.log(Z_nd)
        max_w = _maxmin_eliminate_log(log_chi, s_i_list, "max") - zlog
        min_w = _maxmin_eliminate_log(log_chi, s_i_list, "min") - zlog

        partitions_data.append({
            "s_i_list": s_i_list,
            "states":   states,
            "psi":      psi,
            "phi":      phi,
            "Z_d":      Z_d,
            "Z_nd":     Z_nd,
            "max_w":    max_w,
            "min_w":    min_w,
        })

    # ── 3. Order partitions by weight range (wider first → better pruning) ─
    partitions_data.sort(key=lambda pd: pd["max_w"] - pd["min_w"], reverse=True)

    n_parts = len(partitions_data)
    suffix_max = [0.0] * (n_parts + 1)
    suffix_min = [0.0] * (n_parts + 1)
    for i in range(n_parts - 1, -1, -1):
        suffix_max[i] = suffix_max[i + 1] + partitions_data[i]["max_w"]
        suffix_min[i] = suffix_min[i + 1] + partitions_data[i]["min_w"]

    # ── 4. DFS — weights computed on demand from the factor set ───────────
    EPS = 1e-300

    def dfs(depth, current_log_odds, prob_cond_d, prob_cond_not_d):
        ub = current_log_odds + suffix_max[depth]
        lb = current_log_odds + suffix_min[depth]
        prob_q = lambda: p_d_e * prob_cond_d + p_not_d_e * prob_cond_not_d

        # Bound-based pruning 
        if current_decision_positive:
            if lb >= lambda_threshold:
                return prob_q()
            if ub < lambda_threshold:
                return 0.0
        else:
            if ub < lambda_threshold:
                return prob_q()
            if lb >= lambda_threshold:
                return 0.0

        if depth == n_parts:
            same = (current_log_odds >= lambda_threshold) == current_decision_positive
            return prob_q() if same else 0.0

        part = partitions_data[depth]
        s_i_list = part["s_i_list"]
        states = part["states"]
        Z_d = part["Z_d"]
        Z_nd = part["Z_nd"]
        psi = part["psi"]
        phi = part["phi"]

        total = 0.0
        # Enumerate s_i values via Cartesian product. For each instantiation,
        # look up the joint by indexing into the factor set — no precomputed
        # flat list, no exp(|S_i|) memory allocation.
        for combo in itertools.product(*[states[v] for v in s_i_list]):
            inst = dict(zip(s_i_list, combo))
            psi_val = _factor_product_at(psi, inst)
            phi_val = _factor_product_at(phi, inst)
            p_d_si = max(psi_val / Z_d, EPS)
            p_nd_si = max(phi_val / Z_nd, EPS)
            # Skip configurations with zero probability under BOTH branches
            if p_d_si <= EPS and p_nd_si <= EPS:
                continue
            w_si = math.log(p_d_si) - math.log(p_nd_si)
            total += dfs(
                depth + 1,
                current_log_odds + w_si,
                prob_cond_d * p_d_si,
                prob_cond_not_d * p_nd_si,
            )
        return total

    return dfs(0, log_O_d_e, 1.0, 1.0)