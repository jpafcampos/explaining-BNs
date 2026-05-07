"""
marginalization_experiment.py
------------------------------
Holds Win95pts original CPTs fixed. Marginalizes out each hidden variable
one at a time and tests whether bimodality disappears.

For each hidden variable v:
  1. Compute P(c | parents(c) minus {v}) for every child c of v by summing
     v out of the joint P(c | parents(c)) * P(v | parents(v)).
     This preserves the joint distribution over remaining variables exactly.
  2. Remove v from the network and rewire: parents(v) -> each child of v.
  3. Run MCMC diagnostics on the reduced network.
  4. Compute exact SDP on the reduced network for ground truth.
  5. Record delta_final, error, R-hat, and failure mode.

Priority order: nodes with most children first (most likely explaining-away
bottlenecks), then by number of parents (most fill-in), then leaf nodes.

Leaf nodes (no children) are included but flagged: marginalizing them cannot
affect bimodality since they have no downstream influence.
"""

import copy
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from pgmpy.utils import get_example_model
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD

import numpy as np
import matplotlib.pyplot as plt
import random
import math
import networkx as nx
from monte_carlo_sdp import *
from same_decision_probability_calculation import *


def run_mcmc_diagnostics_v2(bn, target, target_value, patient, threshold,
                             n_iterations=100000, n_chains=4,
                             use_lw_seed=True):
    """
    Runs MCMC diagnostics for the NumPy-vectorised fast_mcmc_sdp_estimation.

    Uses the same integer-indexed inner loop as the production MCMC,
    so the diagnostics reflect the actual chain behaviour.

    Produces three plots:
      1. Trace plot  — running decision fraction per chain (burn-in)
      2. ACF         — autocorrelation of decision indicator (thinning)
      3. R-hat       — Gelman-Rubin convergence over time

    Parameters
    ----------
    use_lw_seed : bool
        If True, seeds via fast ancestral sampling (same as production).
        If False, uses pure random init (for comparison).
    """
    hidden_vars       = [n for n in bn.nodes() if n not in patient and n != target]
    target_states_list = bn.get_cpds(target).state_names[target]
    n_target_states   = len(target_states_list)
    target_state_idx  = list(range(n_target_states))

    # ── Build integer-indexed CPD tables (same as production MCMC) ───────────
    all_nodes   = list(bn.nodes())
    state_index = {}
    for n in all_nodes:
        cpd = bn.get_cpds(n)
        state_index[n] = {s: i for i, s in enumerate(cpd.state_names[n])}

    cpd_array = {}
    cpd_vars  = {}
    for n in all_nodes:
        cpd = bn.get_cpds(n)
        arr = np.asarray(cpd.values, dtype=float)
        arr[arr == 0.0] = 1e-10
        arr = arr / arr.sum(axis=0, keepdims=True)
        cpd_array[n] = arr
        cpd_vars[n]  = list(cpd.variables)

    children_cache = {v: list(bn.get_children(v)) for v in hidden_vars}
    affected_cache = {v: [v] + children_cache[v] for v in hidden_vars}

    inv_state = {v: {i: s for s, i in state_index[v].items()} for v in bn.nodes()}

    # Patient integer lookup
    patient_idx = {v: state_index[v][val] for v, val in patient.items()}

    # ── Helpers ───────────────────────────────────────────────────────────────
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

    def get_seed(use_lw):
        idx = dict(patient_idx)
        if use_lw:
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
                    sampled = np.random.choice(len(probs), p=probs)
                    sample[node]   = inv_state[node][sampled]
                    idx[node]      = sampled
            except Exception as e:
                print(f"    [SEED] Ancestral seed failed ({e}) — random fallback")
                for v in hidden_vars:
                    idx[v] = random.randrange(cpd_array[v].shape[0])
        else:
            for v in hidden_vars:
                idx[v] = random.randrange(cpd_array[v].shape[0])
        return idx

    def run_chain(chain_idx):
        print(f"  Chain {chain_idx + 1}/{n_chains}...")
        current_idx   = get_seed(use_lw_seed)
        current_lj    = [full_log_joint(current_idx, t) for t in target_state_idx]
        current_log_p = log_sum(current_lj)
        print("CURRENT LOG P ")
        print(current_log_p)

        decisions = []

        for i in range(n_iterations):
            var = random.choice(hidden_vars)
            cur_val    = current_idx[var]
            cardinality = cpd_array[var].shape[0]
            if cardinality < 2:
                pass
            else:
                new_val = random.randrange(cardinality - 1)
                if new_val >= cur_val:
                    new_val += 1

                proposed_lj = [0.0] * n_target_states
                recompute   = False

                for t_idx in target_state_idx:
                    if current_lj[t_idx] == float('-inf'):
                        recompute = True
                        break
                    delta  = 0.0
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
                            failed = True; break
                        if p_old == 0.0:
                            recompute = True; break
                        delta += math.log(p_new) - math.log(p_old)
                    if recompute:
                        break
                    if failed:
                        proposed_lj[t_idx] = float('-inf')
                        continue
                    proposed_lj[t_idx] = current_lj[t_idx] + delta

                if recompute:
                    tmp = dict(current_idx)
                    tmp[var]     = new_val
                    proposed_lj  = [full_log_joint(tmp, t) for t in target_state_idx]

                proposed_log_p = log_sum(proposed_lj)
                log_alpha      = proposed_log_p - current_log_p

                if log_alpha >= 0 or (
                    proposed_log_p != float('-inf')
                    and math.log(random.random()) < log_alpha
                ):
                    current_idx[var] = new_val
                    current_lj       = proposed_lj
                    current_log_p    = proposed_log_p

            # Record decision at every iteration
            sample_h = {v: inv_state[v][current_idx[v]] for v in hidden_vars}
            full_ev  = {**patient, **sample_h}
            p_target = get_exact_target_posterior_O1(bn, target, target_value, full_ev)
            decisions.append(1 if p_target >= threshold else 0)

        return decisions

    # ── Run all chains ────────────────────────────────────────────────────────
    all_chains = np.array([run_chain(c) for c in range(n_chains)], dtype=float)

    # ── Plot 1: Trace plot (running mean) ─────────────────────────────────────
    running_means = np.cumsum(all_chains, axis=1) / np.arange(1, n_iterations + 1)

    fig, axes = plt.subplots(3, 1, figsize=(13, 13))
    seed_label = "ancestral seed" if use_lw_seed else "random seed"

    ax = axes[0]
    for c in range(n_chains):
        ax.plot(running_means[c], alpha=0.7, label=f'Chain {c+1}')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Running mean of decision indicator')
    ax.set_title(f'Trace plot — {bn.name} ({seed_label})\n'
                 f'burn-in = iteration where all chains stabilise')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Plot 2: ACF ───────────────────────────────────────────────────────────
    MAX_LAG     = 1000
    signal      = all_chains[0] - all_chains[0].mean()
    acf_raw     = np.correlate(signal, signal, mode='full')
    acf_raw     = acf_raw[len(acf_raw) // 2:]
    acf_vals    = acf_raw / acf_raw[0]
    significance = 1.96 / math.sqrt(n_iterations)

    ax = axes[1]
    ax.bar(np.arange(MAX_LAG), acf_vals[:MAX_LAG], color='steelblue', alpha=0.7)
    ax.axhline( significance, color='red', ls='--', label='95% significance band')
    ax.axhline(-significance, color='red', ls='--')
    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    ax.set_title(f'ACF — chain 1\n'
                 f'thinning = smallest lag where ACF enters the red band')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Plot 3: Gelman-Rubin R-hat ────────────────────────────────────────────
    rhat_values  = []
    check_points = list(range(100, n_iterations, max(100, n_iterations // 500)))

    for t in check_points:
        chains_t     = all_chains[:, :t]
        n, m         = chains_t.shape[1], n_chains
        chain_means  = chains_t.mean(axis=1)
        grand_mean   = chain_means.mean()
        B            = n / (m - 1) * np.sum((chain_means - grand_mean) ** 2)
        W            = np.mean(np.var(chains_t, axis=1, ddof=1))
        var_hat      = (1 - 1/n) * W + B / n
        rhat         = math.sqrt(var_hat / W) if W > 0 else float('nan')
        rhat_values.append(rhat)

    ax = axes[2]
    ax.plot(check_points, rhat_values, color='darkgreen', lw=1.5)
    ax.axhline(1.1, color='red',  ls='--', label='R-hat=1.1 (warning)')
    ax.axhline(1.0, color='gray', ls=':')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Gelman-Rubin R-hat')
    ax.set_title('R-hat convergence\n(should drop and stay below 1.1)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    #plt.suptitle(f'MCMC Diagnostics v2 — {bn.name}  [{seed_label}]',
    #             fontsize=13, fontweight='bold')
    #plt.tight_layout()
    ##plt.savefig(f'mcmc_diagnostics_{bn.name.replace(" ", "_")}.png',
    ##            dpi=130, bbox_inches='tight')
    #plt.show()

    # ── Suggested parameters ──────────────────────────────────────────────────
    final_means = running_means[:, -1]
    stable      = np.all(np.abs(running_means - final_means[:, None]) < 0.05, axis=0)
    suggested_burnin   = int(np.argmax(stable)) if stable.any() else n_iterations // 2

    below = np.where(np.abs(acf_vals[1:MAX_LAG]) < significance)[0]
    suggested_thinning = int(below[0]) + 1 if len(below) > 0 else MAX_LAG

    final_rhat = rhat_values[-1]
    print(f"\n{'='*55}")
    print(f"Network      : {bn.name}")
    print(f"Seed         : {seed_label}")
    print(f"Hidden vars  : {len(hidden_vars)}")
    print(f"Suggested burn-in  : {suggested_burnin}")
    print(f"Suggested thinning : {suggested_thinning}")
    print(f"Final R-hat        : {final_rhat:.4f}  "
          f"{'✓ converged' if final_rhat < 1.1 else '✗ NOT converged'}")
    print(f"{'='*55}\n")

    return {
        'burn_in':   suggested_burnin,
        'thinning':  suggested_thinning,
        'rhat':      final_rhat,
        'chains':    all_chains,
    }

def select_optimal_target_node(bn):
    """
    Selects a target node deeply embedded in the network (highest degree).
    Highly connected nodes provide a smooth, responsive probability gradient 
    for the Hill Climber, making extreme SDP values (0.90+) reachable.
    """
    best_node = None
    max_degree = -1
    
    for node in bn.nodes():
        # Ensure it's a binary node
        if len(bn.get_cpds(node).state_names[node]) != 2:
            continue
            
        degree = len(bn.get_parents(node)) + len(bn.get_children(node))
        if degree > max_degree:
            max_degree = degree
            best_node = node
            
    # Fallback if no binary nodes exist (rare)
    if best_node is None:
        return random.choice(list(bn.nodes()))
        
    return best_node

# ---------------------------------------------------------------------------
# Core: marginalize one node out of a BayesianNetwork
# ---------------------------------------------------------------------------

def marginalize_node(bn, node):
    """
    Return a new BayesianNetwork with `node` marginalized out exactly.

    For each child c of `node`:
        new_parents(c) = (parents(c) - {node}) union parents(node)
        new P(c | new_parents(c)) = sum_{node} P(c | parents(c)) * P(node | parents(node))

    Key insight: parents of `node` that are NOT already parents of `c` must
    still appear in new_parents(c) after marginalization -- they appear in
    P(node | parents(node)) and survive the summation over node's states.
    We must NOT average them out; doing so would introduce false independence.

    Parameters
    ----------
    bn   : BayesianNetwork
    node : str  -- variable to marginalize out

    Returns
    -------
    new_bn       : BayesianNetwork | None
    success      : bool
    reason       : str
    max_new_cols : int   -- tractability metric
    """
    parents_of_v  = list(bn.get_parents(node))
    children_of_v = [c for c in bn.get_children(node) if c != node]
    cpd_v         = bn.get_cpds(node)
    card_v        = cpd_v.variable_card

    # ── Tractability check ────────────────────────────────────────────────
    # Worst-case new CPD size for each child.
    max_new_cols = 0
    for ch in children_of_v:
        new_par = (set(bn.get_parents(ch)) - {node}) | set(parents_of_v)
        cols = 1
        for p in new_par:
            cols *= bn.get_cpds(p).variable_card
        max_new_cols = max(max_new_cols, cols)
    if max_new_cols > 500_000:
        return None, False, f"fill-in too large ({max_new_cols:,} cols)", max_new_cols

    # ── New edge list ─────────────────────────────────────────────────────
    new_edges = [(u, w) for u, w in bn.edges() if u != node and w != node]
    for pv in parents_of_v:
        for ch in children_of_v:
            if (pv, ch) not in new_edges:
                new_edges.append((pv, ch))

    # ── Construct new network ─────────────────────────────────────────────
    surviving = [n for n in bn.nodes() if n != node]
    new_bn = BayesianNetwork(new_edges)
    for n in surviving:
        if n not in new_bn.nodes():
            new_bn.add_node(n)

    # ── Copy unchanged CPDs ───────────────────────────────────────────────
    for n in surviving:
        if n not in children_of_v:
            new_bn.add_cpds(copy.deepcopy(bn.get_cpds(n)))

    # ── Compute new CPD for each child c of node ──────────────────────────
    #
    # We want:
    #   P(c | new_parents_c) = sum_v P(c | old_parents_c) * P(v | parents_v)
    #
    # Array dimensions:
    #   cpd_ch.values  shape: (card_c, card_op1, ..., card_v, ..., card_opk)
    #                         axes follow cpd_ch.variables = [c, op1, ..., v, ..., opk]
    #   cpd_v.values   shape: (card_v, card_pv1, ..., card_pvP)
    #                         axes follow cpd_v.variables = [v, pv1, ..., pvP]
    #
    # Strategy:
    #   1. Build a joint array over all unique variables that appear in
    #      either CPD (except c and v themselves as the axes to keep/sum).
    #   2. We do this by creating an explicit (card_c, card_op*, card_pv*)
    #      array where:
    #        - old parents of c that are not v  -> axes from cpd_ch
    #        - parents of v                     -> axes from cpd_v
    #        - v itself                         -> the summation axis
    #   3. Sum over v axis -> marginalised array of shape (card_c, new_par_axes)
    #   4. Build TabularCPD with new_parents_c in canonical order.

    cpd_v_vars    = cpd_v.variables    # [v, pv1, pv2, ...]
    cpd_v_parents = cpd_v_vars[1:]     # [pv1, pv2, ...]
    cpd_v_arr     = np.asarray(cpd_v.values, dtype=float)
    # cpd_v_arr shape: (card_v, card_pv1, ..., card_pvP)

    for ch in children_of_v:
        cpd_ch      = bn.get_cpds(ch)
        card_ch     = cpd_ch.variable_card
        old_vars_ch = cpd_ch.variables          # [ch, op1, ..., v, ..., opk]
        old_parents = old_vars_ch[1:]           # [op1, ..., v, ..., opk]
        node_pos    = old_parents.index(node)
        node_axis   = node_pos + 1              # axis of v in cpd_ch array

        # Old parents of ch excluding v
        old_par_no_v = [p for p in old_parents if p != node]

        # New parent set in canonical order:
        # keep old parents (minus v) first, then add parents_of_v not already present
        new_parents_ch = list(old_par_no_v)
        for pv in cpd_v_parents:
            if pv not in new_parents_ch:
                new_parents_ch.append(pv)

        # We'll build the result by explicit iteration over all parent configs.
        # This is unambiguous and handles all edge cases.
        # Shape: (card_ch, prod(card(p) for p in new_parents_ch))
        new_par_cards  = [bn.get_cpds(p).variable_card for p in new_parents_ch]
        n_new_cols     = int(np.prod(new_par_cards)) if new_par_cards else 1
        result         = np.zeros((card_ch, n_new_cols), dtype=float)

        # Index mapping helpers
        # col_index(new_par_config) -> column index in result
        def col_index(config_dict, parents, cards):
            idx = 0
            for p, c in zip(parents, cards):
                idx = idx * c + config_dict[p]
            return idx

        # For old cpd_ch: given old_par_config, get slice over v axis
        # old_parents order: [op1..opk_no_v in original order, v inserted at node_pos]
        old_par_no_v_ordered = [p for p in old_parents if p != node]
        old_par_no_v_cards   = [bn.get_cpds(p).variable_card
                                 for p in old_par_no_v_ordered]

        # Rearrange cpd_ch to have axes [ch, v, op1..opk_no_v_in_order]
        # Current axes: [ch, op1..opk_with_v_at_node_pos]
        # Move node_axis to position 1
        cpd_ch_arr = np.asarray(cpd_ch.values, dtype=float)
        axes_order = [0, node_axis] + [i for i in range(1, len(old_vars_ch))
                                        if i != node_axis]
        cpd_ch_reordered = np.transpose(cpd_ch_arr, axes_order)
        # shape: (card_ch, card_v, *old_par_no_v_cards_in_original_order)

        # Rearrange cpd_v to axes [v, pv_in_order_of_cpd_v_parents]
        # (already in this order: [v, pv1, pv2...])
        # We need to iterate over all combinations of new_parents_ch
        # and for each compute sum_v P(ch|old_par)*P(v|pv_parents).

        if not new_parents_ch:
            # No parents at all -> scalar CPD
            # sum_v P(ch|v) * P(v)  (no other parents)
            p_v = cpd_v_arr.flatten()  # shape (card_v,) — cpd_v has no parents when new_parents_ch is empty
            # cpd_ch_reordered shape: (card_ch, card_v)
            col_vals = np.einsum('cv,v->c', cpd_ch_reordered.reshape(card_ch, card_v), p_v)
            result[:, 0] = col_vals
        else:
            # Iterate over all combinations of new_parents_ch
            for new_par_combo in np.ndindex(*new_par_cards):
                new_par_dict = {p: new_par_combo[i]
                                for i, p in enumerate(new_parents_ch)}

                # Index into cpd_ch_reordered: needs old_par_no_v values
                # old_par_no_v_ordered is a subset of new_parents_ch
                old_par_idx = tuple(new_par_dict[p] for p in old_par_no_v_ordered)
                # cpd_ch_reordered[ch_state, v_state, *old_par_idx]
                if old_par_no_v_ordered:
                    ch_given_v = cpd_ch_reordered[(slice(None), slice(None)) + old_par_idx]
                else:
                    ch_given_v = cpd_ch_reordered  # shape (card_ch, card_v)
                # ch_given_v shape: (card_ch, card_v)

                # Index into cpd_v_arr: needs pv values
                if cpd_v_parents:
                    pv_idx = tuple(new_par_dict[p] for p in cpd_v_parents)
                    p_v = cpd_v_arr[(slice(None),) + pv_idx]  # shape (card_v,)
                else:
                    p_v = cpd_v_arr.flatten()  # shape (card_v,)

                # Marginalise: sum_v P(ch|v, old_par)*P(v|pv_par)
                col_vals = ch_given_v @ p_v  # shape (card_ch,)

                # Place in result column
                c_idx = col_index(new_par_dict, new_parents_ch, new_par_cards)
                result[:, c_idx] = col_vals

        # Normalise (should be ~1 already, but guard against float drift)
        col_sums = result.sum(axis=0, keepdims=True)
        col_sums = np.where(col_sums == 0, 1.0, col_sums)
        result /= col_sums

        # Build state_names
        state_names = {ch: cpd_ch.state_names[ch]}
        for p in new_parents_ch:
            state_names[p] = bn.get_cpds(p).state_names[p]

        new_cpd = TabularCPD(
            variable=ch,
            variable_card=card_ch,
            values=result,
            evidence=new_parents_ch if new_parents_ch else None,
            evidence_card=new_par_cards if new_par_cards else None,
            state_names=state_names,
        )
        new_bn.add_cpds(new_cpd)

    try:
        ok = new_bn.check_model()
        assert ok
    except Exception as e:
        return None, False, f"check_model: {e}", max_new_cols

    return new_bn, True, "ok", max_new_cols


def get_priority_order(bn, hidden_vars):
    """
    Sort hidden variables by decreasing structural importance for bimodality.

    Primary key   : n_children (desc)  -- hub nodes are explaining-away centres
    Secondary key : n_parents  (desc)  -- more parents = richer explaining-away
    Tertiary      : alphabetical
    Leaf nodes (n_children == 0) are pushed to the end and flagged.
    """
    rows = []
    for v in hidden_vars:
        parents  = list(bn.get_parents(v))
        children = [c for c in bn.get_children(v) if c != v]
        rows.append((v, len(children), len(parents)))
    rows.sort(key=lambda r: (-r[1], -r[2], r[0]))
    return [(v, nc, np_) for v, nc, np_ in rows]


# ---------------------------------------------------------------------------
# Exact SDP helper
# ---------------------------------------------------------------------------

def compute_exact_sdp(bn, target, target_value, patient, threshold):
    try:
        hidden = [n for n in bn.nodes() if n not in patient and n != target]
        if not hidden:
            return None, False, "no hidden vars"
        partitions = get_partitions(bn, hidden, target, patient)
        sdp = fast_broadcast_sdp(bn, target, target_value, patient, threshold, partitions)
        return float(sdp), True, "ok"
    except Exception as e:
        return None, False, str(e)


# ---------------------------------------------------------------------------
# Chain summary
# ---------------------------------------------------------------------------

def summarise_chains(chains, exact_sdp=None):
    running    = np.cumsum(chains, axis=1) / np.arange(1, chains.shape[1] + 1)
    finals     = running[:, -1]
    delta      = float(finals.max() - finals.min())
    chain_mean = float(finals.mean())
    error      = abs(chain_mean - exact_sdp) if exact_sdp is not None else None
    wrong_flag = (error is not None) and (error >= 0.10) and (delta < 0.10)
    return {
        'finals':       finals,
        'chain_mean':   chain_mean,
        'delta_final':  delta,
        'bimodal_flag': delta >= 0.10,
        'error':        error,
        'wrong_flag':   wrong_flag,
        'any_failure':  delta >= 0.10 or wrong_flag,
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_marginalization_experiment(
    bn,
    target,
    target_value,
    base_patient,
    threshold=0.5,
    n_iterations=7_000,
    n_chains=4,
    output_dir="marginalization_results",
    use_lw_seed=True,
):
    """
    For each hidden variable, marginalize it out and run MCMC diagnostics
    on the resulting network.  Records how much bimodality changes.

    Baseline (full network) is run first so every row can report
    delta vs baseline.
    """
    os.makedirs(output_dir, exist_ok=True)

    hidden_vars = [n for n in bn.nodes()
                   if n not in base_patient and n != target]
    priority    = get_priority_order(bn, hidden_vars)

    print(f"Hidden variables: {len(hidden_vars)}")
    print(f"Running marginalization experiment ({len(priority)} candidates)\n")

    # ── Baseline ─────────────────────────────────────────────────────────
    print("="*65)
    print("BASELINE — full network, no marginalization")
    print("="*65)
    exact_base, ok_base, _ = compute_exact_sdp(
        bn, target, target_value, base_patient, threshold)
    if ok_base:
        print(f"  Exact SDP (baseline): {exact_base:.4f}")
    else:
        print(f"  Exact SDP: unavailable")

    diag_base = run_mcmc_diagnostics_v2(
        bn, target, target_value, base_patient, threshold,
        n_iterations=n_iterations, n_chains=n_chains,
        use_lw_seed=use_lw_seed,
    )
    base_summary = summarise_chains(diag_base['chains'], exact_sdp=exact_base)
    print(f"  delta={base_summary['delta_final']:.3f}  "
          f"R={diag_base['rhat']:.3f}  "
          f"mean={base_summary['chain_mean']:.3f}  "
          f"error={base_summary['error']:.3f}")

    rows = [{
        'node':           '__baseline__',
        'n_children':     -1,
        'n_parents':      -1,
        'is_leaf':        False,
        'marginalize_ok': True,
        'skip_reason':    '',
        'max_new_cols':   0,
        'exact_sdp':      exact_base,
        'rhat':           diag_base['rhat'],
        'delta_final':    base_summary['delta_final'],
        'chain_mean':     base_summary['chain_mean'],
        'error':          base_summary['error'],
        'bimodal_flag':   base_summary['bimodal_flag'],
        'wrong_flag':     base_summary['wrong_flag'],
        'any_failure':    base_summary['any_failure'],
        'delta_vs_base':  0.0,
        'error_vs_base':  0.0,
        'n_hidden_after': len(hidden_vars),
    }]

    # ── Per-node loop ─────────────────────────────────────────────────────
    n_total = len(priority)
    for idx, (node, n_ch, n_par) in enumerate(priority):
        is_leaf = (n_ch == 0)
        print(f"\n{'='*65}")
        print(f"[{idx+1}/{n_total}] Marginalizing: {node}  "
              f"(ch={n_ch}, par={n_par}, leaf={is_leaf})")
        print(f"{'='*65}")

        new_bn, success, reason, max_cols = marginalize_node(bn, node)

        if not success:
            print(f"  SKIPPED: {reason}")
            rows.append({
                'node': node, 'n_children': n_ch, 'n_parents': n_par,
                'is_leaf': is_leaf, 'marginalize_ok': False,
                'skip_reason': reason, 'max_new_cols': max_cols,
                'exact_sdp': None, 'rhat': None, 'delta_final': None,
                'chain_mean': None, 'error': None, 'bimodal_flag': None,
                'wrong_flag': None, 'any_failure': None,
                'delta_vs_base': None, 'error_vs_base': None,
                'n_hidden_after': len(hidden_vars) - 1,
            })
            continue

        # New hidden vars (node is gone)
        new_hidden = [n for n in new_bn.nodes()
                      if n not in base_patient and n != target]

        # Exact SDP on reduced network
        exact_new, ok_exact, exact_msg = compute_exact_sdp(
            new_bn, target, target_value, base_patient, threshold)
        if ok_exact:
            print(f"  Exact SDP (reduced): {exact_new:.4f}  "
                  f"(was {exact_base:.4f})" if exact_base else "")
        else:
            print(f"  Exact SDP: unavailable ({exact_msg})")

        # MCMC diagnostics
        new_bn.name = f"win95pts_no_{node}"
        diag = run_mcmc_diagnostics_v2(
            new_bn, target, target_value, base_patient, threshold,
            n_iterations=n_iterations, n_chains=n_chains,
            use_lw_seed=use_lw_seed,
        )
        s = summarise_chains(diag['chains'], exact_sdp=exact_new)

        d_vs_base = (s['delta_final'] - base_summary['delta_final']
                     if s['delta_final'] is not None else None)
        e_vs_base = ((s['error'] - base_summary['error'])
                     if s['error'] is not None and base_summary['error'] is not None
                     else None)

        status = ('BIMODAL' if s['bimodal_flag']
                  else 'WRONG'  if s['wrong_flag']
                  else 'ok')
        resolved = (base_summary['any_failure'] and not s['any_failure'])

        err_str = f"{s['error']:.3f}" if s['error'] is not None else "N/A"
        print(f"  delta={s['delta_final']:.3f}  R={diag['rhat']:.3f}  "
            f"mean={s['chain_mean']:.3f}  error={err_str}  [{status}]"
            + ("  *** BIMODALITY RESOLVED ***" if resolved else ""))

        rows.append({
            'node':           node,
            'n_children':     n_ch,
            'n_parents':      n_par,
            'is_leaf':        is_leaf,
            'marginalize_ok': True,
            'skip_reason':    '',
            'max_new_cols':   max_cols,
            'exact_sdp':      exact_new,
            'rhat':           diag['rhat'],
            'delta_final':    s['delta_final'],
            'chain_mean':     s['chain_mean'],
            'error':          s['error'],
            'bimodal_flag':   s['bimodal_flag'],
            'wrong_flag':     s['wrong_flag'],
            'any_failure':    s['any_failure'],
            'delta_vs_base':  d_vs_base,
            'error_vs_base':  e_vs_base,
            'n_hidden_after': len(new_hidden),
        })

        # Save CSV incrementally
        pd.DataFrame(rows).to_csv(
            os.path.join(output_dir, "marginalization_results.csv"), index=False)

    df = pd.DataFrame(rows)

    # ── Summary figure ────────────────────────────────────────────────────
    _plot_summary(df, base_summary, output_dir, n_chains)

    # ── Print ranked table ────────────────────────────────────────────────
    _print_table(df, base_summary)

    return df


# ---------------------------------------------------------------------------
# Summary figure
# ---------------------------------------------------------------------------

def _plot_summary(df, base_summary, output_dir, n_chains):
    results = df[df['node'] != '__baseline__'].copy()
    results = results[results['marginalize_ok'] == True].copy()
    results = results.sort_values('delta_final')

    if len(results) == 0:
        return

    PALETTE = {'ok': '#3B6D11', 'bimodal': '#A32D2D', 'wrong': '#534AB7',
               'leaf': '#888780', 'skip': '#B0A090'}

    def row_color(r):
        if r['is_leaf']:           return PALETTE['leaf']
        if r['bimodal_flag']:      return PALETTE['bimodal']
        if r['wrong_flag']:        return PALETTE['wrong']
        return PALETTE['ok']

    colors = [row_color(r) for _, r in results.iterrows()]
    labels = results['node'].str.replace('_', ' ').tolist()

    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, len(results) * 0.38 + 1.5)))
    fig.suptitle(
        'Win95pts — effect of marginalizing each hidden variable\n'
        'sorted by delta_final (lower = more unimodal)',
        fontsize=11, fontweight='bold'
    )

    # Left panel: delta_final
    ax = axes[0]
    bars = ax.barh(labels, results['delta_final'].values,
                   color=colors, alpha=0.85, edgecolor='black', linewidth=0.4)
    ax.axvline(base_summary['delta_final'], color='black', lw=1.2,
               ls='--', label=f"baseline Δ={base_summary['delta_final']:.3f}")
    ax.axvline(0.10, color='#B85C00', lw=0.8, ls=':', label='bimodal threshold 0.10')
    ax.set_xlabel('delta_final (chain spread)', fontsize=9)
    ax.set_title('Chain spread after marginalization', fontsize=9)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.2, axis='x')
    ax.spines[['top', 'right']].set_visible(False)

    # Right panel: absolute error vs exact SDP
    ax = axes[1]
    err_vals = results['error'].fillna(0).values
    ax.barh(labels, err_vals, color=colors, alpha=0.85,
            edgecolor='black', linewidth=0.4)
    if base_summary['error'] is not None:
        ax.axvline(base_summary['error'], color='black', lw=1.2, ls='--',
                   label=f"baseline err={base_summary['error']:.3f}")
    ax.axvline(0.10, color='#B85C00', lw=0.8, ls=':', label='error threshold 0.10')
    ax.set_xlabel('|chain mean − exact SDP|', fontsize=9)
    ax.set_title('Absolute error after marginalization', fontsize=9)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)
    ax.set_yticklabels([])
    ax.grid(True, alpha=0.2, axis='x')
    ax.spines[['top', 'right']].set_visible(False)

    # Colour legend
    from matplotlib.patches import Patch
    legend_els = [
        Patch(facecolor=PALETTE['ok'],      label='ok (unimodal, accurate)'),
        Patch(facecolor=PALETTE['bimodal'], label='bimodal'),
        Patch(facecolor=PALETTE['wrong'],   label='wrong (unimodal, inaccurate)'),
        Patch(facecolor=PALETTE['leaf'],    label='leaf node (no children)'),
    ]
    fig.legend(handles=legend_els, loc='lower center', ncol=4,
               fontsize=8, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    path = os.path.join(output_dir, "marginalization_summary.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSummary figure saved -> {path}")


# ---------------------------------------------------------------------------
# Print table
# ---------------------------------------------------------------------------

def _print_table(df, base_summary):
    results = df[df['node'] != '__baseline__'].copy()
    ok      = results[results['marginalize_ok'] == True].copy()
    ok      = ok.sort_values('delta_final')

    print(f"\n{'='*80}")
    print("RANKED RESULTS — sorted by delta_final ascending")
    print(f"Baseline: delta={base_summary['delta_final']:.3f}  "
          f"error={base_summary['error']:.3f}")
    print(f"{'='*80}")
    print(f"  {'node':<25} {'ch':>3} {'par':>3}  "
          f"{'exact':>6}  {'delta':>6}  {'error':>6}  "
          f"{'Δ-base':>7}  {'status':<10}")
    print('-' * 80)
    for _, r in ok.iterrows():
        status = ('BIMODAL' if r['bimodal_flag']
                  else 'WRONG'  if r['wrong_flag']
                  else 'leaf'   if r['is_leaf']
                  else 'ok')
        exact_s  = f"{r['exact_sdp']:.3f}" if r['exact_sdp'] is not None else "N/A"
        delta_s  = f"{r['delta_final']:.3f}" if r['delta_final'] is not None else "N/A"
        error_s  = f"{r['error']:.3f}"        if r['error']      is not None else "N/A"
        dvbase_s = f"{r['delta_vs_base']:+.3f}" if r['delta_vs_base'] is not None else "N/A"
        print(f"  {r['node']:<25} {int(r['n_children']):>3} {int(r['n_parents']):>3}  "
              f"{exact_s:>6}  {delta_s:>6}  {error_s:>6}  "
              f"{dvbase_s:>7}  {status}")

    skipped = results[results['marginalize_ok'] == False]
    if len(skipped) > 0:
        print(f"\nSkipped ({len(skipped)}): "
              + ", ".join(skipped['node'].tolist()))
    print(f"{'='*80}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from pgmpy.utils import get_example_model

    win95pts_model = get_example_model('win95pts')
    target         = select_optimal_target_node(win95pts_model)
    target_states  = win95pts_model.get_cpds(target).state_names[target]
    target_value   = target_states[1]

    interesting_patient = {
        'TTOK': 'No', 'NtSpd': 'OK', 'PgOrnttnOK': 'Incorrect', 'NetOK': 'Yes',
        'TrTypFnts': 'No', 'PrtPath': 'Correct', 'PrtPaper': 'No_Paper',
        'Problem1': 'No_Output', 'AppData': 'Correct', 'PrtDataOut': 'Yes',
        'PSERRMEM': 'Low_Memory', 'Problem4': 'Yes', 'FntInstlltn': 'Verified',
        'NnTTOK': 'No', 'AppDtGnTm': 'Fast_Enough',
        'DskLocal': 'Greater_than_2_Mb', 'PrtSel': 'Yes', 'Problem5': 'No',
        'IncmpltPS': 'Yes', 'PrtIcon': 'Normal', 'GDIIN': 'Yes',
        'PrtData': 'Yes', 'PrtPScript': 'Yes',
        'HrglssDrtnAftrPrnt': 'Fast_Enough', 'AvlblVrtlMmry': 'Adequate____1Mb_',
        'PrtPort': 'Yes', 'GrbldOtpt': 'No', 'GDIOUT': 'Yes',
        'TnrSpply': 'Adequate', 'PrntrAccptsTrtyp': 'No', 'PrtOn': 'No',
        'Problem6': 'No', 'PrtSpool': 'Disabled',
        'TstpsTxt': 'x_1_Mb_Available_VM', 'PrtTimeOut': 'Long_Enough',
        'PrntPrcssTm': 'Too_Long', 'PrtStatToner': 'Low__None', 'DrvSet': 'Correct',
    }

    df = run_marginalization_experiment(
        bn=win95pts_model,
        target=target,
        target_value=target_value,
        base_patient=interesting_patient,
        threshold=0.5,
        n_iterations=15_000,
        n_chains=4,
        output_dir="marginalization_results",
        use_lw_seed=True,
    )