"""
Entropy computation for Bayesian Networks in pgmpy.

Based on:
    Scutari, M. "Entropy and the Kullback-Leibler Divergence for Bayesian
    Networks: Computational Complexity and Efficient Implementation."
    Algorithms 2023. arXiv:2312.01520v3

Supported model types
---------------------
* Discrete BN  (pgmpy.models.DiscreteBayesianNetwork with TabularCPD factors)
  Section 4.1 — O(N(w(1+l^w) + l^(w-1)) + |Theta|)


The dispatcher `entropy_bn` auto-detects the model type and calls the
appropriate routine.  All results are in **nats** (natural logarithm).
Multiply by 1/log(2) to convert to bits.

Compatibility
-------------
Tested with pgmpy 1.1.0.  In older versions (<1.0) `BayesianNetwork` was not
yet renamed to `DiscreteBayesianNetwork`; both are imported with a fallback so
the code works across versions.
"""

from __future__ import annotations

import numpy as np

import math

# pgmpy imports with version-compatibility shim
try:
    from pgmpy.models import DiscreteBayesianNetwork as _DBN
except ImportError:
    from pgmpy.models import BayesianNetwork as _DBN  # type: ignore[assignment]

from pgmpy.models import LinearGaussianBayesianNetwork as _LGBN
from pgmpy.inference import VariableElimination

# ──────────────────────────────────────────────────────────────────────────────
# Normalized mean conditional entropy (scale-invariant baseline)
# ──────────────────────────────────────────────────────────────────────────────

def mean_conditional_entropy(bn):
    """
    Computes the mean conditional entropy per node H(Xi | Pa(Xi)),
    averaged across all nodes in the network.
    Invariant to network size.
    """
    total_h = 0.0
    n = 0

    for cpd in bn.get_cpds():
        var    = cpd.variable
        states = cpd.state_names[var]
        n     += 1

        # cpd.values has shape (|Xi|, |Pa1|, |Pa2|, ...)
        # Reshape to (|Xi|, n_parent_configs)
        n_states = len(states)
        values   = cpd.values.reshape(n_states, -1)  # (states, parent_configs)
        n_configs = values.shape[1]

        # Marginal probability of each parent config P(pa)
        # For a fitted BN this requires the parent marginals, but a good
        # approximation that preserves the scale-invariance is the
        # uniform-weighted average over parent configurations
        node_h = 0.0
        for j in range(n_configs):
            col = values[:, j]  # P(Xi | pa_j)
            for p in col:
                if p > 0:
                    node_h -= p * math.log2(p)

        total_h += node_h / n_configs  # average over parent configs

    return total_h / n  # average over nodes

# ──────────────────────────────────────────────────────────────────────────────
# Internal helper
# ──────────────────────────────────────────────────────────────────────────────

def _safe_entropy(probs: np.ndarray) -> float:
    """H = -sum p*log(p), skipping zeros to avoid NaN."""
    p = np.asarray(probs, dtype=float).ravel()
    mask = p > 0.0
    return float(-np.dot(p[mask], np.log(p[mask])))


# ──────────────────────────────────────────────────────────────────────────────
# Discrete Bayesian Network  (Section 4.1)
# ──────────────────────────────────────────────────────────────────────────────

def entropy_discrete_bn(model: _DBN) -> float:
    """
    Shannon entropy of a discrete Bayesian Network.

    Implements the decomposition in Equation (9) of Scutari (2023):

        H(B)         = sum_i  H(Xi | Pi_Xi)
        H(Xi | Pi_Xi) = sum_j  P(Pi_Xi = j) * H(Xi | Pi_Xi = j)

    where H(Xi | Pi_Xi = j) = -sum_k  pi_{ik|j} * log pi_{ik|j}.

    The marginal probabilities P(Pi_Xi = j) are obtained via Variable
    Elimination.  For root nodes (no parents) this reduces to the standard
    marginal entropy.

    Complexity
    ----------
    O(N(w(1+l^w) + l^(w-1)) + |Theta|)
    where w = max clique size in the junction tree, l = max cardinality.

    Parameters
    ----------
    model : DiscreteBayesianNetwork
        A fitted pgmpy discrete BN with all CPDs attached.

    Returns
    -------
    float
        H(B) in nats.
    """
    inference = VariableElimination(model)
    total_entropy = 0.0

    for node in model.nodes():
        cpd = model.get_cpds(node)

        # cpd.variables = [node, evidence_0, evidence_1, ...]
        # The ordering of evidence axes in cpd.values matches this list.
        parents_in_cpd_order: list[str] = list(cpd.variables[1:])

        if not parents_in_cpd_order:
            # Root node: H(Xi) = -sum_k pi_k log pi_k
            total_entropy += _safe_entropy(cpd.values)

        else:
            # Non-root node:
            # Query the joint marginal of the parents.
            # We query in cpd.variables[1:] order so that parent_factor.values
            # axes align exactly with cpd.values axes 1, 2, ...
            parent_factor = inference.query(
                variables=parents_in_cpd_order,
                show_progress=False,
            )

            # parent_factor.values : (card_p0, card_p1, ...)
            # cpd.values           : (card_Xi, card_p0, card_p1, ...)
            parent_cards = tuple(
                len(cpd.state_names[p]) for p in parents_in_cpd_order
            )

            for cfg in np.ndindex(*parent_cards):
                p_cfg = float(parent_factor.values[cfg])
                if p_cfg < 1e-300:
                    continue  # negligible contribution, skip

                # P(Xi | parents = cfg) as a 1-D probability vector
                probs_xi = cpd.values[(slice(None),) + cfg]
                total_entropy += p_cfg * _safe_entropy(probs_xi)

    return total_entropy


def export_node_entropies_to_csv(model, filename="node_entropies.csv"):
    """
    Calculates H(Xi | Pi_Xi) for every node in a pgmpy Discrete BN 
    and exports the results to a CSV.
    """
    inference = VariableElimination(model)
    node_data = []

    for node in model.nodes():
        cpd = model.get_cpds(node)
        parents_in_cpd_order = list(cpd.variables[1:])
        
        node_entropy = 0.0

        if not parents_in_cpd_order:
            # Root node calculation
            node_entropy = _safe_entropy(cpd.values)
        else:
            # Conditional entropy: H(Xi | Pi_Xi)
            parent_factor = inference.query(
                variables=parents_in_cpd_order,
                show_progress=False,
            )

            parent_cards = tuple(
                len(cpd.state_names[p]) for p in parents_in_cpd_order
            )

            for cfg in np.ndindex(*parent_cards):
                p_cfg = float(parent_factor.values[cfg])
                if p_cfg < 1e-300:
                    continue

                # Slice CPD to get P(Xi | parents = cfg)
                probs_xi = cpd.values[(slice(None),) + cfg]
                node_entropy += p_cfg * _safe_entropy(probs_xi)

        # Store the result for this node
        node_data.append({
            "Node": node,
            "Entropy_Nats": node_entropy,
            "Parent_Count": len(parents_in_cpd_order)
        })

    # Create DataFrame and Export
    df = pd.DataFrame(node_data)
    df.to_csv(filename, index=False)
    
    print(f"Entropy report saved to {filename}")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Public dispatcher
# ──────────────────────────────────────────────────────────────────────────────

def entropy_bn(model) -> float:
    """
    Compute the Shannon entropy of a pgmpy Bayesian Network.


    Parameters
    ----------
    model
        A fitted pgmpy BN with CPDs attached.

    Returns
    -------
    float
        H(B) in nats.

    Raises
    ------
    TypeError
        If the model type is not supported.

    Examples
    --------
    >>> h = entropy_bn(my_discrete_model)
    >>> h = entropy_bn(my_gaussian_model)
    """
    if isinstance(model, _DBN):
        return entropy_discrete_bn(model)
    else:
        raise TypeError(
            f"Unsupported model type: {type(model).__name__}.\n"
            "Supported: DiscreteBayesianNetwork, LinearGaussianBayesianNetwork."
        )


# ──────────────────────────────────────────────────────────────────────────────
# Numerical verification — reproduces examples from the paper
# ──────────────────────────────────────────────────────────────────────────────

def _verify_discrete_example() -> float:
    """
    Reproduces Example B.3 / Figure 2 (top) of Scutari (2023).
    Expected H(B) = 2.440 nats.

    DAG:  X1 --+
               +--> X3 --> X4
          X2 --+
    """
    from pgmpy.factors.discrete import TabularCPD

    model = _DBN([("X1", "X3"), ("X2", "X3"), ("X3", "X4")])
    model.add_cpds(
        TabularCPD("X1", 2, [[0.53], [0.47]],
                   state_names={"X1": ["a", "b"]}),
        TabularCPD("X2", 2, [[0.34], [0.66]],
                   state_names={"X2": ["c", "d"]}),
        TabularCPD("X3", 2,
                   [[0.15, 0.75, 0.40, 0.80],
                    [0.85, 0.25, 0.60, 0.20]],
                   evidence=["X1", "X2"], evidence_card=[2, 2],
                   state_names={"X3": ["e", "f"],
                                "X1": ["a", "b"], "X2": ["c", "d"]}),
        TabularCPD("X4", 2,
                   [[0.20, 0.42],
                    [0.80, 0.58]],
                   evidence=["X3"], evidence_card=[2],
                   state_names={"X4": ["g", "h"], "X3": ["e", "f"]}),
    )
    assert model.check_model(), "Discrete model validation failed."
    h = entropy_discrete_bn(model)
    status = "OK" if abs(h - 2.440) < 0.01 else "FAIL"
    print(f"  [Discrete BN]   H(B) = {h:.4f} nats  (expected 2.440)  [{status}]")
    return h



import pandas as pd
from ucimlrepo import fetch_ucirepo
from pgmpy.models import NaiveBayes
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination, ApproxInference, BeliefPropagation
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.estimators import BayesianEstimator
from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import BDeuScore, K2Score, BicScore
from pgmpy.metrics import structure_score
from pgmpy.utils import get_example_model
from pgmpy.estimators import ScoreCache

if __name__ == "__main__":
    print("=" * 60)
    print("  Verifying entropy implementations vs. Scutari (2023)")
    print("=" * 60)
    _verify_discrete_example()
    # ── VOTING ──────────────────────────────────────────────────────────────────
    voting = fetch_ucirepo(id=105)
    df_voting = pd.concat([voting.data.features, voting.data.targets], axis=1)
    df_voting.columns = [c.strip() for c in df_voting.columns]

    # Replace '?' missing values — Naive Bayes needs complete data
    df_voting = df_voting.replace('?', pd.NA).dropna()

    # All values must be strings/categories for pgmpy
    df_voting = df_voting.astype(str)

    target_voting = 'Class'   # 'democrat' / 'republican'

    voting_model = NaiveBayes()
    voting_model.fit(df_voting, target_voting,
                    estimator=MaximumLikelihoodEstimator)
    
    print("Loading benchmark models and computing entropies...")

    # ── CHESS ────────────────────────────────────────────────────────────────────
    chess = fetch_ucirepo(id=22)
    df_chess = pd.concat([chess.data.features, chess.data.targets], axis=1)
    df_chess = df_chess.astype(str)

    target_chess = 'skach' 

    chess_model = NaiveBayes()
    chess_model.fit(df_chess, target_chess,
                estimator=MaximumLikelihoodEstimator)
    voting_model = BayesianNetwork(voting_model.edges())
    chess_model = BayesianNetwork(chess_model.edges())

    # fit
    voting_model.fit(df_voting, estimator=MaximumLikelihoodEstimator)
    chess_model.fit(df_chess, estimator=MaximumLikelihoodEstimator)

    alarm_model = get_example_model('alarm')
    child_model = get_example_model('child')
    #asia_model = get_example_model('asia')
    insurance_model = get_example_model('insurance')
    hailfinder_model = get_example_model('hailfinder')
    hepar_model = get_example_model('hepar2')
    barley_model = get_example_model('barley')
    win95pts_model = get_example_model('win95pts')
    andes_model = get_example_model('andes')
    link_model = get_example_model('link')
    pathfinder_model = get_example_model('pathfinder')

    # compute entropies
    print("\nComputing entropies of benchmark models...")
    df_entropies = []
    for model, name in [
        #(voting_model, "voting"),
        #(chess_model, "chess"),
        (alarm_model, "alarm"),
        (child_model, "child"),
        #(asia_model, "asia"),
        (insurance_model, "insurance"),
        (hailfinder_model, "hailfinder"),
        (hepar_model, "hepar"),
        (barley_model, "barley"),
        (win95pts_model, "win95pts"),
        (andes_model, "andes"),
        (link_model, "link"),
        (pathfinder_model, "pathfinder"),
        #(mildew_model, "Mildew"),
        #(water_model, "Water"),
    ]:
        export_node_entropies_to_csv(model, filename=f"model_{name}_node_entropies.csv")
        
        
        
        #h = entropy_bn(model)
        #print(f"  {name:12s} H(B) = {h:.4f} nats")
        #df_entropies.append({"Model": name, "Entropy (nats)": h})
    #df_entropies = pd.DataFrame(df_entropies)
    #print("\nSummary of entropies:")
    #print(df_entropies.to_string(index=False))
    #df_entropies.to_csv("bn_entropies.csv", index=False)

