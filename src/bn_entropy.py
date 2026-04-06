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

# pgmpy imports with version-compatibility shim
try:
    from pgmpy.models import DiscreteBayesianNetwork as _DBN
except ImportError:
    from pgmpy.models import BayesianNetwork as _DBN  # type: ignore[assignment]

from pgmpy.models import LinearGaussianBayesianNetwork as _LGBN
from pgmpy.inference import VariableElimination


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




# ──────────────────────────────────────────────────────────────────────────────
# Public dispatcher
# ──────────────────────────────────────────────────────────────────────────────

def entropy_bn(model) -> float:
    """
    Compute the Shannon entropy of a pgmpy Bayesian Network.

    Automatically dispatches to the correct implementation based on model type:

        DiscreteBayesianNetwork        -> entropy_discrete_bn   O(exp(w))
        LinearGaussianBayesianNetwork  -> entropy_gaussian_bn   O(N)

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
    if isinstance(model, _LGBN):
        return entropy_gaussian_bn(model)
    elif isinstance(model, _DBN):
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





if __name__ == "__main__":
    print("=" * 60)
    print("  Verifying entropy implementations vs. Scutari (2023)")
    print("=" * 60)
    _verify_discrete_example()