"""
Bayesian Network Generator implementing the PMMixed algorithm.

Based on:
    Ide, Cozman, Ramos — "Generating Random Bayesian Networks with
    Constraints on Induced Width" (2004)

Uses NetworkX for all graph operations.
"""

import random
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np


class BayesianNetworkGenerator:
    """
    Generates uniformly distributed random Bayesian networks with constraints
    on induced width, node degree, and number of edges using ergodic Markov
    chains (Algorithm PMMixed + Procedure J from Ide et al. 2004).

    Parameters
    ----------
    n_nodes : int
        Number of variables in the network.
    max_induced_width : int
        Maximum allowed heuristic induced width (minimum-weight heuristic).
    max_degree : int, optional
        Maximum total (in + out) degree per node. Defaults to n_nodes - 1.
    max_edges : int, optional
        Maximum number of directed edges. Defaults to n*(n-1)//2.
    p : float
        Probability of calling AorR from a polytree state.
    q : float
        Probability of calling AR from a polytree state.
        Must satisfy p + q < 1.
    n_iterations : int
        Number of Markov chain steps before returning the DAG.
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_nodes: int,
        max_induced_width: int,
        max_degree: Optional[int] = None,
        max_edges: Optional[int] = None,
        p: float = 0.5,
        q: float = 0.25,
        n_iterations: int = 10_000,
        seed: Optional[int] = None,
    ) -> None:
        if n_nodes < 2:
            raise ValueError("n_nodes must be at least 2.")
        if not (0 < p and 0 < q and p + q < 1):
            raise ValueError("p and q must be positive and p + q < 1.")

        self.n = n_nodes
        self.max_width = max_induced_width
        self.max_degree = max_degree if max_degree is not None else n_nodes - 1
        self.max_edges = max_edges if max_edges is not None else n_nodes * (n_nodes - 1) // 2
        self.p = p
        self.q = q
        self.n_iterations = n_iterations

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.dag: nx.DiGraph = nx.DiGraph()
        self._initialize_graph()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _initialize_graph(self) -> None:
        """Start from a simple chain 0->1->2->...->n-1 (polytree, width 1)."""
        self.dag = nx.DiGraph()
        self.dag.add_nodes_from(range(self.n))
        for i in range(self.n - 1):
            self.dag.add_edge(i, i + 1)

    # ------------------------------------------------------------------
    # Graph structural helpers (all using NetworkX)
    # ------------------------------------------------------------------

    def _is_polytree(self, g: Optional[nx.DiGraph] = None) -> bool:
        """A connected DAG is a polytree iff it has exactly n-1 edges."""
        if g is None:
            g = self.dag
        return g.number_of_edges() == self.n - 1

    def _is_connected(self, g: Optional[nx.DiGraph] = None) -> bool:
        if g is None:
            g = self.dag
        return nx.is_weakly_connected(g)

    def _is_acyclic(self, g: Optional[nx.DiGraph] = None) -> bool:
        if g is None:
            g = self.dag
        return nx.is_directed_acyclic_graph(g)

    def _max_degree(self, g: Optional[nx.DiGraph] = None) -> int:
        if g is None:
            g = self.dag
        undirected = g.to_undirected()
        return max(dict(undirected.degree()).values(), default=0)

    # ------------------------------------------------------------------
    # Heuristic induced width (minimum-weight greedy elimination)
    # ------------------------------------------------------------------

    def _moral_graph(self, g: Optional[nx.DiGraph] = None) -> nx.Graph:
        """
        Build the moral graph: undirected version of g, plus edges connecting
        every pair of parents that share a common child.
        """
        if g is None:
            g = self.dag
        moral = g.to_undirected()
        for node in g.nodes():
            parents = list(g.predecessors(node))
            for i in range(len(parents)):
                for j in range(i + 1, len(parents)):
                    moral.add_edge(parents[i], parents[j])
        return moral

    def _heuristic_width(self, g: Optional[nx.DiGraph] = None) -> int:
        """
        Minimum-weight heuristic for induced width on the moral graph.

        Nodes are eliminated in order of increasing current degree; fill edges
        are added between remaining neighbours before each elimination.
        Returns the maximum degree observed at time of elimination.
        """
        moral = self._moral_graph(g)
        working = moral.copy()
        remaining = set(working.nodes())
        max_w = 0

        while remaining:
            node = min(remaining, key=lambda v: len(set(working.neighbors(v)) & remaining))
            neighbours = list(set(working.neighbors(node)) & remaining)
            max_w = max(max_w, len(neighbours))

            # Add fill edges between all neighbours
            for i in range(len(neighbours)):
                for j in range(i + 1, len(neighbours)):
                    if not working.has_edge(neighbours[i], neighbours[j]):
                        working.add_edge(neighbours[i], neighbours[j])

            remaining.remove(node)

        return max_w

    # ------------------------------------------------------------------
    # Constraint checking
    # ------------------------------------------------------------------

    def _satisfies_constraints(self, g: nx.DiGraph) -> bool:
        if g.number_of_edges() > self.max_edges:
            return False
        if self._max_degree(g) > self.max_degree:
            return False
        if self._heuristic_width(g) > self.max_width:
            return False
        return True

    # ------------------------------------------------------------------
    # Transition procedures
    # ------------------------------------------------------------------

    def _procedure_AorR(self) -> nx.DiGraph:
        """
        Procedure AorR (Add or Remove).

        Pick a random pair (i, j).
        - Arc (i,j) exists -> remove it if the graph stays connected.
        - Arc (i,j) absent -> add it if the graph stays acyclic.
        """
        i, j = random.sample(range(self.n), 2)
        candidate = self.dag.copy()

        if candidate.has_edge(i, j):
            candidate.remove_edge(i, j)
            if self._is_connected(candidate):
                return candidate
        else:
            candidate.add_edge(i, j)
            if self._is_acyclic(candidate):
                return candidate

        return self.dag

    def _procedure_AR(self) -> nx.DiGraph:
        """
        Procedure AR (Add and Remove) — designed for polytrees.

        Pick a random pair (i, j) without an existing arc (i, j).
        With prob 1/2 swap to (j, i).  Find the predecessor k of j on the
        unique undirected path from i to j, remove arc k-j, add arc i-j.
        """
        i, j = random.sample(range(self.n), 2)
        if self.dag.has_edge(i, j):
            return self.dag

        if random.random() < 0.5:
            i, j = j, i

        undirected = self.dag.to_undirected()
        try:
            path = nx.shortest_path(undirected, i, j)
        except nx.NetworkXNoPath:
            return self.dag

        if len(path) < 2:
            return self.dag

        k = path[-2]  # predecessor of j on the path
        candidate = self.dag.copy()

        if candidate.has_edge(k, j):
            candidate.remove_edge(k, j)
        elif candidate.has_edge(j, k):
            candidate.remove_edge(j, k)
        else:
            return self.dag

        candidate.add_edge(i, j)
        if self._is_acyclic(candidate):
            return candidate

        # Try reverse direction as fallback
        candidate.remove_edge(i, j)
        candidate.add_edge(j, i)
        if self._is_acyclic(candidate):
            return candidate

        return self.dag

    def _procedure_J(self) -> nx.DiGraph:
        """
        Procedure J (Jump).

        Enables ergodicity regardless of the heuristic used:
        - Polytree        -> try to add a random arc.
        - Multi-connected -> try to remove a random arc (preserving connectivity).
        """
        i, j = random.sample(range(self.n), 2)
        candidate = self.dag.copy()

        if self._is_polytree(candidate):
            if not candidate.has_edge(i, j):
                candidate.add_edge(i, j)
                if self._is_acyclic(candidate):
                    return candidate
        else:
            if candidate.has_edge(i, j):
                candidate.remove_edge(i, j)
                if self._is_connected(candidate):
                    return candidate

        return self.dag

    # ------------------------------------------------------------------
    # PMMixed main loop (Algorithm PMMixed + Procedure J, Figure 7)
    # ------------------------------------------------------------------

    def _step(self) -> None:
        """Execute one Markov-chain transition of PMMixed."""
        r = random.random()

        if self._is_polytree():
            if r < self.p:
                candidate = self._procedure_AorR()
            elif r < self.p + self.q:
                candidate = self._procedure_AR()
            else:
                candidate = self._procedure_J()

            if self._satisfies_constraints(candidate):
                self.dag = candidate

        else:
            if r < self.p + self.q:
                candidate = self._procedure_AorR()
                if self._satisfies_constraints(candidate):
                    if self._is_polytree(candidate):
                        # Dampen acceptance to maintain detailed balance
                        if random.random() < self.p / (self.p + self.q):
                            self.dag = candidate
                    else:
                        self.dag = candidate
            else:
                candidate = self._procedure_J()
                if self._satisfies_constraints(candidate):
                    self.dag = candidate

    def generate(self) -> "BayesianNetworkGenerator":
        """
        Run PMMixed for ``n_iterations`` steps.

        Returns self for method chaining.
        """
        for _ in range(self.n_iterations):
            self._step()
        return self

    # ------------------------------------------------------------------
    # CPD generation
    # ------------------------------------------------------------------

    def generate_cpds(
        self,
        n_states: int = 2,
        dirichlet_alpha: float = 1.0,
    ) -> Dict[int, np.ndarray]:
        """
        Attach conditional probability distributions to the generated DAG.

        Samples one Dirichlet row per parent configuration.

        Parameters
        ----------
        n_states : int
            Number of discrete states per variable.
        dirichlet_alpha : float
            Concentration parameter for the Dirichlet prior (1.0 = uniform).

        Returns
        -------
        dict
            {node: ndarray of shape (n_parent_configs, n_states)}
        """
        cpds: Dict[int, np.ndarray] = {}
        alpha = np.full(n_states, dirichlet_alpha)

        for node in self.dag.nodes():
            n_parents = self.dag.in_degree(node)
            n_configs = n_states ** n_parents
            cpds[node] = np.vstack(
                [np.random.dirichlet(alpha) for _ in range(n_configs)]
            )
        return cpds

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def get_edges(self) -> List[Tuple[int, int]]:
        """Return current directed edges as a sorted list of (u, v) pairs."""
        return sorted(self.dag.edges())

    def get_parents(self) -> Dict[int, List[int]]:
        """Return {node: [parent, ...]} for all nodes."""
        return {node: sorted(self.dag.predecessors(node)) for node in self.dag.nodes()}

    def get_adjacency_matrix(self) -> np.ndarray:
        """Return the n x n binary adjacency matrix (A[i,j]=1 means i->j)."""
        return nx.to_numpy_array(self.dag, nodelist=sorted(self.dag.nodes()), dtype=int)

    def get_heuristic_width(self) -> int:
        return self._heuristic_width()

    def get_n_edges(self) -> int:
        return self.dag.number_of_edges()

    def is_polytree(self) -> bool:
        return self._is_polytree()

    def summary(self) -> Dict:
        return {
            "n_nodes": self.n,
            "n_edges": self.get_n_edges(),
            "heuristic_induced_width": self.get_heuristic_width(),
            "is_polytree": self.is_polytree(),
            "max_degree": self._max_degree(),
            "is_weakly_connected": self._is_connected(),
            "is_acyclic": self._is_acyclic(),
        }

    def __repr__(self) -> str:
        s = self.summary()
        return (
            f"BayesianNetworkGenerator("
            f"n={s['n_nodes']}, edges={s['n_edges']}, "
            f"heuristic_width={s['heuristic_induced_width']}, "
            f"polytree={s['is_polytree']})"
        )


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== BayesianNetworkGenerator (NetworkX) demo ===\n")

    gen = BayesianNetworkGenerator(
        n_nodes=10,
        max_induced_width=3,
        max_degree=5,
        n_iterations=5_000,
        seed=42,
    )

    gen.generate()
    print("Generated network:", gen)
    print("\nSummary:", gen.summary())
    print("\nEdges:", gen.get_edges())
    print("\nParents:")
    for node, pars in gen.get_parents().items():
        print(f"  X{node} | parents={pars}")

    print("\nAdjacency matrix:")
    print(gen.get_adjacency_matrix())

    cpds = gen.generate_cpds(n_states=2)
    print(f"\nCPD for node 0 (shape {cpds[0].shape}):\n", cpds[0])