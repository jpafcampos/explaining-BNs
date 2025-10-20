import networkx as nx
from typing import Dict, Set, Tuple

def get_markov_blanket(bn: nx.DiGraph, variable: str) -> Set[str]:
    """
    Calculates the Markov blanket for a given variable in a Bayesian Network.

    The Markov blanket of a node V includes its parents, its children, and the
    parents of its children (spouses)[cite: 265].

    Args:
        bn: The Bayesian Network represented as a networkx.DiGraph.
        variable: The name of the node for which to find the Markov blanket.

    Returns:
        A set of node names in the Markov blanket.
    """
    blanket = set()
    # Add parents
    blanket.update(bn.predecessors(variable))
    # Add children
    children = set(bn.successors(variable))
    blanket.update(children)
    # Add parents of children (spouses)
    for child in children:
        blanket.update(bn.predecessors(child))
    # Remove the variable itself if it's in the set
    blanket.discard(variable)
    return blanket

def create_support_graph(bn: nx.DiGraph, variable_of_interest: str) -> nx.DiGraph:
    """
    Constructs a support graph from a Bayesian Network for a specific variable
    of interest, following Algorithm 1 from the paper.

    Args:
        bn: The Bayesian Network represented as a networkx.DiGraph.
        variable_of_interest: The node in the BN to be treated as the root
                              of the support graph.

    Returns:
        A networkx.DiGraph representing the support graph. Nodes are tuples of
        (variable_name, frozenset(forbidden_set)) to ensure uniqueness.
    """
    support_graph = nx.DiGraph()
    
    # We use a dictionary to keep track of the forbidden sets for each unique
    # node in the support graph. A variable can appear multiple times with
    # different forbidden sets[cite: 317].
    forbidden_sets: Dict[Tuple[str, frozenset], Set[str]] = {}

    # Initialize the graph with the root node [cite: 338]
    root_var = variable_of_interest
    root_f_set = frozenset([root_var])
    root_node = (root_var, root_f_set)
    
    support_graph.add_node(root_node, label=root_var, f_set=str(set(root_f_set)))
    forbidden_sets[root_node] = set(root_f_set)
    
    # Recursively expand the graph starting from the root [cite: 341]
    _expand(bn, support_graph, root_node, forbidden_sets)
    
    return support_graph

def _expand(bn: nx.DiGraph, sg: nx.DiGraph, node_i: tuple, f_sets: dict):
    """
    Recursive helper function to expand a node in the support graph.
    This corresponds to the 'expand' function in Algorithm 1[cite: 342].
    """
    var_i, f_set_i = node_i
    f_sets[node_i] = set(f_set_i)

    # Iterate through every variable in the Markov Blanket of V_i [cite: 345]
    for var_j in get_markov_blanket(bn, var_i):
        # The supporter variable V_j must not already be in the forbidden path.
        if var_j in f_sets[node_i]:
            continue

        # Case I: V_j is a parent of V_i [cite: 346, 362]
        if var_j in bn.predecessors(var_i):
            f_new = f_sets[node_i] | {var_j}
            _add_support(bn, sg, node_i, var_j, frozenset(f_new), f_sets)

        # Case II: V_j is a child of V_i [cite: 349, 363]
        elif var_j in bn.successors(var_i):
            f_new = f_sets[node_i] | {var_j}
            # Add other parents of V_j to F_new to prevent traversing an
            # immorality[cite: 402].
            other_parents = set(bn.predecessors(var_j)) - {var_i}
            f_new.update(other_parents)
            _add_support(bn, sg, node_i, var_j, frozenset(f_new), f_sets)
        
        # Case III: V_j is a "spouse" (co-parent of a common child) [cite: 352, 364]
        else: 
            # This case applies if V_j is a parent of a child of V_i.
            # This check finds the common child V_k.
            common_children = set(bn.successors(var_i)) & set(bn.successors(var_j))
            for var_k in common_children:
                # An immorality exists if there is no direct edge between V_i and V_j [cite: 268]
                if not bn.has_edge(var_i, var_j) and not bn.has_edge(var_j, var_i):
                    f_new = f_sets[node_i] | {var_j, var_k}
                    _add_support(bn, sg, node_i, var_j, frozenset(f_new), f_sets)


def _add_support(bn: nx.DiGraph, sg: nx.DiGraph, node_i: tuple, var_j: str, f_new: frozenset, f_sets: dict):
    """
    Adds a new supporter node and edge to the support graph and triggers expansion.
    This corresponds to the 'AddSupport' function in Algorithm 1[cite: 355].
    """
    node_j = (var_j, f_new)

    # Create the new node if it doesn't already exist [cite: 359]
    if node_j not in sg:
        sg.add_node(node_j, label=var_j, f_set=str(set(f_new)))
        # Add edge from supporter (j) to supported (i) [cite: 360]
        sg.add_edge(node_j, node_i)
        # Recursively expand the newly added node [cite: 361]
        _expand(bn, sg, node_j, f_sets)
    elif not sg.has_edge(node_j, node_i):
        # If node exists but edge doesn't, just add the edge.
        sg.add_edge(node_j, node_i)


def prune_support_graph(support_graph: nx.DiGraph, evidence_variables: set) -> nx.DiGraph:
    """
    Prunes a support graph based on a set of evidence variables.
    Definition 28: The pruned support graph is obtained by repeatedly
    removing from G every node N for which either:
    1. N is an ancestor of a node N' for which V(N') is an evidence variable, or
    2. V(N) is not an evidence variable, and N has no unpruned parents.
    """
    pruned_sg = support_graph.copy()
    
    while True:
        nodes_to_remove = set()
        for node in pruned_sg.nodes():
            var = pruned_sg.nodes[node]['label']
            
            # Rule 1: Node is an ancestor of an evidence node.
            # An evidence node has no business having supporters.
            if var in evidence_variables:
                ancestors = nx.ancestors(pruned_sg, node)
                nodes_to_remove.update(ancestors)
            
            # Rule 2: Non-evidence leaf node in the context of the pruned graph.
            # This means it's not evidence itself and all its supporters
            # have been pruned away.
            parents = set(pruned_sg.predecessors(node))
            if var not in evidence_variables and len(parents) == 0:
                # Need to check it's not the root itself unless it meets rule 1
                if len(list(pruned_sg.successors(node))) > 0:
                   nodes_to_remove.add(node)

        if not nodes_to_remove:
            break # No more nodes to prune, exit loop
        
        pruned_sg.remove_nodes_from(list(nodes_to_remove))

    return pruned_sg

if __name__ == "__main__":

    # --- Example Usage ---
    # Recreate the Bayesian Network from Figure 2 in the paper [cite: 264]
    bn_crime = nx.DiGraph()
    bn_crime.add_edges_from([
        ("Motive", "Psych_report"),
        ("Motive", "Crime"),
        ("Crime", "DNA_match"),
        ("Twin", "DNA_match")
    ])

    variable_of_interest = "Crime"

    # Construct the support graph
    support_graph = create_support_graph(bn_crime, variable_of_interest)

    # --- Verification ---
    # Print the results to verify against Figure 5 in the paper 
    print(f"Support Graph for Variable of Interest: '{variable_of_interest}'\n")

    print("Nodes (Variable, Forbidden Set):")
    for node in sorted(support_graph.nodes(data=True), key=lambda x: len(x[0][1]), reverse=True):
        print(f"- {node[1]['label']}: {node[1]['f_set']}")

    print("\nEdges (Supporter -> Supported):")
    for edge in support_graph.edges(data=True):
        supporter_label = support_graph.nodes[edge[0]]['label']
        supported_label = support_graph.nodes[edge[1]]['label']
        print(f"- {supporter_label} -> {supported_label}")

    skid_edges = [
    ('drunk_passenger', 'passenger_pulls_handbrake'),
    ('speeding_in_S_curve', 'loss_of_control_over_vehicle'),
    ('passenger_pulls_handbrake', 'locking_of_wheels'),
    ('passenger_pulls_handbrake', 'drivers_testimony'),
    ('passenger_pulls_handbrake', 'handbrake_in_pulled_position'),
    ('loss_of_control_over_vehicle', 'skidding'),
    ('locking_of_wheels', 'skidding'),
    ('skidding', 'crash'),
    ('skidding', 'tire_marks_present'),
    ('speeding_in_S_curve', 'tire_marks_after_S_curve_suggest_slowing')
    ]
    skid_graph = nx.DiGraph()
    skid_graph.add_edges_from(skid_edges)  
    support_graph2 = create_support_graph(skid_graph, 'speeding_in_S_curve')
    print(f"\nSupport Graph for Variable of Interest: 'speeding_in_S_curve'\n")
    print("Nodes (Variable, Forbidden Set):")
    for node in sorted(support_graph2.nodes(data=True), key=lambda x: len(x[0][1]), reverse=True):
        print(f"- {node[1]['label']}: {node[1]['f_set']}")
    print("\nEdges (Supporter -> Supported):")
    for edge in support_graph2.edges(data=True):
        supporter_label = support_graph2.nodes[edge[0]]['label']
        supported_label = support_graph2.nodes[edge[1]]['label']
        print(f"- {supporter_label} -> {supported_label}")