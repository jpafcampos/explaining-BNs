"""
Network profiler for Bayesian Networks.
Computes structural metrics relevant to exact inference tractability.
"""
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from math import prod
import warnings
warnings.filterwarnings("ignore")


def profile_network(bn, name=None):
    """
    Computes a comprehensive structural profile of a Bayesian Network.

    Returns a dict of metrics relevant to:
      - General graph structure
      - Cardinality / state space
      - Inference tractability (treewidth, moral graph density)
      - SDP-specific metrics (partition structure under random evidence)
    """
    name = name or getattr(bn, 'name', 'unknown')
    G    = nx.DiGraph(bn.edges())
    G.add_nodes_from(bn.nodes())
    n    = G.number_of_nodes()
    e    = G.number_of_edges()

    # ── Basic graph metrics ───────────────────────────────────────────────
    in_degrees  = [d for _, d in G.in_degree()]
    out_degrees = [d for _, d in G.out_degree()]
    degrees     = [d for _, d in G.degree()]

    roots   = [nd for nd in G.nodes() if G.in_degree(nd) == 0]
    leaves  = [nd for nd in G.nodes() if G.out_degree(nd) == 0]

    # ── Cardinality metrics ───────────────────────────────────────────────
    cardinalities = [len(bn.get_cpds(nd).state_names[nd]) for nd in bn.nodes()]
    binary_nodes  = sum(1 for c in cardinalities if c == 2)
    total_states  = sum(cardinalities)

    # CPD size = product of cardinalities of node + parents
    cpd_sizes = []
    for nd in bn.nodes():
        cpd    = bn.get_cpds(nd)
        c_size = prod(
            len(bn.get_cpds(v).state_names[v]) for v in cpd.variables
        )
        cpd_sizes.append(c_size)

    # ── Moral graph (for treewidth approximation) ─────────────────────────
    moral = nx.moral_graph(G.to_undirected())
    moral_edges   = moral.number_of_edges()
    moral_density = nx.density(moral)

    # Min-degree elimination order (greedy treewidth upper bound)
    def greedy_treewidth(graph):
        H       = graph.copy()
        max_clique = 0
        order   = []
        for _ in range(len(H.nodes())):
            if not H.nodes():
                break
            # Pick node with minimum degree
            node = min(H.nodes(), key=lambda v: H.degree(v))
            neighbors = list(H.neighbors(node))
            clique_size = len(neighbors)
            max_clique  = max(max_clique, clique_size)
            # Connect all neighbors (fill-in)
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    if not H.has_edge(neighbors[i], neighbors[j]):
                        H.add_edge(neighbors[i], neighbors[j])
            H.remove_node(node)
            order.append(node)
        return max_clique, order

    treewidth_ub, elim_order = greedy_treewidth(moral.copy())

    # ── Longest path (network "depth") ────────────────────────────────────
    try:
        longest_path_len = nx.dag_longest_path_length(G)
        longest_path     = nx.dag_longest_path(G)
    except Exception:
        longest_path_len = -1
        longest_path     = []

    # ── Connected components (undirected) ─────────────────────────────────
    undirected     = G.to_undirected()
    n_components   = nx.number_connected_components(undirected)
    component_sizes = sorted(
        [len(c) for c in nx.connected_components(undirected)], reverse=True
    )

    # ── SDP-relevant: partition structure under 3 random evidence sets ────
    import random
    from pgmpy.inference import VariableElimination

    def get_partitions_local(hidden_vars, patient):
        """Simplified partition computation for profiling."""
        Gp = nx.DiGraph(bn.edges())
        nodes_to_cut = set(patient.keys())
        for node in nodes_to_cut:
            if node in Gp:
                Gp.remove_edges_from(list(Gp.out_edges(node)))
        keep = set(hidden_vars) | set(patient.keys())
        while True:
            leaves_to_rm = [nd for nd in Gp.nodes()
                            if Gp.out_degree(nd) == 0 and nd not in keep]
            if not leaves_to_rm:
                break
            Gp.remove_nodes_from(leaves_to_rm)
        components = list(nx.weakly_connected_components(Gp))
        partitions = []
        for comp in components:
            s_i = [v for v in comp if v in set(hidden_vars)]
            if s_i:
                partitions.append(s_i)
        return partitions

    partition_profiles = {}
    all_nodes_list = list(bn.nodes())

    for h_ratio in [0.1, 0.25, 0.5]:
        n_hidden   = max(1, int((n - 1) * h_ratio))
        n_evidence = (n - 1) - n_hidden
        max_parts  = []
        n_parts    = []
        max_tensors = []

        for trial in range(5):
            try:
                target_nd  = random.choice(all_nodes_list)
                avail      = [nd for nd in all_nodes_list if nd != target_nd]
                ev_vars    = random.sample(avail, min(n_evidence, len(avail)))
                h_vars     = [nd for nd in all_nodes_list
                              if nd not in ev_vars and nd != target_nd]
                patient    = {v: random.choice(bn.get_cpds(v).state_names[v])
                              for v in ev_vars}
                parts      = get_partitions_local(h_vars, patient)
                if parts:
                    max_parts.append(max(len(p) for p in parts))
                    n_parts.append(len(parts))
                    max_tensors.append(max(
                        prod(len(bn.get_cpds(v).state_names[v]) for v in p)
                        for p in parts
                    ))
            except Exception:
                pass

        partition_profiles[h_ratio] = {
            'avg_max_partition':  np.mean(max_parts)   if max_parts   else None,
            'avg_n_partitions':   np.mean(n_parts)     if n_parts     else None,
            'avg_max_tensor':     np.mean(max_tensors) if max_tensors else None,
        }

    # ── Assemble result ───────────────────────────────────────────────────
    return {
        'name':               name,
        # Basic
        'n_nodes':            n,
        'n_edges':            e,
        'density':            e / (n * (n - 1)) if n > 1 else 0,
        'n_roots':            len(roots),
        'n_leaves':           len(leaves),
        'n_connected_components': n_components,
        'largest_component':  component_sizes[0] if component_sizes else 0,
        # Degree
        'avg_degree':         np.mean(degrees),
        'max_degree':         max(degrees),
        'avg_in_degree':      np.mean(in_degrees),
        'max_in_degree':      max(in_degrees),
        'avg_out_degree':     np.mean(out_degrees),
        'max_out_degree':     max(out_degrees),
        # Depth / width
        'longest_path':       longest_path_len,
        'n_levels':           longest_path_len + 1,
        # Cardinality
        'avg_cardinality':    np.mean(cardinalities),
        'max_cardinality':    max(cardinalities),
        'pct_binary':         binary_nodes / n * 100,
        'total_states':       total_states,
        'total_cpd_entries':  sum(cpd_sizes),
        'max_cpd_entries':    max(cpd_sizes),
        # Treewidth (inference hardness)
        'treewidth_ub':       treewidth_ub,
        'moral_density':      moral_density,
        'moral_edges':        moral_edges,
        # Partition structure
        'partition_h10':      partition_profiles[0.10],
        'partition_h25':      partition_profiles[0.25],
        'partition_h50':      partition_profiles[0.50],
    }


def profile_multiple_networks(networks_dict):
    """
    Profiles multiple networks and returns a summary DataFrame.
    networks_dict: {name: bn_model}
    """
    profiles = []
    for name, bn in networks_dict.items():
        print(f"Profiling {name}...")
        p = profile_network(bn, name=name)
        profiles.append(p)

    # Flatten partition profiles for DataFrame
    rows = []
    for p in profiles:
        row = {k: v for k, v in p.items()
               if not isinstance(v, dict)}
        for h_ratio, pp in [('h10', p['partition_h10']),
                             ('h25', p['partition_h25']),
                             ('h50', p['partition_h50'])]:
            if pp:
                row[f'avg_max_partition_{h_ratio}'] = pp['avg_max_partition']
                row[f'avg_n_partitions_{h_ratio}']  = pp['avg_n_partitions']
                row[f'avg_max_tensor_{h_ratio}']    = pp['avg_max_tensor']
        rows.append(row)

    return pd.DataFrame(rows).set_index('name')


def print_profile(p):
    """Pretty-print a single network profile."""
    print(f"\n{'='*60}")
    print(f"Network: {p['name']}")
    print(f"{'='*60}")

    print(f"\n── Basic structure")
    print(f"  Nodes:              {p['n_nodes']}")
    print(f"  Edges:              {p['n_edges']}")
    print(f"  Density:            {p['density']:.6f}")
    print(f"  Roots:              {p['n_roots']}")
    print(f"  Leaves:             {p['n_leaves']}")
    print(f"  Connected comps:    {p['n_connected_components']}")

    print(f"\n── Depth / width")
    print(f"  Longest path:       {p['longest_path']}")
    print(f"  Levels:             {p['n_levels']}")
    print(f"  Avg degree:         {p['avg_degree']:.2f}")
    print(f"  Max degree:         {p['max_degree']}")
    print(f"  Max in-degree:      {p['max_in_degree']}")
    print(f"  Max out-degree:     {p['max_out_degree']}")

    print(f"\n── Cardinality")
    print(f"  Avg cardinality:    {p['avg_cardinality']:.2f}")
    print(f"  Max cardinality:    {p['max_cardinality']}")
    print(f"  % binary nodes:     {p['pct_binary']:.1f}%")
    print(f"  Total CPD entries:  {p['total_cpd_entries']:,}")
    print(f"  Max CPD entries:    {p['max_cpd_entries']:,}")

    print(f"\n── Inference hardness")
    print(f"  Treewidth (UB):     {p['treewidth_ub']}")
    print(f"  Moral graph density:{p['moral_density']:.6f}")
    print(f"  Moral graph edges:  {p['moral_edges']}")

    print(f"\n── SDP partition structure (5 random trials per ratio)")
    for label, key in [('H=10%', 'partition_h10'),
                       ('H=25%', 'partition_h25'),
                       ('H=50%', 'partition_h50')]:
        pp = p[key]
        if pp and pp['avg_max_partition'] is not None:
            print(f"  {label}: avg_max_partition={pp['avg_max_partition']:.1f} | "
                  f"avg_n_partitions={pp['avg_n_partitions']:.1f} | "
                  f"avg_max_tensor={pp['avg_max_tensor']:.0f}")
        else:
            print(f"  {label}: could not compute")


def plot_profiles(df):
    """
    Visual comparison of network profiles.
    df: DataFrame from profile_multiple_networks()
    """
    BLUE, RED, AMBER, GREEN, PURP, GRAY = (
        '#185FA5', '#A32D2D', '#B85C00', '#3B6D11', '#534AB7', '#888780'
    )

    fig = plt.figure(figsize=(16, 20))
    fig.patch.set_facecolor('white')
    gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.35)

    nets = df.index.tolist()
    x    = np.arange(len(nets))
    w    = 0.6

    def bar(ax, values, title, ylabel, color=BLUE, log=False):
        bars = ax.bar(x, values, width=w, color=color)
        ax.set_xticks(x)
        ax.set_xticklabels(nets, rotation=30, ha='right', fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=10)
        if log:
            ax.set_yscale('log')
        ax.grid(True, alpha=0.3, axis='y')
        for bar_, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(bar_.get_x() + bar_.get_width()/2, bar_.get_height(),
                        f'{val:.0f}', ha='center', va='bottom', fontsize=8)

    # 1. Nodes and edges
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(x - 0.2, df['n_nodes'], width=0.35, color=BLUE, label='nodes')
    ax1.bar(x + 0.2, df['n_edges'], width=0.35, color=AMBER, label='edges')
    ax1.set_xticks(x)
    ax1.set_xticklabels(nets, rotation=30, ha='right', fontsize=9)
    ax1.set_title('Nodes and edges')
    ax1.set_yscale('log')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')

    # 2. Treewidth upper bound
    ax2 = fig.add_subplot(gs[0, 1])
    bar(ax2, df['treewidth_ub'].values.astype(float),
        'Treewidth upper bound\n(inference hardness)', 'treewidth', color=RED)

    # 3. Max in-degree
    ax3 = fig.add_subplot(gs[1, 0])
    bar(ax3, df['max_in_degree'].values.astype(float),
        'Max in-degree\n(CPD complexity)', 'max in-degree', color=AMBER)

    # 4. Max CPD entries (log)
    ax4 = fig.add_subplot(gs[1, 1])
    bar(ax4, df['max_cpd_entries'].values.astype(float),
        'Max CPD entries (log)\n(largest conditional table)', 'entries (log)',
        color=PURP, log=True)

    # 5. % binary nodes
    ax5 = fig.add_subplot(gs[2, 0])
    bar(ax5, df['pct_binary'].values.astype(float),
        '% binary nodes', '% binary', color=GREEN)
    ax5.set_ylim(0, 110)

    # 6. Longest path
    ax6 = fig.add_subplot(gs[2, 1])
    bar(ax6, df['longest_path'].values.astype(float),
        'Longest path\n(network depth)', 'depth', color=TEAL
        if 'TEAL' in dir() else '#0F6E56')

    # 7. Avg max partition size at different H ratios
    ax7 = fig.add_subplot(gs[3, 0])
    for i, (label, col, color) in enumerate([
        ('H=10%', 'avg_max_partition_h10', BLUE),
        ('H=25%', 'avg_max_partition_h25', AMBER),
        ('H=50%', 'avg_max_partition_h50', RED),
    ]):
        if col in df.columns:
            vals = df[col].fillna(0).values.astype(float)
            ax7.bar(x + (i - 1) * 0.25, vals, width=0.22,
                    color=color, label=label, alpha=0.85)
    ax7.set_xticks(x)
    ax7.set_xticklabels(nets, rotation=30, ha='right', fontsize=9)
    ax7.set_title('Avg max partition size by H ratio\n(SDP tractability indicator)')
    ax7.set_ylabel('avg max partition size')
    ax7.legend(fontsize=9)
    ax7.grid(True, alpha=0.3, axis='y')

    # 8. Avg max tensor size at H=25% (log)
    ax8 = fig.add_subplot(gs[3, 1])
    if 'avg_max_tensor_h25' in df.columns:
        vals = df['avg_max_tensor_h25'].fillna(1).values.astype(float)
        bar(ax8, vals,
            'Avg max tensor size at H=25% (log)\n(memory cost indicator)',
            'tensor entries (log)', color=PURP, log=True)

    fig.suptitle('Bayesian Network structural profiles',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.savefig('network_profiles.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    print("\nPlot saved to network_profiles.png")
    plt.show()