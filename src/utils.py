import pandas as pd
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
from pgmpy.inference.CausalInference import CausalInference
import networkx as nx
import bnlearn as bn
import itertools
import math
import matplotlib.pyplot as plt
import networkx as nx


def get_partitions(model, H, D, E):
    """
    Finds independent partitions of H given D and E by pruning the network structure.
    Based on Section 5.2.1 of Chen, Choi, & Darwiche (2014).
    """
    # Create a directed graph from the model
    G = nx.DiGraph(model.edges())
    # relevant = set(H) | {D} | set(E.keys())
    # ancestors = set()

    #H = prune_d_separated_variables(model, D, E)
    
    # 1. Delete edges outgoing from nodes in evidence E and hypothesis D
    nodes_to_cut = set(E.keys()) | {D}
    for node in nodes_to_cut:
        if node in G:
            edges_to_remove = list(G.out_edges(node))
            G.remove_edges_from(edges_to_remove)
            
    # 2. Successively delete all leaf nodes that are neither in H, E, or D
    keep_nodes = set(H) | set(E.keys()) | {D}
    while True:
        # A node is a leaf if out_degree is 0
        leaves = [n for n, d in G.out_degree() if d == 0 and n not in keep_nodes]
        if not leaves:
            break
        G.remove_nodes_from(leaves)
        
    # 3. Identify weakly connected components to form partitions S_i
    components = list(nx.weakly_connected_components(G))
    
    # Extract only the variables in H for each component
    partitions = []
    for comp in components:
        s_i = list(set(comp) & set(H))
        if s_i:  # Only keep non-empty partitions
            partitions.append(s_i)
            
    return partitions

def prune_d_separated_variables(model, target, evidence):
    """
    Identifies and returns only the hidden variables that actually have an 
    active trail (are NOT d-separated) to the target given the evidence.
    """
    # 1. Get the list of observed evidence variables
    observed_vars = list(evidence.keys())
    
    # 2. pgmpy finds all nodes with an active trail to the target
    # This automatically handles v-structures, blocked paths, and evidence routing
    active_trails = model.active_trail_nodes(variables=target, observed=observed_vars)
    
    # active_trails returns a dictionary: {target: {set_of_active_nodes}}
    active_nodes = active_trails[target]
    
    # 3. Filter out the evidence and the target itself to get the pruned H
    pruned_H = [node for node in active_nodes if node not in observed_vars and node != target]
    
    # For reporting, calculate how many we pruned
    all_hidden = [node for node in model.nodes() if node not in observed_vars and node != target]
    pruned_count = len(all_hidden) - len(pruned_H)
    
    #if pruned_count > 0:
    #    print(f"Pruned {pruned_count} d-separated variables. New H size: {len(pruned_H)}")
        
    return pruned_H

from pgmpy.sampling import BayesianModelSampling
from pgmpy.factors.discrete import State

def sample_posterior_likelihood(model, evidence, num_samples=5000):
    """
    Draws samples from P(H|e) using Likelihood Weighting.
    Initializes instantly and guarantees 100% sample acceptance.
    """
    sampler = BayesianModelSampling(model)
    
    # Format evidence for pgmpy
    evidence_states = [State(var, state) for var, state in evidence.items()]
    
    # Generate weighted samples
    # Returns a DataFrame with your variables + a special '_weight' column
    weighted_samples_df = sampler.likelihood_weighted_sample(evidence=evidence_states, 
                                                    size=num_samples, 
                                                    show_progress=True)
    
    # Drop the evidence columns (we only need the hidden variables + the weights)
    cols_to_keep = [col for col in weighted_samples_df.columns if col not in evidence.keys()]
    
    return weighted_samples_df[cols_to_keep].reset_index(drop=True)

def plot_experiment_results(results_df, model_name):
    plt.figure(figsize=(10, 6))
    
    # Scatter plot of the estimation errors
    plt.scatter(results_df['True_SDP'], results_df['Error'], 
                alpha=0.6, color='blue', edgecolor='k', s=50, label='Trial Error')
    
    # Zero-error baseline
    plt.axhline(0, color='red', linestyle='--', linewidth=2, label='Zero Error (Exact Truth)')
    
    plt.title('Monte Carlo SDP Estimation Error vs. True SDP,'+ model_name + ' Network', fontsize=14)
    plt.xlabel('Exact Same-Decision Probability', fontsize=12)
    plt.ylabel('Estimation Error (MC Estimate - Exact Truth)', fontsize=12)
    
    # Formatting
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    plt.show()

def plot_experiment_results_exact_vs_est(results_df, model_name):
    plt.figure(figsize=(16, 8))  # A square figure is best for y=x agreement plots
    
    # Scatter plot of True vs Estimated SDP
    plt.scatter(results_df['True_SDP'], results_df['Estimated_SDP'], 
                alpha=0.6, color='blue', edgecolor='k', s=50, label='MC Trial Estimate')
    
    # Zero-error baseline (Diagonal y=x line)
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', linewidth=2, label='Error Baseline')

    # Mark the exact SDP values on x-axis labels
    plt.xticks(results_df['True_SDP'].unique())

    # Titles and Labels
    plt.title(f'Estimated vs. Exact Same-Decision Probability\n{model_name} Network', fontsize=14)
    plt.xlabel('Exact Same-Decision Probability', fontsize=12)
    plt.ylabel('Estimated Same-Decision Probability (Monte Carlo)', fontsize=12)
    
    # Lock the axes to [0, 1] probability space with a tiny padding for edge points
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    
    # Formatting
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    plt.show()


