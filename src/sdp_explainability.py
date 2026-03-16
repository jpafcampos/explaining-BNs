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
from utils import *


# Rank the variables inside e by their individual impact, comparing P(D|e) vs P(D|e - {var})
def rank_evidence_variables(model, D, d_value, evidence, metric='delta'):
    '''
    Ranks evidence variables by their impact on the decision.
    metric: 'delta' for absolute change in probability;
            'KL' for KL divergence between distributions with and without the variable.
    '''
    inference = VariableElimination(model)
    
    # Get opposite state for D
    d_states = model.get_cpds(D).state_names[D]
    d_index = d_states.index(d_value)
    not_d_value = d_states[1] if d_index == 0 else d_states[0]
    
    # Compute baseline probability with all evidence
    dist_full = inference.query(variables=[D], evidence=evidence, show_progress=False)
    p_d_full = dist_full.values[d_index]
    
    variable_impact = []
    
    for var in evidence.keys():
        reduced_evidence = {k: v for k, v in evidence.items() if k != var}
        dist_reduced = inference.query(variables=[D], evidence=reduced_evidence, show_progress=False)
        p_d_reduced = dist_reduced.values[d_index]
        
        
        if metric == 'delta':
            impact = abs(p_d_full - p_d_reduced)
        elif metric == 'KL':
            p_not_d_full = dist_full.values[1 - d_index]
            p_not_d_reduced = dist_reduced.values[1 - d_index]
            # KL divergence from full to reduced
            impact = (p_d_full * math.log(p_d_full / p_d_reduced)) + (p_not_d_full * math.log(p_not_d_full / p_not_d_reduced))
        else:
            raise ValueError("Unsupported metric. Use 'delta' or 'KL'.")


        variable_impact.append((var, impact))
        
    variable_impact.sort(key=lambda x: x[1], reverse=True)
    
    return variable_impact



def get_decision_flipping_scenarios(model, D, d_value, evidence, threshold):
    """
    Identifies the exact hidden variable scenarios that would flip the current decision.
    
    Returns a list of dictionaries containing:
      - 'scenario': The specific hidden variable instantiation.
      - 'prob_scenario': How likely this scenario is to happen, Pr(h|e).
      - 'new_decision_prob': The new probability of the diagnosis, Pr(D|h,e).
    """
    inference = VariableElimination(model)
    
    # 1. Identify hidden variables
    all_vars = set(model.nodes())
    observed_vars = set(evidence.keys())
    H = list(all_vars - observed_vars - {D})
    
    # 2. Determine the CURRENT decision
    current_dist = inference.query(variables=[D], evidence=evidence, show_progress=False)
    d_index = model.get_cpds(D).state_names[D].index(d_value)
    p_d_initial = current_dist.values[d_index]
    
    current_decision = p_d_initial >= threshold
    
    # 3. Setup state spaces for iteration
    state_spaces = [model.get_cpds(var).state_names[var] for var in H]
    all_assignments = list(itertools.product(*state_spaces))
    
    # Pre-compute Pr(h|e)
    p_h_dist = inference.query(variables=H, evidence=evidence, show_progress=False)
    
    flipped_scenarios = []
    
    # 4. Evaluate each scenario
    for assignment in all_assignments:
        h_dict = dict(zip(H, assignment))
        
        # Pr(h|e)
        p_h_given_e = p_h_dist.get_value(**h_dict)
        if p_h_given_e == 0:
            continue
            
        # Pr(D | e, h)
        query_e_h = {**evidence, **h_dict}
        p_d_given_e_h_dist = inference.query(variables=[D], evidence=query_e_h, show_progress=False)
        p_d_given_e_h = p_d_given_e_h_dist.values[d_index]
        
        # 5. Check if the decision FLIPS
        new_decision = p_d_given_e_h >= threshold
        
        if new_decision != current_decision:
            flipped_scenarios.append({
                'scenario': h_dict,
                'prob_scenario': p_h_given_e,
                'new_decision_prob': p_d_given_e_h
            })
            
    # Sort the flipped scenarios by how likely they are to occur (highest probability first)
    flipped_scenarios.sort(key=lambda x: x['prob_scenario'], reverse=True)
    
    return {
        'current_prob': p_d_initial,
        'current_decision': current_decision,
        'threshold': threshold,
        'flipping_scenarios': flipped_scenarios
    }