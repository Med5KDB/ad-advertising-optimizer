import pulp
import pandas as pd
import numpy as np
from model_training import predict_revenue, model, feature_names

def optimize_ad_budget(total_budget, channels, min_budget_per_channel=0, max_budget_per_channel=None):
    """
    Optimize advertising budget allocation across channels using linear programming.
    
    Parameters:
    - total_budget: float, total available budget
    - channels: list of str, available advertising channels
    - min_budget_per_channel: float, minimum budget per channel
    - max_budget_per_channel: float, maximum budget per channel (if None, uses total_budget)
    """
    
    # Create optimization problem (maximization)
    prob = pulp.LpProblem("Ad_Budget_Optimization", pulp.LpMaximize)
    
    # Define decision variables (budget for each channel)
    budget_vars = pulp.LpVariable.dicts("Budget",
                                      channels,
                                      lowBound=min_budget_per_channel,
                                      upBound=max_budget_per_channel or total_budget)
    

    def create_test_data(budget_values):
        test_data = pd.DataFrame({
            'Budget': [budget_values[ch] for ch in channels],
            'Impressions': [budget_values[ch] * 50 for ch in channels], 
            'Clics': [budget_values[ch] * 0.5 for ch in channels],     
            'Conversions': [budget_values[ch] * 0.05 for ch in channels],
            'CPC': [2.0] * len(channels),
            'CPA': [20.0] * len(channels),
            'Facteur_Saisonnier': [1.0] * len(channels),
            'Budget_Total_Semaine': [total_budget] * len(channels),
            'Canal': channels,
            'Mois': ['Janvier'] * len(channels),
            'Jour_Semaine': ['Lundi'] * len(channels)
        })
        return test_data

    def objective_function(budget_values):
        test_data = create_test_data(budget_values)
        predictions = predict_revenue(model, test_data, feature_names)
        return sum(predictions)
    
    prob += pulp.lpSum([budget_vars[ch] for ch in channels])
    
   # Constraints
    prob += pulp.lpSum([budget_vars[ch] for ch in channels]) <= total_budget
    
    for ch in channels:
        prob += budget_vars[ch] >= min_budget_per_channel
        
    prob.solve()
    optimal_allocation = {ch: budget_vars[ch].value() for ch in channels}
    test_data = create_test_data(optimal_allocation)
    expected_revenue = predict_revenue(model, test_data, feature_names)
    
    return {
        'status': pulp.LpStatus[prob.status],
        'optimal_allocation': optimal_allocation,
        'expected_revenue': expected_revenue,
        'objective_value': pulp.value(prob.objective)
    }

if __name__ == "__main__":
    channels = ['Facebook', 'Google']
    total_budget = 10000
    min_budget = 1000
    max_budget = 8000
    
    result = optimize_ad_budget(
        total_budget=total_budget,
        channels=channels,
        min_budget_per_channel=min_budget,
        max_budget_per_channel=max_budget
    )
    
    print("\nOptimization Results:")
    print(f"Status: {result['status']}")
    print("\nOptimal Budget Allocation:")
    for channel, budget in result['optimal_allocation'].items():
        print(f"{channel}: €{budget:,.2f}")
    print(f"\nExpected Revenue: €{result['expected_revenue'].sum():,.2f}")