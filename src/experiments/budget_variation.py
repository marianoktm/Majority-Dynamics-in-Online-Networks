"""
Budget variation experiment.

Measures how the influence of the selected seed set grows as the available budget
increases, running the seed-selection function on a range of budgets and simulating
the cascade for each resulting seed set.
"""

import math

from src import simulate_cascade


def budget_variation_experiment(graph, costs, budget, seed_set_function):
    """
    Function: budget_variation_experiment

    Runs the seed-selection function with increasing budgets (about 50 steps from 0 to the
    total cost) and measures the size of the cascade each seed set produces. Stops early
    if every node is influenced.

    Args:
        graph (snap graph): Undirected graph.
        costs (dict): Maps each node ID (int) to its cost (int).
        budget (int): Unused; budgets are varied internally. Kept for a uniform experiment interface.
        seed_set_function (callable): Function (graph, costs, budget) returning a seed set.

    Returns:
        dict: Maps each tested budget k (int) to the number of influenced nodes (int).
    """
    max_cost = sum(costs.values())
    # print("Max Cost: ", max_cost)
    results = {}
    for k in range(0, max_cost, math.ceil(max_cost / 50)):
        # print("k:", k)
        seed_set = seed_set_function(graph=graph, costs=costs, budget=k)
        # print(f"Seed Set size for budget {k}: {len(seed_set)}")
        influence_set = simulate_cascade.simulate_cascade(graph=graph, seed_set=seed_set)
        influence_set_length = len(influence_set)
        results[k] = influence_set_length
        # print(f"Influence set size for budget {k} = {results[k]}")
        if results[k] >= graph.GetNodes():
            break
    return results
