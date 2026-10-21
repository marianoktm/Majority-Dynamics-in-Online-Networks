import math

from src import simulate_cascade


def budget_variation_experiment(graph, costs, budget, seed_set_function):
    max_cost = sum(costs.values())
    #print("Max Cost: ", max_cost)
    results = {}
    for k in range(0, max_cost, math.ceil(max_cost/50)):
        #print("k:", k)
        seed_set = seed_set_function(graph=graph, costs=costs, budget=k)
        #print(f"Seed Set size for budget {k}: {len(seed_set)}")
        influence_set = simulate_cascade.simulate_cascade(graph=graph, seed_set=seed_set)
        influence_set_length = len(influence_set)
        results[k] = influence_set_length
        #print(f"Influence set size for budget {k} = {results[k]}")
        if results[k] >= graph.GetNodes():
            break
    return results
