from src import simulate_cascade

def budget_variation_experiment(graph, costs, budget, seed_set_function):
    max_cost = sum(costs.values())
    print("Max Cost: ", max_cost)
    results = {}
    for k in range(0, max_cost, 10):
        print("k:", k)
        seed_set = seed_set_function(graph=graph, costs=costs, budget=k)
        print(f"Seed Set size for budget {k}: {len(seed_set)}")
        influence_set = simulate_cascade.simulate_cascade(graph=graph, seed_set=seed_set)
        results[k] = len(influence_set)
        print(f"results[{k}] = {results[k]}")
    return results
