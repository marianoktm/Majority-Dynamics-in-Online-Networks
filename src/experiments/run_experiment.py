def run_experiment(experiment_function, graph, experiment_pairs, budget = 0):
    print("Running experiment: ", experiment_function.__name__)

    results = {}

    # Run the experiment function for each pair of seed_set_function and graph_costs
    for name, (costs, seed_set_function) in experiment_pairs.items():
        print("Running with: ", name)
        print("Costs:", costs)
        print("Seed Set Function:", seed_set_function.__name__)

        results[name] = experiment_function(graph=graph, costs=costs, seed_set_function=seed_set_function, budget=budget)
    return results

