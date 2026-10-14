def run_experiment(experiment_function, graph, experiment_pairs, budget = 0):
    # TODO: This function should run the experiment and return something like results[num, InfGS_size]
    # TODO: where num can be:
    # TODO: budget variation experiment: [k, InfGS_size]
    # TODO: edge removal experiment: [x, InfGS_size]
    # TODO: vertex removal experiment: [y, InfGS_size]

    print("Running experiment: ", experiment_function.__name__)

    # Run the experiment function for each pair of seed_set_function and graph_costs
    for seed_set_function, graph_cost in experiment_pairs:
        print("Graph costs: ", graph_cost)
        print("Seed set function: ", seed_set_function.__name__)


    return
