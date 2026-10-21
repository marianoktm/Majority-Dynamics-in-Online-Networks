"""
Experiment runner.

Provides a generic helper that executes one experiment function for every
(costs, seed-set function) pair on the same graph and collects the results by label.
"""

def run_experiment(experiment_function, graph, experiment_pairs, budget=0):
    """
    Function: run_experiment

    Runs one experiment function for every (costs, seed-set function) pair on the same graph.

    Args:
        experiment_function (callable): Experiment taking (graph, costs, seed_set_function, budget).
        graph (snap graph): Graph used by every run.
        experiment_pairs (dict): Maps a label (str) to a tuple (costs, seed_set_function).
        budget (int, optional): Budget forwarded to the experiment. Default 0.

    Returns:
        dict: Maps each label to the result returned by the experiment function.
    """
    # print("Running experiment: ", experiment_function.__name__)
    results = {}
    # Run the experiment function for each pair of seed_set_function and graph_costs
    for name, (costs, seed_set_function) in experiment_pairs.items():
        # print("Running with: ", name)
        # print("Costs:", costs)
        # print("Seed Set Function:", seed_set_function.__name__)
        results[name] = experiment_function(graph=graph, costs=costs, seed_set_function=seed_set_function,
                                            budget=budget)
    return results
