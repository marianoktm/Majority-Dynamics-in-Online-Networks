from concurrent.futures import ProcessPoolExecutor, as_completed

import os
import random
from itertools import product

import snap
import networkx
import matplotlib.pyplot as pyplot
import argparse
import math

from src import graph_utils, costs_functions, plotting
from src.experiments.run_experiment import run_experiment
from src.experiments import budget_variation, edges_removal, vertices_removal

from src.seedset.cost_seeds_greedy import cost_seeds_greedy, cost_seeds_greedy_best
from src.seedset.wtss import wtss
from src.seedset.centrality_cost_seeds_greedy import centrality_cost_seeds_greedy


def run_experiment_group(
    experiment_name,
    dataset_path,
    test_mode,
    budget,
    save_path,
):
    """Run all three cost configurations for one experiment type."""

    # Load a separate graph in each process.
    graph = graph_utils.read_snap_graph(
        file_path=dataset_path,
        graph_type="undirected",
        format_type="edgelist",
    )

    if test_mode:
        graph = graph_utils.get_connected_random_subgraph(
            graph=graph,
            sample_size=math.ceil(graph.GetNodes() * 0.05),
        )

    # Build costs locally in this worker.
    random_costs = costs_functions.random_cost(graph, 1, 10)
    degree_costs = costs_functions.degree_cost(graph)
    avg_neighbor_degree_costs = (
        costs_functions.avg_neighbor_degree_cost(graph)
    )

    experiment_pairs = {
        "Random Costs": {
            "random-costs-cost-seeds-greedy": (
                random_costs, cost_seeds_greedy_best
            ),
            "random-costs-wtss": (random_costs, wtss),
            "random-costs-centrality-cost-seeds-greedy": (
                random_costs, centrality_cost_seeds_greedy
            ),
        },
        "Degree Costs": {
            "degree-costs-cost-seeds-greedy": (
                degree_costs, cost_seeds_greedy_best
            ),
            "degree-costs-wtss": (degree_costs, wtss),
            "degree-costs-centrality-cost-seeds-greedy": (
                degree_costs, centrality_cost_seeds_greedy
            ),
        },
        "Average Neighbor Degree Costs": {
            "avg-neighbor-degree-costs-cost-seeds-greedy": (
                avg_neighbor_degree_costs, cost_seeds_greedy_best
            ),
            "avg-neighbor-degree-costs-wtss": (
                avg_neighbor_degree_costs, wtss
            ),
            "avg-neighbor-degree-costs-centrality-cost-seeds-greedy": (
                avg_neighbor_degree_costs,
                centrality_cost_seeds_greedy,
            ),
        },
    }

    experiment_functions = {
        "budget": (
            budget_variation.budget_variation_experiment,
            "Budget",
        ),
        "edges": (
            edges_removal.edges_removal_experiment,
            "Removed Edges",
        ),
        "vertices": (
            vertices_removal.vertices_removal_experiment,
            "Removed Vertices",
        ),
    }

    experiment_function, xlabel = experiment_functions[experiment_name]
    results = {}

    for cost_name, pairs in experiment_pairs.items():
        kwargs = {
            "experiment_function": experiment_function,
            "graph": graph,
            "experiment_pairs": pairs,
        }

        if experiment_name != "budget":
            kwargs["budget"] = budget

        result = run_experiment(**kwargs)
        results[cost_name] = result

        print(
            f"\n[{experiment_name.upper()}] "
            f"{cost_name} results:\n{result}",
            flush=True,
        )

        plotting.plot_experiment_results(
            data=result,
            xlabel=xlabel,
            ylabel="|Inf[G,S]|",
            title=f"{experiment_name.title()} Experiment Results "
                  f"({cost_name})",
            save_path=save_path,
        )

    return experiment_name, results

# Parser
parser = argparse.ArgumentParser()

# Dataset input argument.
parser.add_argument('-d',
                    '--dataset', type=str,
                    help='Path of the graph to be loaded')

# Test mode argument (uses a subgraph for performance reasons).
parser.add_argument('--test',
                    action='store_true',
                    help='Enable test mode to run on a smaller random subgraph for performance')

# Budget input argument (if not set, calculates and uses the average budget based on random costs).
parser.add_argument('-b',
                    '--budget',
                    type=int,
                    default=None,
                    help='Budget available for seed selection. If not available, calculates the average budget based on random costs')

# Save path argument for saving plots figures (defaults to "results" in the current working directory)
parser.add_argument('-s',
                    '--save-path',
                    type=str,
                    default=os.path.join(os.getcwd(), 'results'),
                    help='Path to the results folder to save the plots (defaults to the "result" folder in the current working directory)')


def main():
    args = parser.parse_args()

    if not args.dataset:
        parser.error("Please provide a dataset using -d or --dataset.")

    data_path = args.dataset
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)

    print("Data Path:", data_path)

    # Load the graph to determine the budget.
    graph = graph_utils.read_snap_graph(
        file_path=data_path,
        graph_type="undirected",
        format_type="edgelist",
    )

    print(
        f"Graph Nodes: {graph.GetNodes()}, "
        f"Graph Edges: {graph.GetEdges()}"
    )

    if args.test:
        print("Test mode enabled: using a 5% random subgraph.")
        graph = graph_utils.get_connected_random_subgraph(
            graph=graph,
            sample_size=math.ceil(graph.GetNodes() * 0.05),
        )

    # Compute the budget before starting the parallel jobs.
    random_costs = costs_functions.random_cost(graph, 1, 10)

    if args.budget is not None:
        budget = args.budget
    else:
        budget = math.ceil(
            costs_functions.average_cost(random_costs)
            * graph.GetNodes()
            * 0.20
        )

    print("Budget:", budget)
    print("Starting the three experiments in parallel...")

    experiment_names = ["budget", "edges", "vertices"]
    results = {}

    # One process per experiment type.
    with ProcessPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(
                run_experiment_group,
                experiment_name,
                data_path,
                args.test,
                budget,
                save_path,
            ): experiment_name
            for experiment_name in experiment_names
        }

        for future in as_completed(futures):
            experiment_name = futures[future]

            try:
                name, experiment_results = future.result()
                results[name] = experiment_results
                print(f"Completed experiment: {name}", flush=True)

            except Exception as exc:
                print(
                    f"Experiment '{experiment_name}' failed: {exc}",
                    flush=True,
                )
                raise

    print("\nAll three experiment types completed.")
    for experiment_name, experiment_results in results.items():
        print(f"\n{experiment_name.upper()}")
        for cost_name, result in experiment_results.items():
            print(f"\n{cost_name}:\n{result}")


if __name__ == "__main__":
    main()