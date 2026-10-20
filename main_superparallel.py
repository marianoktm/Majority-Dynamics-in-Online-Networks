import os
import math
import time
import random
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

from src import graph_utils, costs_functions, plotting
from src.experiments.run_experiment import run_experiment
from src.experiments import (
    budget_variation,
    edges_removal,
    vertices_removal,
)

from src.seedset.cost_seeds_greedy import cost_seeds_greedy_best
from src.seedset.wtss import wtss
from src.seedset.centrality_cost_seeds_greedy import (
    centrality_cost_seeds_greedy,
)


# ================================================================
# Configuration
# ================================================================

RANDOM_SEED = 42

parser = argparse.ArgumentParser()

parser.add_argument(
    "-d", "--dataset",
    type=str,
    required=True,
    help="Path to the input graph edge list",
)

parser.add_argument(
    "--test",
    action="store_true",
    help="Run experiments on a 5% connected random subgraph",
)

parser.add_argument(
    "-b", "--budget",
    type=int,
    default=None,
    help="Seed-selection budget",
)

parser.add_argument(
    "-s", "--save-path",
    type=str,
    default=os.path.join(os.getcwd(), "results"),
    help="Directory for experiment plots",
)

parser.add_argument(
    "-w", "--workers",
    type=int,
    default=9,
    help="Maximum number of parallel worker processes",
)


# ================================================================
# Worker-local state
# These objects are created separately in every worker process.
# ================================================================

_WORKER_GRAPH = None
_WORKER_COSTS = None
_WORKER_EXPERIMENTS = None


def initialize_worker(graph_path, random_seed):
    """
    Initialize graph, costs, and experiment implementations locally.

    No SNAP objects are sent between processes.
    """

    global _WORKER_GRAPH
    global _WORKER_COSTS
    global _WORKER_EXPERIMENTS

    random.seed(random_seed)

    _WORKER_GRAPH = graph_utils.read_snap_graph(
        file_path=graph_path,
        graph_type="undirected",
        format_type="edgelist",
    )

    # Compute all cost dictionaries inside this worker.
    random_costs = costs_functions.random_cost(
        _WORKER_GRAPH, 1, 10
    )

    degree_costs = costs_functions.degree_cost(
        _WORKER_GRAPH
    )

    avg_neighbor_degree_costs = (
        costs_functions.avg_neighbor_degree_cost(
            _WORKER_GRAPH
        )
    )

    _WORKER_COSTS = {
        "random": random_costs,
        "degree": degree_costs,
        "avg_neighbor_degree": avg_neighbor_degree_costs,
    }

    _WORKER_EXPERIMENTS = {
        "budget": budget_variation.budget_variation_experiment,
        "edges": edges_removal.edges_removal_experiment,
        "vertices": vertices_removal.vertices_removal_experiment,
    }

    print(
        f"[WORKER READY] PID={os.getpid()}, "
        f"nodes={_WORKER_GRAPH.GetNodes()}, "
        f"edges={_WORKER_GRAPH.GetEdges()}",
        flush=True,
    )


# ================================================================
# Experiment worker
# Only serializable strings, numbers, and booleans are submitted.
# ================================================================

def execute_experiment(
    experiment_name,
    experiment_type,
    cost_type,
    save_path,
    budget,
    random_seed,
):
    global _WORKER_GRAPH
    global _WORKER_COSTS
    global _WORKER_EXPERIMENTS

    start_time = time.perf_counter()

    # Build the algorithm/cost pairs locally.
    # The cost dictionaries never leave the worker process.
    algorithms = {
        "cost_seeds_greedy": cost_seeds_greedy_best,
        "wtss": wtss,
        "centrality_cost_seeds_greedy": (
            centrality_cost_seeds_greedy
        ),
    }

    experiment_pairs = {}

    for algorithm_name, algorithm in algorithms.items():
        experiment_pairs[
            f"{cost_type}-{algorithm_name}"
        ] = (
            _WORKER_COSTS[cost_type],
            algorithm,
        )

    experiment_function = _WORKER_EXPERIMENTS[
        experiment_type
    ]

    print(
        f"[START] {experiment_name} "
        f"(PID={os.getpid()})",
        flush=True,
    )

    kwargs = {
        "experiment_function": experiment_function,
        "graph": _WORKER_GRAPH,
        "experiment_pairs": experiment_pairs,
    }

    if experiment_type != "budget":
        kwargs["budget"] = budget

    results = run_experiment(**kwargs)

    # Save each experiment's plot using its own title.
    if experiment_type == "budget":
        xlabel = "Budget"
        ylabel = "|Inf[G,S]|"
    elif experiment_type == "edges":
        xlabel = "Removed Edges"
        ylabel = "|Inf[G',S]|"
    else:
        xlabel = "Removed Vertices"
        ylabel = "|Inf[G',S]|"

    plotting.plot_experiment_results(
        data=results,
        xlabel=xlabel,
        ylabel=ylabel,
        title=experiment_name,
        save_path=save_path,
    )

    elapsed = time.perf_counter() - start_time

    print(
        f"[DONE] {experiment_name}: "
        f"{elapsed:.2f} seconds",
        flush=True,
    )

    # Return only the results and timing, never the graph.
    return experiment_name, results, elapsed


# ================================================================
# Graph serialization for test mode
# ================================================================

def save_snap_graph_as_edgelist(graph, file_path):
    """
    Save a SNAP graph as a plain-text edge list.

    This allows worker processes to load the exact same subgraph
    without trying to pickle a SWIG-backed SNAP object.
    """

    with open(file_path, "w", encoding="utf-8") as file:
        for edge in graph.Edges():
            source = edge.GetSrcNId()
            target = edge.GetDstNId()
            file.write(f"{source} {target}\n")


# ================================================================
# Main
# ================================================================

def main():
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be at least 1")

    if args.budget is not None and args.budget < 0:
        parser.error("--budget cannot be negative")

    os.makedirs(args.save_path, exist_ok=True)

    # ------------------------------------------------------------
    # Load graph in the parent process.
    # ------------------------------------------------------------

    print(f"Loading graph: {args.dataset}", flush=True)

    graph = graph_utils.read_snap_graph(
        file_path=args.dataset,
        graph_type="undirected",
        format_type="edgelist",
    )

    print(
        f"Original graph: {graph.GetNodes()} nodes, "
        f"{graph.GetEdges()} edges",
        flush=True,
    )

    # ------------------------------------------------------------
    # Generate test graph once, not once per worker.
    # ------------------------------------------------------------

    graph_path = args.dataset

    if args.test:
        print("Generating test subgraph...", flush=True)

        random.seed(RANDOM_SEED)

        graph = graph_utils.get_connected_random_subgraph(
            graph=graph,
            sample_size=math.ceil(
                graph.GetNodes() * 0.05
            ),
        )

        graph_path = os.path.join(
            args.save_path,
            "_parallel_test_graph.edgelist",
        )

        save_snap_graph_as_edgelist(graph, graph_path)

        print(
            f"Test graph: {graph.GetNodes()} nodes, "
            f"{graph.GetEdges()} edges",
            flush=True,
        )

    print("=" * 70)

    # ------------------------------------------------------------
    # Determine budget.
    #
    # Use the same random seed as the workers so that random costs
    # are reproducible, assuming random_cost uses Python's random
    # module and iterates nodes deterministically.
    # ------------------------------------------------------------

    random.seed(RANDOM_SEED)

    random_costs = costs_functions.random_cost(
        graph, 1, 10
    )

    if args.budget is not None:
        budget = args.budget
    else:
        budget = math.ceil(
            costs_functions.average_cost(random_costs)
            * graph.GetNodes()
            * 0.20
        )

    print(f"Budget: {budget}", flush=True)
    print("=" * 70)

    # ------------------------------------------------------------
    # Define the nine experiment configurations.
    #
    # Each tuple contains only serializable values.
    # ------------------------------------------------------------

    experiments = [
        # Budget variation
        (
            "budget-random-costs",
            "budget",
            "random",
        ),
        (
            "budget-degree-costs",
            "budget",
            "degree",
        ),
        (
            "budget-average-neighbor-degree-costs",
            "budget",
            "avg_neighbor_degree",
        ),

        # Edge removal
        (
            "edges-random-costs",
            "edges",
            "random",
        ),
        (
            "edges-degree-costs",
            "edges",
            "degree",
        ),
        (
            "edges-average-neighbor-degree-costs",
            "edges",
            "avg_neighbor_degree",
        ),

        # Vertex removal
        (
            "vertices-random-costs",
            "vertices",
            "random",
        ),
        (
            "vertices-degree-costs",
            "vertices",
            "degree",
        ),
        (
            "vertices-average-neighbor-degree-costs",
            "vertices",
            "avg_neighbor_degree",
        ),
    ]

    # ------------------------------------------------------------
    # Execute all nine experiments concurrently.
    # ------------------------------------------------------------

    results = {}
    timings = {}

    total_start = time.perf_counter()

    worker_count = min(
        args.workers,
        len(experiments),
    )

    print(
        f"Launching {len(experiments)} experiments "
        f"with {worker_count} workers.",
        flush=True,
    )

    total_start = time.perf_counter()

    with ProcessPoolExecutor(
        max_workers=worker_count,
        initializer=initialize_worker,
        initargs=(graph_path, RANDOM_SEED),
    ) as executor:

        future_to_name = {}

        for name, experiment_type, cost_type in experiments:
            future = executor.submit(
                execute_experiment,
                name,
                experiment_type,
                cost_type,
                args.save_path,
                budget,
                RANDOM_SEED,
            )

            future_to_name[future] = name

        errors = []

        for future in as_completed(future_to_name):
            name = future_to_name[future]

            try:
                experiment_name, experiment_results, elapsed = (
                    future.result()
                )

                results[experiment_name] = experiment_results
                timings[experiment_name] = elapsed

            except Exception as exc:
                errors.append((name, repr(exc)))

                print(
                    f"[ERROR] {name}: {exc}",
                    flush=True,
                )

    total_elapsed = time.perf_counter() - total_start

    # ------------------------------------------------------------
    # Print summary.
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("EXECUTION TIME SUMMARY")
    print("=" * 70)

    for name, elapsed in sorted(
        timings.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(f"{name:<48} {elapsed:>9.2f} s")

    print("-" * 70)
    print(f"Successful experiments: {len(results)}/9")
    print(f"Failed experiments:     {len(errors)}/9")
    print(f"Total elapsed time:     {total_elapsed:.2f} s")

    if errors:
        print("\nFailures:")
        for name, error in errors:
            print(f"  {name}: {error}")

        raise RuntimeError(
            f"{len(errors)} experiment(s) failed. "
            "See errors above."
        )

    print("\nAll experiments completed successfully.")

    for name, experiment_results in results.items():
        print(f"\n{name}:")
        print(experiment_results)

    total_elapsed = time.perf_counter() - total_start
    print(f"\nTotal parallel execution time: {total_elapsed:.2f} seconds")

if __name__ == "__main__":
    main()