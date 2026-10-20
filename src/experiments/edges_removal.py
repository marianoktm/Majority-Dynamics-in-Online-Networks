import math
import random
import snap

import graph_utils
from src import simulate_cascade


def edges_removal_experiment_recalc(graph, costs, budget, seed_set_function, max_steps=None, seed=None):
    """
    Randomly remove edges and measure influence.

    Step 0: original graph.
    Step i: exactly i edges have been removed.

    Measurements are taken at approximately 50 evenly spaced
    intervals, including step 0 and the final step.

    Args:
        graph: Original SNAP undirected graph.
        costs: Node costs passed to seed_set_function.
        budget: Budget passed to seed_set_function.
        seed_set_function: Function that computes the seed set.
        max_steps: Maximum number of edges to remove.
                   Defaults to removing all edges.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary mapping the number of removed edges
        to the influence set size.
    """

    # Copy the graph without modifying the original.
    graph_copy = snap.TUNGraph.New()

    for node in graph.Nodes():
        graph_copy.AddNode(node.GetId())

    for edge in graph.Edges():
        graph_copy.AddEdge(edge.GetSrcNId(), edge.GetDstNId())

    # Randomize edge removal order.
    rng = random.Random(seed)
    edges = [(e.GetSrcNId(), e.GetDstNId()) for e in graph_copy.Edges()]
    rng.shuffle(edges)

    no_of_edges = len(edges)

    # Determine the maximum number of edges to remove.
    if max_steps is None:
        max_removed = no_of_edges
    else:
        if max_steps < 0:
            raise ValueError("max_steps must be non-negative")
        max_removed = min(max_steps, no_of_edges)

    # Choose measurement points, including both endpoints.
    step_size = max(1, math.ceil(no_of_edges / 50))
    checkpoints = list(range(0, max_removed + 1, step_size))

    if checkpoints[-1] != max_removed:
        checkpoints.append(max_removed)

    results = {}
    removed = 0

    for x in checkpoints:
        # Remove every edge needed to reach exactly x removals.
        while removed < x:
            u, v = edges[removed]
            graph_copy.DelEdge(u, v)
            removed += 1
        seed_set = seed_set_function(graph=graph_copy, costs=costs, budget=budget)
        influence_set = simulate_cascade.simulate_cascade(graph=graph_copy, seed_set=seed_set)
        results[x] = len(influence_set)
        print(f"Step {x}: Edges left = {graph_copy.GetEdges()}")
        print(f"Influence set size with {x} edges removed: {results[x]}")

    return results


def edges_removal_experiment(
    graph,
    costs,
    budget,
    seed_set_function,
    max_steps=None,
    seed=None,
):
    """
    Randomly remove edges and measure influence using the original seed set.

    Step 0: original graph.
    Step i: exactly i edges have been removed.
    The seed set is computed once on the original graph and remains fixed.
    The original graph is not modified.
    """

    if max_steps is not None and max_steps < 0:
        raise ValueError("max_steps must be non-negative")

    # Compute the seed set BEFORE modifying the graph.
    original_seed_set = seed_set_function(
        graph=graph,
        costs=costs,
        budget=budget,
    )

    # Copy the graph without modifying the original.
    graph_copy = snap.TUNGraph.New()

    for node in graph.Nodes():
        graph_copy.AddNode(node.GetId())

    for edge in graph.Edges():
        graph_copy.AddEdge(
            edge.GetSrcNId(),
            edge.GetDstNId(),
        )

    # Randomize edge-removal order.
    rng = random.Random(seed)
    edges = [
        (e.GetSrcNId(), e.GetDstNId())
        for e in graph_copy.Edges()
    ]
    rng.shuffle(edges)

    no_of_edges = len(edges)

    # Determine the maximum number of edges to remove.
    if max_steps is None:
        max_removed = no_of_edges
    else:
        max_removed = min(max_steps, no_of_edges)

    # Choose measurement points, including both endpoints.
    step_size = max(1, math.ceil(no_of_edges / 50))
    checkpoints = list(range(0, max_removed + 1, step_size))

    if not checkpoints or checkpoints[-1] != max_removed:
        checkpoints.append(max_removed)

    results = {}
    removed = 0

    for x in checkpoints:
        # Remove edges until exactly x edges have been removed.
        while removed < x:
            u, v = edges[removed]
            graph_copy.DelEdge(u, v)
            removed += 1

        # Reuse the original seed set without recomputing it.
        influence_set = simulate_cascade.simulate_cascade(
            graph=graph_copy,
            seed_set=original_seed_set,
        )

        results[x] = len(influence_set)

        print(
            f"Step {x}: "
            f"Edges left = {graph_copy.GetEdges()}, "
            f"Original seeds = {len(original_seed_set)}, "
            f"Influence = {results[x]}"
        )

    return results
