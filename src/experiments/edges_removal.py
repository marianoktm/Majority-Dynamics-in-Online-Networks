"""
Edges removal experiment.

Computes a seed set once on the original graph and then measures how its influence
decreases as edges are progressively removed, in random order, from a copy of the
graph (robustness of the seed set to edge failures).
"""

import math
import random
import snap

from src import simulate_cascade


def edges_removal_experiment(graph, costs, budget, seed_set_function, max_steps=None, seed=None):
    """
    Function: edges_removal_experiment

    Computes a seed set on the original graph, then removes edges randomly (on a copy) and
    measures how the influence of that fixed seed set changes at about 50 checkpoints.

    Args:
        graph (snap.TUNGraph): Undirected graph (not modified).
        costs (dict): Maps each node ID (int) to its cost (int).
        budget (int): Maximum total cost allowed for the seed set.
        seed_set_function (callable): Function (graph, costs, budget) returning a seed set.
        max_steps (int, optional): Maximum number of edges to remove; all edges if None. Must be non-negative.
        seed (int, optional): Seed for the random edge order.

    Returns:
        dict: Maps number of removed edges (int) to the number of influenced nodes (int).

    Raises:
        ValueError: If max_steps is negative.
    """
    if max_steps is not None and max_steps < 0:
        raise ValueError("max_steps must be non-negative")

    # Compute the seed set BEFORE modifying the graph.
    original_seed_set = seed_set_function(graph=graph, costs=costs, budget=budget)

    # Copy the graph without modifying the original.
    graph_copy = snap.TUNGraph.New()

    for node in graph.Nodes():
        graph_copy.AddNode(node.GetId())

    for edge in graph.Edges():
        graph_copy.AddEdge(edge.GetSrcNId(), edge.GetDstNId())

    # Randomize edge-removal order.
    rng = random.Random(seed)
    edges = [(e.GetSrcNId(), e.GetDstNId()) for e in graph_copy.Edges()]
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
        influence_set = simulate_cascade.simulate_cascade(graph=graph_copy, seed_set=original_seed_set)

        results[x] = len(influence_set)

        # print(f"Step {x}: Edges left = {graph_copy.GetEdges()}, Original seeds = {len(original_seed_set)}, Influence = {results[x]}")

    return results
