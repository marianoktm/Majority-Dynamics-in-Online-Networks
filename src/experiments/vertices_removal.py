"""
Vertices removal experiment.

Computes a seed set once on the original graph and then measures how its influence
changes as randomly chosen vertices, together with their neighbors, are progressively
removed from a copy of the graph (robustness of the seed set to node failures).
"""

import random
import snap

from src import simulate_cascade


def vertices_removal_experiment(graph, costs, budget, seed_set_function, max_steps=None, seed=None):
    """
    Function: vertices_removal_experiment

    Computes a seed set on the original graph, then repeatedly removes a random node
    together with its neighbors (on a copy) and measures the influence of the surviving
    seeds after each step.

    Args:
        graph (snap.TUNGraph): Undirected graph (not modified).
        costs (dict): Maps each node ID (int) to its cost (int).
        budget (int): Maximum total cost allowed for the seed set.
        seed_set_function (callable): Function (graph, costs, budget) returning a seed set.
        max_steps (int, optional): Maximum number of removal steps; until the graph is empty if None. Must be non-negative.
        seed (int, optional): Seed for the random node choice.

    Returns:
        dict: Maps cumulative number of removed nodes (int) to the number of influenced nodes (int).

    Raises:
        ValueError: If max_steps is negative.
    """
    if max_steps is not None and max_steps < 0:
        raise ValueError("max_steps must be non-negative")

    # Compute the seed set on the original, unmodified graph.
    original_seed_set = seed_set_function(graph=graph, costs=costs, budget=budget)

    # Copy the graph without modifying the original.
    graph_copy = snap.TUNGraph.New()

    for node in graph.Nodes():
        graph_copy.AddNode(node.GetId())

    for edge in graph.Edges():
        graph_copy.AddEdge(edge.GetSrcNId(), edge.GetDstNId())

    rng = random.Random(seed)

    results = {}
    cumulatively_removed = 0
    step = 0

    while True:
        remaining = [node.GetId() for node in graph_copy.Nodes()]

        # Use only original seed nodes that still exist in the graph.
        active_seed_set = [node for node in original_seed_set if graph_copy.IsNode(node)]

        # Measure influence before the next removal.
        influence_set = simulate_cascade.simulate_cascade(graph=graph_copy, seed_set=active_seed_set)
        results[cumulatively_removed] = len(influence_set)

        # print(f"Step {step}: Cumulatively removed = {cumulatively_removed}, Nodes left = {len(remaining)}, Active seeds = {len(active_seed_set)}, Influence = {len(influence_set)}")

        # Stop if the graph is empty or the step limit is reached.
        if not remaining:
            break
        if max_steps is not None and step >= max_steps:
            break

        # Select a random node and its immediate neighbors.
        v = rng.choice(remaining)
        NI = graph_copy.GetNI(v)

        to_remove = {v}
        to_remove.update(NI.GetNbrNId(i) for i in range(NI.GetDeg()))

        removed_this_step = len(to_remove)

        for node in to_remove:
            graph_copy.DelNode(node)

        cumulatively_removed += removed_this_step
        step += 1
        # print(f"Removed node {v} and its neighbors: {removed_this_step} nodes")

    return results
