import math
import random
import snap

from src import simulate_cascade


def edges_removal_experiment(graph, costs, budget, seed_set_function, max_steps=None, seed=None):
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

        #print(f"Step {x}: Edges left = {graph_copy.GetEdges()}, Original seeds = {len(original_seed_set)}, Influence = {results[x]}")

    return results
