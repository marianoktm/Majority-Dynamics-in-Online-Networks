import math
import random
import snap

from src import simulate_cascade


def edges_removal_experiment(graph, costs, budget, seed_set_function):
    """
    Remove one random edge per step and measure the influence at each step.

    The budget is fixed. Step 0 is the original graph; at step i, i edges
    have been removed. The input graph is NOT modified (a copy is used).

    max_steps : stop after this many removed edges (default: remove all of them,
                which can be slow because the seed set is recomputed every step)
    seed      : seed of the random generator, for reproducible runs
    Returns   : dict {number of removed edges: influence set size}
    """
    graph_copy = snap.ConvertGraph(snap.TUNGraph, graph) # work on a copy
    rng = random.Random()

    edges = [(e.GetSrcNId(), e.GetDstNId()) for e in graph_copy.Edges()]
    rng.shuffle(edges) # random removal order

    results = {}
    no_of_edges = len(edges)
    for x in range(0, no_of_edges + 1, math.ceil(no_of_edges / 50)):
        if x > 0:
            graph_copy.DelEdge(*edges[x - 1]) # remove exactly one edge per step
        seed_set = seed_set_function(graph=graph, costs=costs, budget=budget)
        influence_set = simulate_cascade.simulate_cascade(graph=graph, seed_set=seed_set)
        influence_set_length = len(influence_set)
        results[x] = influence_set_length
        print(f"Step {x}: Edges left = {graph_copy.GetEdges()}")
        print(f"Influence set size: {influence_set_length}")
    return results