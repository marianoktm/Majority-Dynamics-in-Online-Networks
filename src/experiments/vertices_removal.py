import random
import snap

from src import simulate_cascade


def vertices_removal_experiment(graph, costs, budget, seed_set_function):
    """
    Remove one random vertex per step (together with its neighbors if
    remove_neighbors=True) and measure the influence at each step.

    The budget is fixed. Step 0 is the original graph. The input graph is NOT
    modified (a copy is used). The loop stops when the graph is empty or after
    max_steps removals.

    Returns : dict {step: influence set size}
    """
    G = snap.ConvertGraph(snap.TUNGraph, graph)  # work on a copy
    rng = random.Random()

    results = {}
    step = 0
    while True:
        # the seed set functions expect a cost for every node still in the graph
        remaining = [NI.GetId() for NI in G.Nodes()]

        seed_set = seed_set_function(graph=graph, costs=costs, budget=budget)
        influence_set = simulate_cascade.simulate_cascade(graph=graph, seed_set=seed_set)
        influence_set_length = len(influence_set)
        results[step] = influence_set_length
        print(f"Step {step}: Nodes left = {len(remaining)}")
        print(f"Influence set size: {influence_set_length}")

        # pick the vertex to remove
        v = rng.choice(remaining)
        to_remove = {v}
        # remove its neighbors
        NI = G.GetNI(v)
        to_remove.update(NI.GetNbrNId(i) for i in range(NI.GetDeg()))

        for n in to_remove:
            G.DelNode(n)  # also deletes its incident edges
        step += 1
        if G.GetNodes() == 0:
            break

    return results
