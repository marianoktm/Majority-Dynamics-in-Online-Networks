import random
import snap

from src import simulate_cascade



import random
import snap


import random
import snap


def vertices_removal_experiment_recalc(
    graph,
    costs,
    budget,
    seed_set_function,
    max_steps=None,
    seed=None,
):
    """
    Randomly remove a node and all its immediate neighbors at each step,
    then recompute the seed set and measure influence on the modified graph.

    The original graph is not modified.

    Args:
        graph: Original SNAP undirected graph.
        costs: Dictionary mapping node IDs to costs.
        budget: Fixed seed-selection budget.
        seed_set_function: Function that computes the seed set.
        max_steps: Maximum number of removal steps; None means continue
                   until the graph is empty.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary mapping the cumulative number of removed nodes
        to the size of the resulting influence set.
        Example: {0: 100, 5: 85, 12: 60}
    """

    if max_steps is not None and max_steps < 0:
        raise ValueError("max_steps must be non-negative")

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

        # Recompute the seed set and influence on the current graph.
        remaining_costs = {node: costs[node] for node in remaining}

        seed_set = seed_set_function(graph=graph_copy, costs=remaining_costs, budget=budget)
        influence_set = simulate_cascade.simulate_cascade(graph=graph_copy, seed_set=seed_set)
        results[cumulatively_removed] = len(influence_set)
        print(f"Step {step}: Removed this step = 0 (measurement before removal), Cumulatively removed = {cumulatively_removed}, Nodes left = {len(remaining)}, Influence = {len(influence_set)}")

        # Stop if the graph is empty or the step limit is reached.
        if not remaining:
            break

        if max_steps is not None and step >= max_steps:
            break

        # Choose a random node and collect its immediate neighbors.
        v = rng.choice(remaining)
        to_remove = {v}

        NI = graph_copy.GetNI(v)
        to_remove.update(
            NI.GetNbrNId(i)
            for i in range(NI.GetDeg())
        )

        # Count the nodes removed in this step.
        removed_this_step = len(to_remove)

        # Remove the selected node and its neighbors.
        for node in to_remove:
            graph_copy.DelNode(node)

        cumulatively_removed += removed_this_step
        step += 1

        print(f"Removed node {v} and its neighbors: {removed_this_step} nodes")
        print(f"Cumulative nodes removed: {cumulatively_removed}")

    return results



def vertices_removal_experiment(
    graph,
    costs,
    budget,
    seed_set_function,
    max_steps=None,
    seed=None,
):
    """
    Randomly remove a node and its immediate neighbors at each step.

    The seed set is computed once on the original graph and reused
    throughout the experiment. Influence is measured on each modified
    graph using the surviving original seed nodes.

    The original graph is not modified.
    """
    if max_steps is not None and max_steps < 0:
        raise ValueError("max_steps must be non-negative")

    # Compute the seed set on the original, unmodified graph.
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
        graph_copy.AddEdge(edge.GetSrcNId(), edge.GetDstNId())

    rng = random.Random(seed)

    results = {}
    cumulatively_removed = 0
    step = 0

    while True:
        remaining = [
            node.GetId() for node in graph_copy.Nodes()
        ]

        # Use only original seed nodes that still exist in the graph.
        active_seed_set = [
            node for node in original_seed_set
            if graph_copy.IsNode(node)
        ]

        # Measure influence before the next removal.
        influence_set = simulate_cascade.simulate_cascade(
            graph=graph_copy,
            seed_set=active_seed_set,
        )

        results[cumulatively_removed] = len(influence_set)

        print(
            f"Step {step}: "
            f"Cumulatively removed = {cumulatively_removed}, "
            f"Nodes left = {len(remaining)}, "
            f"Active seeds = {len(active_seed_set)}, "
            f"Influence = {len(influence_set)}"
        )

        # Stop if the graph is empty or the step limit is reached.
        if not remaining:
            break

        if max_steps is not None and step >= max_steps:
            break

        # Select a random node and its immediate neighbors.
        v = rng.choice(remaining)
        NI = graph_copy.GetNI(v)

        to_remove = {v}
        to_remove.update(
            NI.GetNbrNId(i)
            for i in range(NI.GetDeg())
        )

        removed_this_step = len(to_remove)

        for node in to_remove:
            graph_copy.DelNode(node)

        cumulatively_removed += removed_this_step
        step += 1

        print(
            f"Removed node {v} and its neighbors: "
            f"{removed_this_step} nodes"
        )

    return results
