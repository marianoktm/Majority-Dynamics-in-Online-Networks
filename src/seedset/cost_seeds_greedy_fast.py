"""
Cost-Seeds-Greedy, fast implementation.

Greedy seed selection using target functions f1, f2 and f3 with the majority
(half-degree) threshold. Optimization over the slow version: the marginal gain of a
candidate is computed locally from its neighbors instead of re-evaluating the target
function on the whole graph. Every selection still scans all candidates linearly.
"""

import math
import snap


def _seed_set_cost(seed_set, costs):
    """
    Function: _seed_set_cost

    Computes the total cost of a group of seed nodes. Nodes missing from `costs` default to
    cost 1.

    Args:
        seed_set (set): Node IDs of the seeds.
        costs (dict): Maps node ID to cost (int).

    Returns:
        int: Sum of the costs of all nodes in `seed_set`.
    """
    return sum(costs.get(node, 1) for node in seed_set)


def _get_adjacency_dictionary(graph):
    """
    Function: _get_adjacency_dictionary

    Converts the SNAP graph into a plain Python dictionary {node_id: [neighbor_ids]}.
    Dictionary lookups are much faster than repeated calls into SNAP.

    Args:
        graph (snap.TUNGraph): Undirected SNAP graph.

    Returns:
        dict: Maps each node ID (int) to the list of its neighbor IDs (list of int).
    """
    adjacency = {}

    for NI in graph.Nodes():
        adjacency[NI.GetId()] = [NI.GetNbrNId(i) for i in range(NI.GetDeg())]
    return adjacency


def _half_degree_threshold(adjacency, node):
    """
    Function: _half_degree_threshold

    Computes the half-degree threshold of a node, i.e. how many seeded neighbors are needed
    to 'win it over': ceil(degree / 2).

    Args:
        adjacency (dict): Maps node ID to its list of neighbors.
        node (int): ID of the node.

    Returns:
        int: ceil(degree(node) / 2).
    """
    node_degree = len(adjacency[node])
    return math.ceil(node_degree / 2)


def _marginal_gain_f1(adjacency, seeded_neighbor_count, candidate_vertex):
    """
    Function: _marginal_gain_f1

    Marginal gain for target function f1. Each neighbor of the candidate that has not yet
    reached its half-degree threshold contributes +1.

    Args:
        adjacency (dict): Maps node ID to its list of neighbors.
        seeded_neighbor_count (dict): Maps node ID to how many of its neighbors are currently seeded.
        candidate_vertex (int): ID of the node whose marginal gain is evaluated.

    Returns:
        int or float: Marginal gain obtained by seeding `candidate_vertex` now.
    """
    total_gain = 0

    for neighbor in adjacency[candidate_vertex]:
        threshold = _half_degree_threshold(adjacency, neighbor)
        current_count = seeded_neighbor_count[neighbor]

        if current_count < threshold:
            total_gain += 1

    return total_gain


def _marginal_gain_f2(adjacency, seeded_neighbor_count, candidate_vertex):
    """
    Function: _marginal_gain_f2

    Marginal gain for target function f2. Each neighbor contributes (threshold - current
    seeded count) while this is positive, so early seeded neighbors are worth the most.

    Args:
        adjacency (dict): Maps node ID to its list of neighbors.
        seeded_neighbor_count (dict): Maps node ID to how many of its neighbors are currently seeded.
        candidate_vertex (int): ID of the node whose marginal gain is evaluated.

    Returns:
        int or float: Marginal gain obtained by seeding `candidate_vertex` now.
    """
    total_gain = 0

    for neighbor in adjacency[candidate_vertex]:
        threshold = _half_degree_threshold(adjacency, neighbor)
        current_count = seeded_neighbor_count[neighbor]

        remaining_to_threshold = threshold - current_count
        if remaining_to_threshold > 0:
            total_gain += remaining_to_threshold

    return total_gain


def _marginal_gain_f3(adjacency, seeded_neighbor_count, candidate_vertex):
    """
    Function: _marginal_gain_f3

    Marginal gain for target function f3. Like f2, but each neighbor's contribution is
    divided by its number of still-unseeded neighbors: (threshold - count) / (degree -
    count).

    Args:
        adjacency (dict): Maps node ID to its list of neighbors.
        seeded_neighbor_count (dict): Maps node ID to how many of its neighbors are currently seeded.
        candidate_vertex (int): ID of the node whose marginal gain is evaluated.

    Returns:
        int or float: Marginal gain obtained by seeding `candidate_vertex` now.
    """
    total_gain = 0.0

    for neighbor in adjacency[candidate_vertex]:
        neighbor_degree = len(adjacency[neighbor])
        threshold = _half_degree_threshold(adjacency, neighbor)
        current_count = seeded_neighbor_count[neighbor]

        remaining_to_threshold = threshold - current_count
        unseeded_neighbors = neighbor_degree - current_count

        # Skip if already past the threshold, or if the division would be invalid
        if remaining_to_threshold > 0 and unseeded_neighbors > 0:
            total_gain += remaining_to_threshold / unseeded_neighbors

    return total_gain


def _select_best_cost_effectiveness_vertex(adjacency, costs, current_seed_set, seeded_neighbor_count, target_function):
    """
    Function: _select_best_cost_effectiveness_vertex

    Linear scan over all vertices not yet in the seed set; returns the one with the highest
    cost-effectiveness (marginal gain / cost). Ties keep the first vertex found.

    Args:
        adjacency (dict): Maps node ID to its list of neighbors.
        costs (dict): Maps node ID to cost (default 1).
        current_seed_set (set): Nodes already selected.
        seeded_neighbor_count (dict): Maps node ID to its number of seeded neighbors.
        target_function (callable): Marginal gain function (f1, f2 or f3).

    Returns:
        int or None: Best vertex ID, or None if no candidate is left.
    """
    best_vertex = None
    best_cost_effectiveness = float("-inf")

    for candidate_vertex in adjacency:
        if candidate_vertex not in current_seed_set:
            marginal_gain = target_function(adjacency, seeded_neighbor_count, candidate_vertex)
            cost_effectiveness = marginal_gain / costs.get(candidate_vertex, 1)

            if cost_effectiveness > best_cost_effectiveness:
                best_cost_effectiveness = cost_effectiveness
                best_vertex = candidate_vertex

    return best_vertex


def _cost_seeds_greedy_single(adjacency, costs, budget, target_function):
    """
    Function: _cost_seeds_greedy_single

    Runs the greedy algorithm for one target function. Vertices are added by best cost-effectiveness
    until the total cost reaches or exceeds the budget. The vertex that made
    the budget be reached is discarded.

    Args:
        adjacency (dict): Maps node ID to its list of neighbors.
        costs (dict): Maps each node ID (int) to its cost (int).
        budget (int): Maximum total cost allowed for the seed set.
        target_function (callable): Marginal gain function (f1, f2 or f3).

    Returns:
        set: Seed set as it was before the last vertex was added (the last set built by
            the loop before the budget was reached).
    """
    # initialize seeded neighbors count for each vertex
    seeded_neighbor_count = {nodes: 0 for nodes in adjacency}

    previous_seed_set = set()
    current_seed_set = set()

    while _seed_set_cost(current_seed_set, costs) < budget:
        best_vertex = _select_best_cost_effectiveness_vertex(adjacency, costs, current_seed_set, seeded_neighbor_count,
                                                             target_function)

        if best_vertex is None:
            break

        previous_seed_set = current_seed_set
        current_seed_set = previous_seed_set | {best_vertex}

        for neighbor in adjacency[best_vertex]:
            seeded_neighbor_count[neighbor] += 1

    return previous_seed_set


def cost_seeds_greedy(graph, costs, budget):
    """
    Function: cost_seeds_greedy

    Cost-Seeds-Greedy algorithm. Runs the greedy selection with target functions f1, f2 and
    f3 and returns the largest of the three resulting seed sets.

    Args:
        graph (snap.TUNGraph): Undirected SNAP graph.
        costs (dict): Maps each node ID (int) to its cost (int).
        budget (int): Maximum total cost allowed for the seed set.

    Returns:
        set: Node IDs of the chosen seed set.
    """
    # Build adjacency dictionary for performance purposes
    adjacency = _get_adjacency_dictionary(graph)

    # Execute Cost-Seeds-Greedy for each of the three target functions
    seed_set_1 = _cost_seeds_greedy_single(adjacency=adjacency, costs=costs, budget=budget,
                                           target_function=_marginal_gain_f1)
    seed_set_2 = _cost_seeds_greedy_single(adjacency=adjacency, costs=costs, budget=budget,
                                           target_function=_marginal_gain_f2)
    seed_set_3 = _cost_seeds_greedy_single(adjacency=adjacency, costs=costs, budget=budget,
                                           target_function=_marginal_gain_f3)

    candidate_sets = [seed_set_1, seed_set_2, seed_set_3]

    # Return the biggest seed set
    return max(candidate_sets, key=len)
