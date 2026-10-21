"""
Cost-Seeds-Greedy, slow (reference) implementation.

Direct implementation of the algorithm from the Social Network course slides:
target functions f1, f2 and f3 are re-evaluated on the whole graph for every
candidate at every step. Simple and easy to verify, but expensive on large graphs.
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


def _f1(adjacency, seed_set):
    """
    Function: _f1

    Target function f1: sum over all vertices v of min(|N(v) ∩ S|, ceil(deg(v) / 2)).

    Args:
        adjacency (dict): Maps node ID to its list of neighbors.
        seed_set (set): Seed nodes S.

    Returns:
        float: Value of f1 for the seed set.
    """
    total = 0.0

    for v, v_neighbors in adjacency.items():
        v_degree = len(v_neighbors)
        neighbors_seed_set_intersection_size = sum(u in seed_set for u in v_neighbors)
        total += min(neighbors_seed_set_intersection_size, math.ceil(v_degree / 2))

    return total


def _f2(adjacency, seed_set):
    """
    Function: _f2

    Target function f2: for each vertex v, sums max(ceil(deg(v)/2) - i + 1, 0) for i = 1...|N(v) ∩ S|.

    Args:
        adjacency (dict): Maps node ID to its list of neighbors.
        seed_set (set): Seed nodes S.

    Returns:
        float: Value of f2 for the seed set.
    """
    total = 0.0

    for v, v_neighbors in adjacency.items():
        v_degree = len(v_neighbors)
        neighbors_seed_set_intersection_size = sum(u in seed_set for u in v_neighbors)

        for i in range(1, neighbors_seed_set_intersection_size + 1):
            total += max(math.ceil(v_degree / 2) - i + 1, 0)

    return total


def _f3(adjacency, seed_set):
    """
    Function: _f3

    Target function f3: for each vertex v, sums max((ceil(deg(v)/2) - i + 1) / (deg(v) - i + 1), 0) for i = 1...|N(v) ∩ S|.

    Args:
        adjacency (dict): Maps node ID to its list of neighbors.
        seed_set (set): Seed nodes S.

    Returns:
        float: Value of f3 for the seed set.
    """
    total = 0.0

    for v, v_neighbors in adjacency.items():
        v_degree = len(v_neighbors)
        neighbors_seed_set_intersection_size = sum(u in seed_set for u in v_neighbors)

        for i in range(1, neighbors_seed_set_intersection_size + 1):
            total += max((math.ceil(v_degree / 2) - i + 1) / (v_degree - i + 1), 0)

    return total


def _select_best_cost_effectiveness_vertex(adjacency, costs, current_seed_set, target_function):
    """
    Function: _select_best_cost_effectiveness_vertex

    Evaluates the target function from scratch for every candidate vertex and returns the
    one with the highest marginal gain / cost.

    Args:
        adjacency (dict): Maps node ID to its list of neighbors.
        costs (dict): Maps node ID to cost (default 1).
        current_seed_set (set): Nodes already selected.
        target_function (callable): Target function (_f1, _f2 or _f3).

    Returns:
        int or None: Best vertex ID, or None if no candidate is left.
    """
    current_value = target_function(adjacency, current_seed_set)

    max_cost_effectiveness = float("-inf")
    best_vertex = None

    for v in adjacency:
        if v not in current_seed_set:
            new_seed_set = current_seed_set | {v}
            marginal_gain = (target_function(adjacency, new_seed_set) - current_value)
            cost_effectiveness = marginal_gain / costs.get(v, 1)

            if cost_effectiveness > max_cost_effectiveness:
                max_cost_effectiveness = cost_effectiveness
                best_vertex = v

    return best_vertex


def _cost_seeds_greedy_single(adjacency, costs, budget, target_function):
    """
    Function: _cost_seeds_greedy_single

    Runs the (slow) greedy algorithm for one target function until the total cost reaches or
    exceeds the budget; the last added vertex is discarded.

    Args:
        adjacency (dict): Maps node ID to its list of neighbors.
        costs (dict): Maps each node ID (int) to its cost (int).
        budget (int): Maximum total cost allowed for the seed set.
        target_function (callable): Target function (_f1, _f2 or _f3).

    Returns:
        set: Seed set as it was before the last vertex was added.
    """
    previous_seed_set = set()
    current_seed_set = set()

    while _seed_set_cost(current_seed_set, costs) < budget:
        best_vertex = _select_best_cost_effectiveness_vertex(adjacency, costs, current_seed_set, target_function)

        if best_vertex is None:
            break

        previous_seed_set = current_seed_set
        current_seed_set = previous_seed_set | {best_vertex}

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
    seed_set_1 = _cost_seeds_greedy_single(adjacency=adjacency, costs=costs, budget=budget, target_function=_f1)
    seed_set_2 = _cost_seeds_greedy_single(adjacency=adjacency, costs=costs, budget=budget, target_function=_f2)
    seed_set_3 = _cost_seeds_greedy_single(adjacency=adjacency, costs=costs, budget=budget, target_function=_f3)

    candidate_sets = [seed_set_1, seed_set_2, seed_set_3]

    # Return the biggest seed set
    return max(candidate_sets, key=len)
