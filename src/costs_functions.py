"""
Node cost functions.

Provides different ways of assigning an integer cost to every node of a graph
(random, degree-based, average-neighbor-degree-based) and a helper that computes the
average cost. The returned dictionaries (node ID -> cost) are used by the
seed-selection algorithms to respect the budget.
"""

import math
import random


def random_cost(graph, min_val=1, max_val=10):
    """
    Function: random_cost

    Assigns each node a random integer cost.

    Args:
        graph (snap graph): Graph whose nodes receive a cost.
        min_val (int, optional): Minimum cost, inclusive. Default 1.
        max_val (int, optional): Maximum cost, inclusive. Default 10.

    Returns:
        dict: Maps node ID to a random integer cost in [min_val, max_val].
    """
    costs = {}
    for ni in graph.Nodes():
        costs[ni.GetId()] = random.randint(min_val, max_val)
    return costs


def degree_cost(graph):
    """
    Function: degree_cost

    Assigns each node a cost equal to half its degree, rounded up.

    Args:
        graph (snap graph): Graph whose nodes receive a cost.

    Returns:
        dict: Maps node ID to ceil(degree / 2).
    """
    costs = {}
    for ni in graph.Nodes():
        costs[ni.GetId()] = math.ceil(ni.GetDeg() / 2)
    return costs


def avg_neighbor_degree_cost(graph):
    """
    Function: avg_neighbor_degree_cost

    Assigns each node a cost equal to the average degree of its neighbors, rounded up, with
    a minimum of 1. Isolated nodes get cost 1.

    Args:
        graph (snap graph): Graph whose nodes receive a cost.

    Returns:
        dict: Maps node ID to max(1, ceil(average neighbor degree)).
    """
    costs = {}
    for ni in graph.Nodes():
        node_id = ni.GetId()
        deg = ni.GetDeg()

        # Handle isolated nodes (degree 0) to avoid division by zero
        if deg == 0:
            costs[node_id] = 1
            continue

        # Sum the degrees of all neighbors
        neighbor_degrees_sum = 0
        for i in range(deg):
            nbr_id = ni.GetNbrNId(i)
            nbr_node = graph.GetNI(nbr_id)
            neighbor_degrees_sum += nbr_node.GetDeg()

        # Compute average neighbor degree
        avg_neighbor_deg = neighbor_degrees_sum / deg

        # Assign integer cost (minimum cost of 1, rounded up)
        costs[node_id] = max(1, math.ceil(avg_neighbor_deg))

    return costs


def average_cost(costs):
    """
    Function: average_cost

    Computes the average of a cost dictionary, rounded up.

    Args:
        costs (dict): Maps node ID to cost.

    Returns:
        int: ceil(mean cost), or 0 if `costs` is empty.
    """
    if not costs:
        return 0
    return math.ceil(sum(costs.values()) / len(costs))
