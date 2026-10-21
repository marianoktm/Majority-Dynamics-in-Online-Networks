import math
import snap
from networkx.classes import neighbors

from simulate_cascade import simulate_cascade

def _seed_set_cost(seed_set, costs):
    return sum(costs.get(node, 1) for node in seed_set)


def _get_adjacency_dictionary(graph):
    adjacency = {}

    for NI in graph.Nodes():
        adjacency[NI.GetId()] = [NI.GetNbrNId(i) for i in range(NI.GetDeg())]
    return adjacency


def _f1(adjacency, seed_set):
    total = 0.0

    for v, v_neighbors in adjacency.items():
        v_degree = len(v_neighbors)
        neighbors_seed_set_intersection_size = sum(u in seed_set for u in v_neighbors)
        total += min(neighbors_seed_set_intersection_size, math.ceil(v_degree / 2))

    return total


def _f2(adjacency, seed_set):
    total = 0.0

    for v, v_neighbors in adjacency.items():
        v_degree = len(v_neighbors)
        neighbors_seed_set_intersection_size = sum(u in seed_set for u in v_neighbors)

        for i in range(1, neighbors_seed_set_intersection_size + 1):
            total += max(math.ceil(v_degree / 2) - i + 1, 0)

    return total

def _f3(adjacency, seed_set):
    total = 0.0

    for v, v_neighbors in adjacency.items():
        v_degree = len(v_neighbors)
        neighbors_seed_set_intersection_size = sum(u in seed_set for u in v_neighbors)

        for i in range(1, neighbors_seed_set_intersection_size + 1):
            total += max((math.ceil(v_degree / 2) - i + 1) / (v_degree - i + 1), 0)

    return total


def _select_best_cost_effectiveness_vertex(adjacency, costs, current_seed_set, target_function):
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
    Determines the best maximal seed set using the Cost-Seeds-Greedy algorithm
    evaluated across the three target functions f1, f2, and f3.

    Parameters:
    - graph: An undirected SNAP graph (snap.TUNGraph).
    - costs: A dictionary mapping SNAP node IDs (integers) to integer costs.
    - budget: An integer representing the budget k.

    Returns:
    - A python set() containing node IDs representing the best seed set.
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
