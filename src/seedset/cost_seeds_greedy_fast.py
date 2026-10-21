import math
import snap


def _seed_set_cost(seed_set, costs):
    return sum(costs.get(node, 1) for node in seed_set)


def _get_adjacency_dictionary(graph):
    adjacency = {}

    for NI in graph.Nodes():
        adjacency[NI.GetId()] = [NI.GetNbrNId(i) for i in range(NI.GetDeg())]
    return adjacency


def _half_degree_threshold(adjacency, node):
    """How many seeded neighbors a node needs to be 'won over': ceil(degree / 2)."""
    node_degree = len(adjacency[node])
    return math.ceil(node_degree / 2)


def _marginal_gain_f1(adjacency, seeded_neighbor_count, candidate_vertex):
    """
    f1: a neighbor gains +1 only if it has not yet reached its half-degree threshold.
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
    f2: a neighbor gains (threshold - current_count), so early seeded neighbors
    are worth the most and the value shrinks to 0 at the threshold.
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
    f3: like f2, but divided by how many of the neighbor's own neighbors
    are still not seeded: (threshold - count) / (degree - count).
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
    # initialize seeded neighbors count for each vertex
    seeded_neighbor_count = {nodes: 0 for nodes in adjacency}

    previous_seed_set = set()
    current_seed_set = set()

    while _seed_set_cost(current_seed_set, costs) < budget:
        best_vertex = _select_best_cost_effectiveness_vertex(adjacency, costs, current_seed_set, seeded_neighbor_count, target_function)

        if best_vertex is None:
            break

        previous_seed_set = current_seed_set
        current_seed_set = previous_seed_set | {best_vertex}

        for neighbor in adjacency[best_vertex]:
            seeded_neighbor_count[neighbor] += 1

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
    seed_set_1 = _cost_seeds_greedy_single(adjacency=adjacency, costs=costs, budget=budget, target_function=_marginal_gain_f1)
    seed_set_2 = _cost_seeds_greedy_single(adjacency=adjacency, costs=costs, budget=budget, target_function=_marginal_gain_f2)
    seed_set_3 = _cost_seeds_greedy_single(adjacency=adjacency, costs=costs, budget=budget, target_function=_marginal_gain_f3)

    candidate_sets = [seed_set_1, seed_set_2, seed_set_3]

    # Return the biggest seed set
    return max(candidate_sets, key=len)