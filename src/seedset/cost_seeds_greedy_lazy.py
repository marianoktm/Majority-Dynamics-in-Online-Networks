import heapq
import math

import snap

from simulate_cascade import simulate_cascade

def _seed_set_cost(seed_set, costs):
    # Total price of a group of seeds. Nodes with no listed cost default to 1.
    return sum(costs.get(node, 1) for node in seed_set)


def _get_adjacency_dictionary(graph):
    # Convert the SNAP graph into a plain {node: [neighbors]} dictionary.
    # Lookups are much faster than calling into SNAP repeatedly.
    adjacency = {}

    for NI in graph.Nodes():
        adjacency[NI.GetId()] = [NI.GetNbrNId(i) for i in range(NI.GetDeg())]
    return adjacency


def _half_degree_threshold(adjacency, node):
    """How many seeded neighbors a node needs to be 'won over': ceil(degree / 2)."""
    node_degree = len(adjacency[node])
    return math.ceil(node_degree / 2)


def _get_threshold_dictionary(adjacency):
    """Precompute the half-degree threshold of every node once."""
    # The threshold never changes, so compute it once up front instead of recomputing len() and ceil() every time a gain is evaluated.
    return {node: _half_degree_threshold(adjacency, node) for node in adjacency}


# The three marginal gain functions below all answer the same question:
# "if I seed candidate_vertex right now, how much progress do I make?"
# Seeding a vertex only helps its NEIGHBORS (it moves them closer to their threshold),
# so each function loops over the candidate's neighbors and adds up their gains.
# seeded_neighbor_count[x] = "how many of x's neighbors are already seeded".
def _marginal_gain_f1(adjacency, thresholds, seeded_neighbor_count, candidate_vertex):
    """
    f1: a neighbor gains +1 only if it has not yet reached its half-degree threshold.
    """
    # Each neighbor who is still not seeded gives 1 point.
    # Neighbors who are already seeded give nothing (no point convincing them twice).
    total_gain = 0

    for neighbor in adjacency[candidate_vertex]:
        threshold = thresholds[neighbor]
        current_count = seeded_neighbor_count[neighbor]

        if current_count < threshold:
            total_gain += 1

    return total_gain


def _marginal_gain_f2(adjacency, thresholds, seeded_neighbor_count, candidate_vertex):
    """
    f2: a neighbor gains (threshold - current_count), so early seeded neighbors
    are worth the most and the value shrinks to 0 at the threshold.
    """
    # A neighbor who is far from being seeded is worth a LOT of points,
    # A neighbor who is almost seeded is worth only a few, and a seeded neighbor is worth zero.
    total_gain = 0

    for neighbor in adjacency[candidate_vertex]:
        threshold = thresholds[neighbor]
        current_count = seeded_neighbor_count[neighbor]

        remaining_to_threshold = threshold - current_count
        if remaining_to_threshold > 0:
            total_gain += remaining_to_threshold

    return total_gain


def _marginal_gain_f3(adjacency, thresholds, seeded_neighbor_count, candidate_vertex):
    """
    f3: like f2, but divided by how many of the neighbor's own neighbors
    are still not seeded: (threshold - count) / (degree - count).
    """
    # Same as f2, but we also ask "how many of that neighbor's neighbors are still
    # available to help?" If few are left, each remaining one matters more, so the
    # points are shared out per available neighbor. This number always stays <= 1 per
    # neighbor and only ever shrinks as more vertices are seeded.
    total_gain = 0.0

    for neighbor in adjacency[candidate_vertex]:
        neighbor_degree = len(adjacency[neighbor])
        threshold = thresholds[neighbor]
        current_count = seeded_neighbor_count[neighbor]

        remaining_to_threshold = threshold - current_count
        unseeded_neighbors = neighbor_degree - current_count

        # Skip if already past the threshold, or if the division would be invalid
        if remaining_to_threshold > 0 and unseeded_neighbors > 0:
            total_gain += remaining_to_threshold / unseeded_neighbors

    return total_gain


def _build_cost_effectiveness_heap(adjacency, costs, thresholds, seeded_neighbor_count, target_function):
    """
    Max-heap (via negated keys) of (-cost_effectiveness, order, vertex, evaluated_at).
    'order' is the position in adjacency, so ties break exactly like a linear scan
    that keeps the first strictly-best vertex. 'evaluated_at' is the seed set size
    at which the entry was computed.
    """
    # Idea:
    # Build the initial heap.
    # Everyone gets a score of cost-effectiveness, and the best-scoring vertex sits at the top.
    #
    # order: If two vertices have the exact same score, the tuple comparison falls back to 'order', so the one that came first in adjacency wins.
    #
    # evaluated_at': It's a timestamp recording the number of seeded vertices when the related score was computed.
    # Its purpose is to allow "stale" entries to be detected.
    # If a cost-effectiveness score was computed before the latest seed was added to the seed set, it may no longer be accurate.
    heap = []

    for order, candidate_vertex in enumerate(adjacency):
        marginal_gain = target_function(adjacency, thresholds, seeded_neighbor_count, candidate_vertex)
        cost_effectiveness = marginal_gain / costs.get(candidate_vertex, 1)

        heap.append((-cost_effectiveness, order, candidate_vertex, 0))

    # Turn the plain list into a heap.
    heapq.heapify(heap)
    return heap


def _select_best_cost_effectiveness_vertex(adjacency, costs, thresholds, heap, current_seed_size, seeded_neighbor_count, target_function):
    """
    Lazy greedy selection. Marginal gains never increase as the seed set grows,
    so a stale heap key is an upper bound on the true value. Pop the top; if it was
    computed for the current seed set it is the true best, otherwise recompute it
    and push it back.
    """
    # Idea:
    # Look at whoever is on top of the heap.
    # If their score is from RIGHT NOW (fresh), nobody below can beat them,
    # because everyone else's real score is at most their (old, optimistic) score.
    # Therefore, pick them.
    # If their score is OLD (stale), it might be too high. Re-score just this
    # vertex, put them back in the right spot, and look at the top again.
    # Usually only a few vertices need re-scoring each round instead of everyone.
    while heap:
        _, order, candidate_vertex, evaluated_at = heapq.heappop(heap)

        # Fresh score: it was computed with the current number of seeds, so it is exact.
        # The vertex stays popped, which is what we want: it is about to become a seed
        # and must never be a candidate again.
        if evaluated_at == current_seed_size:
            return candidate_vertex

        # Stale score: recompute with the current seeded_neighbor_count and reinsert.
        marginal_gain = target_function(adjacency, thresholds, seeded_neighbor_count, candidate_vertex)
        cost_effectiveness = marginal_gain / costs.get(candidate_vertex, 1)

        heapq.heappush(heap, (-cost_effectiveness, order, candidate_vertex, current_seed_size))

    # Heap is empty: every vertex has already been seeded.
    return None


def _cost_seeds_greedy_single(adjacency, costs, budget, target_function):
    # Runs the greedy algorithm for ONE choice of gain function (f1, f2 or f3).

    # Initialize seeded neighbors count for each vertex (nobody is seeded yet, so every vertex has 0 convinced neighbors)
    seeded_neighbor_count = {nodes: 0 for nodes in adjacency}
    thresholds = _get_threshold_dictionary(adjacency)

    # Build the heap once. After this we only refresh entries lazily.
    heap = _build_cost_effectiveness_heap(adjacency, costs, thresholds, seeded_neighbor_count, target_function)

    # We track two sets because the loop stops AFTER the budget is reached or exceeded.
    # 'previous_seed_set' is the set from before the last pick, i.e. the last one that did not overshoot.
    previous_seed_set = set()
    current_seed_set = set()

    # Keep "buying" seeds until we have spent the whole budget.
    while _seed_set_cost(current_seed_set, costs) < budget:
        # current_seed_size is the "timestamp" used to decide whether a heap entry is fresh.
        best_vertex = _select_best_cost_effectiveness_vertex(adjacency, costs, thresholds, heap, len(current_seed_set), seeded_neighbor_count, target_function)

        # Nobody left to pick.
        if best_vertex is None:
            break

        previous_seed_set = current_seed_set
        current_seed_set = previous_seed_set | {best_vertex}

        # Seeding best_vertex means every one of its neighbors now has one more seeded friend.
        # This is the only thing that changes scores, and it is why all old scores become upper bounds.
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