"""
Cost-Seeds-Greedy, lazy implementation.

Greedy seed selection using target functions f1, f2 and f3 with the majority
(half-degree) threshold. Builds on the fast version by keeping the candidates in a
priority queue and re-evaluating a score only when it is stale (lazy evaluation),
exploiting the fact that marginal gains never increase as the seed set grows.
"""

import heapq
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
    # Total price of a group of seeds. Nodes with no listed cost default to 1.
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
    # Convert the SNAP graph into a plain {node: [neighbors]} dictionary.
    # Lookups are much faster than calling into SNAP repeatedly.
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


def _get_threshold_dictionary(adjacency):
    """
    Function: _get_threshold_dictionary

    Precomputes the half-degree threshold of every node once, since thresholds never change.

    Args:
        adjacency (dict): Maps node ID to its list of neighbors.

    Returns:
        dict: Maps node ID to ceil(degree / 2).
    """
    # The threshold never changes, so compute it once up front instead of recomputing len() and ceil() every time a gain is evaluated.
    return {node: _half_degree_threshold(adjacency, node) for node in adjacency}


# The three marginal gain functions below all answer the same question:
# "if I seed candidate_vertex right now, how much progress do I make?"
# Seeding a vertex only helps its NEIGHBORS (it moves them closer to their threshold),
# so each function loops over the candidate's neighbors and adds up their gains.
# seeded_neighbor_count[x] = "how many of x's neighbors are already seeded".
def _marginal_gain_f1(adjacency, thresholds, seeded_neighbor_count, candidate_vertex):
    """
    Function: _marginal_gain_f1

    Marginal gain for target function f1. Each neighbor of the candidate that has not yet
    reached its half-degree threshold contributes +1.

    Args:
        adjacency (dict): Maps node ID to its list of neighbors.
        thresholds (dict): Precomputed half-degree threshold of each node.
        seeded_neighbor_count (dict): Maps node ID to how many of its neighbors are currently seeded.
        candidate_vertex (int): ID of the node whose marginal gain is evaluated.

    Returns:
        int or float: Marginal gain obtained by seeding `candidate_vertex` now.
    """
    total_gain = 0

    # Each neighbor who is still not seeded gives 1 point.
    # Neighbors who are already seeded give nothing (no point "convincing" them twice).
    for neighbor in adjacency[candidate_vertex]:
        threshold = thresholds[neighbor]
        current_count = seeded_neighbor_count[neighbor]

        if current_count < threshold:
            total_gain += 1

    return total_gain


def _marginal_gain_f2(adjacency, thresholds, seeded_neighbor_count, candidate_vertex):
    """
    Function: _marginal_gain_f2

    Marginal gain for target function f2. Each neighbor contributes (threshold - current
    seeded count) while this is positive, so early seeded neighbors are worth the most.

    Args:
        adjacency (dict): Maps node ID to its list of neighbors.
        thresholds (dict): Precomputed half-degree threshold of each node.
        seeded_neighbor_count (dict): Maps node ID to how many of its neighbors are currently seeded.
        candidate_vertex (int): ID of the node whose marginal gain is evaluated.

    Returns:
        int or float: Marginal gain obtained by seeding `candidate_vertex` now.
    """
    total_gain = 0

    # A neighbor who is far from being seeded is worth a LOT of points,
    # A neighbor who is almost seeded is worth only a few, and a seeded neighbor is worth zero.
    for neighbor in adjacency[candidate_vertex]:
        threshold = thresholds[neighbor]
        current_count = seeded_neighbor_count[neighbor]

        remaining_to_threshold = threshold - current_count
        if remaining_to_threshold > 0:
            total_gain += remaining_to_threshold

    return total_gain


def _marginal_gain_f3(adjacency, thresholds, seeded_neighbor_count, candidate_vertex):
    """
    Function: _marginal_gain_f3

    Marginal gain for target function f3. Like f2, but each neighbor's contribution is
    divided by its number of still-unseeded neighbors: (threshold - count) / (degree -
    count).

    Args:
        adjacency (dict): Maps node ID to its list of neighbors.
        thresholds (dict): Precomputed half-degree threshold of each node.
        seeded_neighbor_count (dict): Maps node ID to how many of its neighbors are currently seeded.
        candidate_vertex (int): ID of the node whose marginal gain is evaluated.

    Returns:
        int or float: Marginal gain obtained by seeding `candidate_vertex` now.
    """
    total_gain = 0.0
    # Same as f2, but we also ask "how many of that neighbor's neighbors are still
    # available to help?" If few are left, each remaining one matters more, so the
    # points are shared out per available neighbor. This number always stays <= 1 per
    # neighbor and only ever shrinks as more vertices are seeded.

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
    Function: _build_cost_effectiveness_heap

    Builds the initial max-heap (via negated keys) of cost-effectiveness scores for all
    vertices. Entries are (-cost_effectiveness, order, vertex, evaluated_at); `order` breaks
    ties like a linear scan, and `evaluated_at` is the seed set size when the score was
    computed (used to detect stale entries).

    Args:
        adjacency (dict): Maps node ID to its list of neighbors.
        costs (dict): Maps node ID to cost (default 1).
        thresholds (dict): Half-degree threshold of each node.
        seeded_neighbor_count (dict): Maps node ID to its number of seeded neighbors.
        target_function (callable): Marginal gain function (f1, f2 or f3).

    Returns:
        list: Heapified list of tuples (-cost_effectiveness, order, vertex, evaluated_at).
    """
    heap = []

    # Idea:
    # Build the initial heap.
    # Everyone gets a score of cost-effectiveness, and the best-scoring vertex sits at the top.
    #
    # order: If two vertices have the exact same score, the tuple comparison falls back to 'order', so the one that came first in adjacency wins.
    #
    # evaluated_at': It's a timestamp recording the number of seeded vertices when the related score was computed.
    # Its purpose is to allow "stale" entries to be detected.
    # If a cost-effectiveness score was computed before the latest seed was added to the seed set, it may no longer be accurate.
    for order, candidate_vertex in enumerate(adjacency):
        marginal_gain = target_function(adjacency, thresholds, seeded_neighbor_count, candidate_vertex)
        cost_effectiveness = marginal_gain / costs.get(candidate_vertex, 1)

        heap.append((-cost_effectiveness, order, candidate_vertex, 0))

    # Turn the plain list into a heap.
    heapq.heapify(heap)
    return heap


def _select_best_cost_effectiveness_vertex(adjacency, costs, thresholds, heap, current_seed_size, seeded_neighbor_count,
                                           target_function):
    """
    Function: _select_best_cost_effectiveness_vertex

    Lazy greedy selection. Stale scores are upper bounds of the true values, so the top of
    the heap is popped: if its score is fresh it is the best vertex; otherwise it is
    recomputed and pushed back.

    Args:
        adjacency (dict): Maps node ID to its list of neighbors.
        costs (dict): Maps node ID to cost (default 1).
        thresholds (dict): Half-degree threshold of each node.
        heap (list): Heap built by _build_cost_effectiveness_heap (modified in place).
        current_seed_size (int): Current number of seeds, used as the freshness timestamp.
        seeded_neighbor_count (dict): Maps node ID to its number of seeded neighbors.
        target_function (callable): Marginal gain function (f1, f2 or f3).

    Returns:
        int or None: Best vertex ID (already removed from the heap), or None if the heap is empty.
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
        set: Seed set as it was before the last vertex was added (the last set built by the loop before the budget was reached).
    """
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
        best_vertex = _select_best_cost_effectiveness_vertex(adjacency, costs, thresholds, heap, len(current_seed_set),
                                                             seeded_neighbor_count, target_function)

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
