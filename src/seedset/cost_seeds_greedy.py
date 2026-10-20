import heapq
import math
import snap

import src.simulate_cascade


def convert_snap_graph_to_python(graph):
    """
    Convert a SNAP undirected graph into Python adjacency and threshold maps.

    The conversion removes self-loops and represents each node's neighborhood
    as a Python set. Degrees and majority thresholds are computed from these
    cleaned neighborhoods, ensuring that all returned structures are
    consistent.

    Parameters
    ----------
    graph : snap.TUNGraph
        Undirected input graph.

    Returns
    -------
    adj : dict[int, set[int]]
        Mapping from each node ID to its set of neighbors, excluding itself.

    degrees : dict[int, int]
        Degree of each node, computed from the cleaned adjacency sets.

    thresh : dict[int, int]
        Majority activation threshold for each node, defined as
        ceil(degrees[v] / 2).

    Notes
    -----
    Isolated nodes are retained in the adjacency dictionary with empty
    neighborhood sets. The returned structures use ordinary Python objects,
    making repeated neighborhood operations independent of SNAP's iterators.
    """
    adj = {}

    # Extract the neighborhood of every node.
    for NI in graph.Nodes():
        v = NI.GetId()

        # Convert SNAP's neighbor iterator into a Python set.
        neighbors = {NI.GetNbrNId(i) for i in range(NI.GetDeg())}

        # A node must not count itself as an influenced neighbor.
        neighbors.discard(v)
        adj[v] = neighbors

    # Compute degrees after removing self-loops.
    degrees = {v: len(neighbors) for v, neighbors in adj.items()}

    # Cache the majority threshold used by the objective functions.
    thresh = {
        v: math.ceil(degree / 2)
        for v, degree in degrees.items()
    }

    return adj, degrees, thresh


# ---------------------------------------------------------------------
# Objective functions
# ---------------------------------------------------------------------

def f1(seed_set, adj, degrees, thresh):
    """
    Evaluate the first seed-selection objective.

    Each node receives one unit of contribution for every seeded neighbor,
    up to its majority threshold. Contributions from additional seeded
    neighbors beyond that threshold are ignored.

    Mathematically,

        f1(S) = sum_v min(|N(v) ∩ S|, thresh[v])

    where S is the seed set and N(v) is the neighborhood of node v.

    Parameters
    ----------
    seed_set : set[int]
        Currently selected seed nodes.
    adj : dict[int, set[int]]
        Node-to-neighbor mapping.
    degrees : dict[int, int]
        Degree of each node. Retained for a consistent objective signature.
    thresh : dict[int, int]
        Majority threshold of each node.

    Returns
    -------
    int
        Objective value for the given seed set.
    """
    return sum(
        min(len(adj[v] & seed_set), thresh[v])
        for v in adj
    )


def f2(seed_set, adj, degrees, thresh):
    """
    Evaluate the second seed-selection objective.

    The contribution of a node decreases with the number of its seeded
    neighbors. Its first seeded neighbor contributes thresh[v], its second
    contributes thresh[v] - 1, and so on, with negative contributions
    replaced by zero.

    Mathematically,

        f2(S) = sum_v sum_{i=1}^{|N(v) ∩ S|}
                    max(thresh[v] - i + 1, 0)

    Parameters
    ----------
    seed_set : set[int]
        Currently selected seed nodes.
    adj : dict[int, set[int]]
        Node-to-neighbor mapping.
    degrees : dict[int, int]
        Degree of each node. Retained for a consistent objective signature.
    thresh : dict[int, int]
        Majority threshold of each node.

    Returns
    -------
    int
        Objective value for the given seed set.
    """
    total = 0

    for v in adj:
        # Count how many neighbors of v have been selected as seeds.
        seeded_neighbors = len(adj[v] & seed_set)

        # Add the diminishing contribution of each seeded neighbor.
        for i in range(1, seeded_neighbors + 1):
            total += max(thresh[v] - i + 1, 0)

    return total


def f3(seed_set, adj, degrees, thresh):
    """
    Evaluate the third seed-selection objective.

    This objective uses a normalized contribution that accounts for both
    the majority threshold and the node's remaining unseeded neighbors.

    Mathematically,

        f3(S) = sum_v sum_{i=1}^{|N(v) ∩ S|}
                    max((thresh[v] - i + 1)
                        / (degrees[v] - i + 1), 0)

    For a node of degree d, the denominator of the i-th contribution is
    d - i + 1. As more neighbors are seeded, the remaining-neighbor count
    decreases.

    Parameters
    ----------
    seed_set : set[int]
        Currently selected seed nodes.
    adj : dict[int, set[int]]
        Node-to-neighbor mapping.
    degrees : dict[int, int]
        Degree of each node.
    thresh : dict[int, int]
        Majority threshold of each node.

    Returns
    -------
    float
        Objective value for the given seed set.

    Notes
    -----
    Isolated nodes have no neighbors, so their inner sum is empty and
    their contribution is zero. For every nonempty inner sum, the
    denominator is positive.
    """
    total = 0.0

    for v in adj:
        seeded_neighbors = len(adj[v] & seed_set)

        for i in range(1, seeded_neighbors + 1):
            numerator = thresh[v] - i + 1
            denominator = degrees[v] - i + 1

            total += max(numerator / denominator, 0.0)

    return total


# ---------------------------------------------------------------------
# Marginal-gain functions
# ---------------------------------------------------------------------

def gain_f1(c, d, h):
    """
    Compute the marginal contribution to f1 from one additional seeded
    neighbor.

    Parameters
    ----------
    c : int
        Number of neighbors of the affected node that are already seeded.
    d : int
        Degree of the affected node. Unused; retained for a common signature.
    h : int
        Majority threshold of the affected node.

    Returns
    -------
    float
        1.0 if c < h, otherwise 0.0.
    """
    return 1.0 if c < h else 0.0


def gain_f2(c, d, h):
    """
    Compute the marginal contribution to f2 from one additional seeded
    neighbor.

    The next contribution is h - c, provided it is positive.

    Parameters
    ----------
    c : int
        Number of neighbors of the affected node that are already seeded.
    d : int
        Degree of the affected node. Unused; retained for a common signature.
    h : int
        Majority threshold of the affected node.

    Returns
    -------
    float
        max(h - c, 0).
    """
    return float(max(h - c, 0))


def gain_f3(c, d, h):
    """
    Compute the marginal contribution to f3 from one additional seeded
    neighbor.

    The next contribution is normalized by the number of neighbors that
    remain unseeded after accounting for the existing c seeded neighbors.

    Parameters
    ----------
    c : int
        Number of neighbors of the affected node that are already seeded.
    d : int
        Degree of the affected node.
    h : int
        Majority threshold of the affected node.

    Returns
    -------
    float
        max((h - c) / (d - c), 0).

    Notes
    -----
    This function is called only for neighbors of a candidate node.
    Such a neighbor has d >= 1, and the maintained count satisfies c < d,
    so the denominator is positive.
    """
    return max((h - c) / (d - c), 0.0)


# ---------------------------------------------------------------------
# Cost-constrained greedy seed selection
# ---------------------------------------------------------------------

def cost_seeds_greedy(adj, deg, half, costs, budget, gain_fn):
    """
    Select seed nodes using a cost-sensitive greedy algorithm.

    At each iteration, the algorithm selects the remaining node with the
    highest ratio of marginal objective gain to node cost:

        ratio(u) = marginal_gain(u) / costs[u]

    Marginal gains are maintained incrementally. Since selecting node u
    directly changes the objective contributions of its neighbors, only
    those neighbors need to be considered when calculating its gain.

    A max-heap, implemented using negated priorities, supports lazy
    greedy evaluation. The stored priorities are upper bounds on current
    priorities when the marginal-gain function is nonincreasing as the
    seed set grows. A candidate's gain is recomputed only when it reaches
    the top of the heap.

    The budget is checked after a candidate is added. If its addition
    exceeds the budget, the algorithm returns the seed set from immediately
    before that addition.

    Parameters
    ----------
    adj : dict[int, set[int]]
        Node-to-neighbor mapping.
    deg : dict[int, int]
        Degree of each node.
    half : dict[int, int]
        Majority threshold of each node.
    costs : dict[int, int | float]
        Nonnegative cost associated with every node.
    budget : int | float
        Maximum permitted total cost of the returned seed set.
    gain_fn : callable
        Function gain_fn(c, d, h) that computes the marginal contribution
        from adding one more seeded neighbor to a node.

    Returns
    -------
    set[int]
        Selected seed nodes with total cost no greater than the budget.

    Notes
    -----
    A cost of zero is handled by replacing the denominator with 1e-9.
    This gives zero-cost nodes a large priority when their marginal gain
    is positive. Negative costs are not supported.

    If all remaining candidates have zero marginal gain, the algorithm
    stops early.
    """
    # cnt[v] tracks the number of neighbors of v that are currently seeds.
    # Updating these counters avoids repeatedly computing set intersections.
    cnt = {v: 0 for v in adj}

    # S_d is the current candidate solution.
    # S_p stores the solution immediately before the most recent addition.
    S_d = set()
    S_p = set()
    spent = 0

    def marginal(u):
        """Return the objective gain from adding u to the current seed set."""
        return sum(
            gain_fn(cnt[v], deg[v], half[v])
            for v in adj[u]
        )

    def ratio(u):
        """Return the current marginal gain per unit cost of node u."""
        return marginal(u) / max(costs[u], 1e-9)

    # Python's heapq is a min-heap, so negate priorities to simulate a max-heap. Node IDs provide a deterministic secondary ordering.
    heap = [(-ratio(u), u) for u in adj]
    heapq.heapify(heap)

    while heap:
        # Lazy evaluation recomputes the current priority of the best stored candidate.
        # If it no longer beats the next stored bound, reinsert it with its updated priority and inspect the new top.
        while heap:
            old_priority, u = heapq.heappop(heap)
            new_ratio = ratio(u)

            # The candidate is optimal among the remaining candidates if its current ratio is at least as high as every other stored upper bound, within floating-point tolerance.
            if not heap or -new_ratio <= heap[0][0] + 1e-12:
                break

            # Its priority decreased below another candidate's bound.
            heapq.heappush(heap, (-new_ratio, u))
        else:
            break

        # No positive marginal gain remains, so further selections cannot improve this objective.
        if new_ratio <= 0:
            break

        # Save the feasible solution before attempting the next addition.
        S_p = set(S_d)

        S_d.add(u)
        spent += costs[u]

        # Preserve the previous solution if this addition violates budget.
        if spent > budget:
            return S_p

        # The candidate was accepted. Update the affected neighbors' seeded-neighbor counts so subsequent gains use the current set.
        for v in adj[u]:
            cnt[v] += 1

    return S_d


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------

def cost_seeds_greedy_best(graph, costs, budget):
    """
    Run cost-sensitive greedy seed selection for all three objectives and
    return the seed set producing the largest majority-cascade activation.

    The algorithm independently runs the greedy procedure using f1, f2,
    and f3 as its optimization objective. Since these objectives have
    different scales, their numerical values are not compared directly.
    Instead, each resulting seed set is evaluated using the actual cascade
    simulation.

    Candidate solutions are ranked lexicographically by:

        (number of activated nodes, -total seed cost)

    Thus, the solution activating the most nodes is preferred. If multiple
    solutions activate the same number of nodes, the least expensive
    solution is returned.

    Parameters
    ----------
    graph : snap.TUNGraph
        Undirected input graph on which to run the cascade.
    costs : dict[int, int | float]
        Mapping from every node ID to its nonnegative selection cost.
    budget : int | float
        Maximum permitted total seed cost.

    Returns
    -------
    set[int]
        The best seed set among the three greedy solutions, evaluated
        by cascade size and then total seed cost.

    Raises
    ------
    KeyError
        If a graph node is missing from costs.

    Notes
    -----
    This function assumes that src.simulate_cascade.simulate_cascade
    accepts the SNAP graph and a seed set and returns the set of activated
    nodes. The cascade implementation determines the precise activation
    semantics.

    The greedy procedures optimize different surrogate objectives. Running
    all three and choosing the best cascade outcome is a heuristic; it does
    not guarantee a globally optimal seed set.
    """
    # Convert the graph once. All three greedy runs reuse the same structures.
    adj, deg, half = convert_snap_graph_to_python(graph)

    # Associate each surrogate objective with its corresponding marginal gain.
    gains_functions = {
        "f1": gain_f1,
        "f2": gain_f2,
        "f3": gain_f3,
    }

    # Initialize the best result so that even an empty seed set can win, for example when the budget is too small to select any useful node.
    best_set = set()
    best_key = (-1, 0)

    for name, gain_fn in gains_functions.items():
        # Generate a candidate seed set optimized for the selected objective.
        current_seed_set = cost_seeds_greedy(
            adj, deg, half, costs, budget, gain_fn
        )

        # Evaluate the candidate using the actual majority-cascade process.
        influence_set = src.simulate_cascade.simulate_cascade(
            graph=graph,
            seed_set=current_seed_set,
        )
        influence_set_size = len(influence_set)

        # Prefer larger cascades; break ties by choosing the cheaper seed set.
        total_cost = sum(costs[v] for v in current_seed_set)
        key = (influence_set_size, -total_cost)

        if key > best_key:
            best_key = key
            best_set = current_seed_set

    return set(best_set)