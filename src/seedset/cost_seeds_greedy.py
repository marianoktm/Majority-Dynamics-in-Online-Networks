import heapq
import math

import src.simulate_cascade


def calculate_influence(graph, seed_set, func_type):
    total_influence = 0
    for v in graph.Nodes():
        node_id = v.GetId()
        neighbors = set(v.GetOutEdges())
        active_neighbors = neighbors.intersection(seed_set)
        num_active = len(active_neighbors)
        degree = len(neighbors)
        threshold = math.ceil(degree / 2)

        if func_type == "f1":
            total_influence += min(num_active, threshold)
        elif func_type == "f2":
            for i in range(1, num_active + 1):
                total_influence += max(threshold - i + 1, 0)
        elif func_type == "f3":
            for i in range(1, num_active + 1):
                total_influence += max((threshold - i + 1) / (degree - i + 1), 0)

    return total_influence


def cost_seeds_greedy_old(graph, costs, budget):
    types = ["f1", "f2", "f3"]
    best_seed_set = set()
    max_cascade_impact = -1
    best_strategy_name = ""

    print("Budget:", budget)

    for func_type in types:
        seed_set = set()
        current_cost = 0

        # Generate seed set for each influence function [f1, f2, f3]
        while True:
            best_node = None
            best_ratio = -1
            current_inf = calculate_influence(graph, seed_set, func_type)

            for v in graph.Nodes():
                node_id = v.GetId()
                if node_id not in seed_set:
                    seed_set.add(node_id)
                    new_inf = calculate_influence(graph, seed_set, func_type)
                    seed_set.remove(node_id)

                    marginal_gain = new_inf - current_inf

                    if current_cost + costs[node_id] <= budget:
                        cost_node = costs[node_id]
                        if cost_node > 0:
                            ratio = marginal_gain / cost_node
                            if ratio > best_ratio:
                                best_ratio = ratio
                                best_node = node_id

            if best_node is not None:
                seed_set.add(best_node)
                current_cost += costs.get(best_node, 1)
            else:
                break

        # Let's see which of the 3 strategies activated the most nodes...
        final_active_nodes = src.simulate_cascade.simulate_cascade(graph, seed_set)
        cascade_impact = len(final_active_nodes)  # Total nodes activated

        # Compare and select the best strategy
        if cascade_impact > max_cascade_impact:
            max_cascade_impact = cascade_impact
            best_seed_set = seed_set
            best_strategy_name = func_type

    print(f"Best Strategy based on Seed Set Diffusion: {best_strategy_name} (Activated nodes: {max_cascade_impact})")
    return best_seed_set





"""
Cost-Seeds-Greedy (Algorithm 1) for the cost majority cascade problem.

Given an undirected SNAP graph, a cost per node and a budget, the algorithm
greedily builds a seed set by repeatedly adding the node with the best
"marginal gain per unit cost", using one of three submodular functions
(f1, f2, f3). The three resulting seed sets are compared by simulating the
majority cascade (threshold ceil(d(v)/2)) and the best one is returned.
"""
import heapq
import math

import snap  # noqa: F401  (the input graph is a snap.TUNGraph; kept for clarity)


def _build_adj(graph):
    """
    Convert a SNAP graph into plain Python structures.

    Parameters
    ----------
    graph : snap.TUNGraph
        Undirected input graph.

    Returns
    -------
    adj : dict[int, set[int]]
        Neighbor set of every node (self-loops removed).
    deg : dict[int, int]
        Degree d(v) of every node, computed on the cleaned adjacency.
    half : dict[int, int]
        Majority threshold ceil(d(v) / 2) of every node.
    """
    adj = {}
    for NI in graph.Nodes():
        v = NI.GetId()
        nb = {NI.GetNbrNId(i) for i in range(NI.GetDeg())}
        # A self-loop would make a node "influence itself" and inflate its
        # degree, which has no meaning in the cascade, so we drop it.
        nb.discard(v)
        adj[v] = nb
    # Degree is computed from the cleaned sets (not NI.GetDeg()) so that
    # deg, half and adj are always consistent with each other.
    deg = {v: len(nb) for v, nb in adj.items()}
    # Threshold is cached because it is used constantly in the greedy loop.
    half = {v: math.ceil(d / 2) for v, d in deg.items()}   # ceil(d(v)/2)
    return adj, deg, half


# ---------------------------------------------------------------
# f1, f2, f3
# ---------------------------------------------------------------
def f1(S, adj, deg, half):
    """
    Objective f1(S) = sum_v min(|N(v) ∩ S|, ceil(d(v)/2)).

    Each node counts one point per seeded neighbor, but only up to its
    threshold: neighbors beyond the threshold are "wasted" and do not help.

    Parameters
    ----------
    S : set[int]
        Current seed set.
    adj, deg, half : see `_build_adj`.

    Returns
    -------
    int
        Value of f1 on S.
    """
    return sum(min(len(adj[v] & S), half[v]) for v in adj)


def f2(S, adj, deg, half):
    """
    Objective f2(S) = sum_v sum_{i=1}^{|N(v) ∩ S|} max(ceil(d(v)/2) - i + 1, 0).

    Like f1, but the i-th seeded neighbor of v is worth (threshold - i + 1)
    instead of 1, so early neighbors count more than later ones.

    Parameters
    ----------
    S : set[int]
        Current seed set.
    adj, deg, half : see `_build_adj`.

    Returns
    -------
    int
        Value of f2 on S.
    """
    total = 0
    for v in adj:
        for i in range(1, len(adj[v] & S) + 1):
            # max(..., 0) stops the contribution once i exceeds the
            # threshold; without it the terms would become negative.
            total += max(half[v] - i + 1, 0)
    return total


def f3(S, adj, deg, half):
    """
    Objective f3(S) = sum_v sum_{i=1}^{|N(v) ∩ S|}
                      max((ceil(d(v)/2) - i + 1) / (d(v) - i + 1), 0).

    Same shape as f2, but each term is normalised by the number of
    neighbors of v that are still not seeded (d(v) - i + 1). Low-degree
    nodes are therefore "closer" to being activated per seeded neighbor.

    Parameters
    ----------
    S : set[int]
        Current seed set.
    adj, deg, half : see `_build_adj`.

    Returns
    -------
    float
        Value of f3 on S.
    """
    total = 0
    for v in adj:
        for i in range(1, len(adj[v] & S) + 1):
            # i <= |N(v) ∩ S| <= d(v), so the denominator d(v) - i + 1 >= 1:
            # no division by zero is possible here.
            total += max((half[v] - i + 1) / (deg[v] - i + 1), 0)
    return total


# Marginal gains. Adding u to S only changes the terms of u's NEIGHBORS: for a
# neighbor v that already has c seeded neighbors, the new term is the one with
# i = c + 1 in the inner sum. Computing only this term is much faster than
# evaluating f(S ∪ {u}) - f(S) from scratch over the whole graph.
def _gain_f1(c, d, h):
    """
    Marginal gain of one more seeded neighbor for f1.

    Parameters
    ----------
    c : int  Seeded neighbors the node already has, |N(v) ∩ S|.
    d : int  Degree of the node (unused, kept for a uniform signature).
    h : int  Threshold ceil(d/2) of the node.

    Returns
    -------
    float
        1.0 while the node is below its threshold, else 0.0.
    """
    return 1.0 if c < h else 0.0


def _gain_f2(c, d, h):
    """
    Marginal gain of one more seeded neighbor for f2 (term i = c + 1).

    Parameters
    ----------
    c, d, h : see `_gain_f1`.

    Returns
    -------
    float
        max(h - c, 0).
    """
    return float(max(h - c, 0))


def _gain_f3(c, d, h):
    """
    Marginal gain of one more seeded neighbor for f3 (term i = c + 1).

    Parameters
    ----------
    c, d, h : see `_gain_f1`.

    Returns
    -------
    float
        max((h - c) / (d - c), 0).
    """
    # The gain is evaluated for a neighbor v of an UNSEEDED node u, so u is
    # one of v's neighbors that is not seeded yet: c <= d - 1 and the
    # denominator d - c is at least 1.
    return max((h - c) / (d - c), 0.0)


# Lookup tables so the entry point can loop over the three functions by name.
F_FUNCS = {"f1": f1, "f2": f2, "f3": f3}
GAINS = {"f1": _gain_f1, "f2": _gain_f2, "f3": _gain_f3}


# ---------------------------------------------------------------
# Cost-Seeds-Greedy, lazy-greedy implementation
# ---------------------------------------------------------------
def cost_seeds_greedy(adj, deg, half, costs, budget, gain_fn):
    """
    Cost-Seeds-Greedy for one objective function.

    At each step add the node u maximising  marginal_gain(u) / cost(u).
    Stop as soon as the total cost exceeds the budget and return the set
    from just before that last node was added (S_p in the pseudocode).

    Parameters
    ----------
    adj, deg, half : see `_build_adj`.
    costs : dict[int, int]
        Cost of every node.
    budget : int
        Maximum total cost of the seed set.
    gain_fn : callable(c, d, h) -> float
        Marginal gain function of the chosen objective (_gain_f1/2/3).

    Returns
    -------
    set[int]
        Seed set with total cost <= budget.
    """
    # cnt[v] = |N(v) ∩ S_d|, maintained incrementally so that marginal gains
    # can be computed without rebuilding intersections every time.
    cnt = {v: 0 for v in adj}
    S_d = set()
    S_p = set()
    spent = 0

    def marginal(u):
        """Marginal gain f(S_d ∪ {u}) - f(S_d): sum over u's neighbors."""
        return sum(gain_fn(cnt[v], deg[v], half[v]) for v in adj[u])

    def ratio(u):
        """Gain per unit cost, the quantity maximised at line 3."""
        # max(..., 1e-9) protects against a zero cost (division by zero)
        # while still making free nodes extremely attractive.
        return marginal(u) / max(costs[u], 1e-9)

    # Max-heap simulated with negated keys (heapq is a min-heap).
    heap = [(-ratio(u), u) for u in adj]
    heapq.heapify(heap)

    while heap:
        # Lazy evaluation. The objectives are submodular, so a node's gain
        # can only DECREASE as S_d grows: every ratio stored in the heap is
        # an upper bound of the true current ratio. We therefore recompute
        # only the top node; if it still beats the best stored bound of all
        # the others it is guaranteed to be the true argmax, and we avoid
        # re-evaluating every candidate at every step.
        while heap:
            old, u = heapq.heappop(heap)
            new = ratio(u)
            # The small tolerance avoids infinite re-pushing caused by
            # floating-point noise when two ratios are (almost) equal.
            if not heap or -new <= heap[0][0] + 1e-12:
                break
            # Its real value dropped below another candidate's bound:
            # put it back with the fresh value and look at the new top.
            heapq.heappush(heap, (-new, u))
        else:
            break

        # If the best remaining ratio is 0 the objective is saturated:
        # adding more nodes would only waste budget, so we stop early.
        # (Small deviation from the pseudocode.)
        if new <= 0:
            break

        S_p = set(S_d)                      # S_p = S_d
        S_d.add(u)                          # S_d = S_p ∪ {u}
        spent += costs[u]
        # The pseudocode adds the node first and checks the budget after
        # ("until c(S_d) > k"), so the node that breaks the budget is
        # discarded by returning the previous set S_p.
        if spent > budget:
            return S_p
        # Update the counters only now: they must reflect S_d only when the
        # node is actually kept.
        for v in adj[u]:
            cnt[v] += 1

    return S_d                              # budget never exceeded


# ---------------------------------------------------------------
# Majority cascade simulation (threshold ceil(d/2)), used to pick the best
# ---------------------------------------------------------------
def cascade_size(seed_set, adj, half):
    """
    Simulate the majority cascade and count the activated nodes.

    A node becomes active when at least ceil(d(v)/2) of its neighbors are
    active. Seeds are active from the start.

    Parameters
    ----------
    seed_set : set[int]
        Initially active nodes.
    adj, half : see `_build_adj`.

    Returns
    -------
    int
        Number of active nodes when the cascade stops.
    """
    active = set(seed_set)
    # Isolated nodes have threshold 0: they are active without any help.
    active.update(v for v in adj if half[v] == 0)
    # Queue-based propagation: each node is processed once, when it becomes
    # active, so the cost is O(n + m) instead of re-scanning the whole graph.
    queue = list(active)
    cnt = {v: 0 for v in adj}               # active neighbors of each node
    while queue:
        u = queue.pop()
        for v in adj[u]:
            if v in active:
                continue
            cnt[v] += 1
            if cnt[v] >= half[v]:
                active.add(v)
                queue.append(v)
    return len(active)


# ---------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------
def cost_seeds_greedy_best(graph, costs, budget):
    """
    Run Cost-Seeds-Greedy with f1, f2 and f3 and return the best seed set.

    "Best" means the set that activates the most nodes in the majority
    cascade; ties are broken in favour of the cheaper set.

    Parameters
    ----------
    graph : snap.TUNGraph
        Undirected input graph.
    costs : dict[int, int]
        Mapping node id -> integer cost (must contain every node).
    budget : int
        Maximum total cost of the seed set.

    Returns
    -------
    set[int]
        The selected seed set.
    """
    adj, deg, half = _build_adj(graph)

    # The key is (nodes activated, -cost): tuples compare element by element,
    # so more activated nodes wins and, on a tie, the lower cost wins.
    # We compare with the real cascade because the values of f1, f2 and f3
    # live on different scales and cannot be compared with each other.
    # best_key starts at -1 so that even the empty set (0 activated nodes,
    # e.g. a budget too small for any seed) can replace it.
    best_set, best_key = set(), (-1, 0)
    for name in ("f1", "f2", "f3"):
        S = cost_seeds_greedy(adj, deg, half, costs, budget, GAINS[name])
        key = (cascade_size(S, adj, half), -sum(costs[v] for v in S))
        if key > best_key:
            best_key, best_set = key, S

    seed_set = set(best_set)
    return seed_set