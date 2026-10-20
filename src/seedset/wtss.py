import heapq
import math


def wtss2(graph, budget, costs):
    """
    Implements the WTSS algorithm to find a maximal seed set S
    such that c(S) <= budget, using SNAP.py.

    Parameters:
    - graph: snap.TNGraph or snap.TUNGraph, the social network graph
    - costs: dict, mapping node IDs (int) to their costs (int)
    - budget: int, the maximum allowed cost (k) for the seed set S

    Returns:
    - S: set, the selected seed set
    """
    S = set()
    U = set()

    delta = {}  # Remaining degree in U
    threshold = {}  # Current threshold k(v) = ceil(d(v)/2) initially
    N = {}  # Neighbor sets for each node

    # Step 1 & 2: Initialize graph structures and thresholds for all nodes in V
    for NI in graph.Nodes():
        v = NI.GetId()
        deg = NI.GetDeg()
        U.add(v)
        delta[v] = deg
        # Threshold t(v) = ceil(d(v)/2)
        threshold[v] = (deg + 1) // 2

        # Collect neighbors
        neighbors = set()
        for i in range(NI.GetDeg()):
            # Use GetNbrNId(i) for general graphs, or handle both directed/undirected if needed.
            # For TUNGraph/TNGraph, GetNbrNId(i) or GetOutNId(i) gets the neighbor ID at index i.
            try:
                n_id = NI.GetOutNId(i)
            except AttributeError:
                n_id = NI.GetNbrNId(i)
            neighbors.add(n_id)
        N[v] = neighbors

    def get_current_cost(seed_set):
        return sum(costs.get(v, 0) for v in seed_set)

    # Step 3: Loop while U is not empty
    while U:
        # --- Case 1: Find v in U such that k(v) == 0 ---
        case1_v = None
        for v in U:
            if threshold[v] == 0:
                case1_v = v
                break

        if case1_v is not None:
            v = case1_v
            S.add(v)
            for u in N[v]:
                if u in U:
                    threshold[u] = max(0, threshold[u] - 1)
            U.remove(v)
            continue

        # --- Case 2: Find v in U such that delta(v) < k(v) ---
        case2_v = None
        for v in U:
            if delta[v] < threshold[v]:
                case2_v = v
                break

        if case2_v is not None:
            v = case2_v
            # Check budget constraint
            if get_current_cost(S) + costs.get(v, 0) <= budget:
                S.add(v)
                for u in N[v]:
                    if u in U:
                        threshold[u] = max(0, threshold[u] - 1)
                U.remove(v)
                continue  # Success, move to next iteration
            else:
                # Budget exceeded for Case 2.
                # According to pseudo-code "else", we fall through to Case 3
                pass

        # --- Case 3: Greedy selection on remaining vertices in U ---
        best_v = None
        max_val = -1.0

        for u in U:
            d_u = delta[u]
            k_u = threshold[u]
            c_u = costs.get(u, 0)

            if d_u * (d_u + 1) == 0:
                val = 0.0
            else:
                # Formula: c(u) * k(u) / (delta(u) * (delta(u) + 1))
                val = (c_u * k_u) / float(d_u * (d_u + 1))

            if val > max_val:
                max_val = val
                best_v = u

        if best_v is not None:
            v = best_v
            # Remove v from the graph and update neighbor degrees/sets
            for u in N[v]:
                if u in U:
                    delta[u] = delta[u] - 1
                    if v in N[u]:
                        N[u].remove(v)
            U.remove(v)
        else:
            # Safety break if no node can be selected (should not happen with correct logic)
            break

    return S


def wtss(graph, costs, budget):
    """
    WTSS (Weighted Target Set Selection) on an undirected SNAP graph.

    graph  : snap.TUNGraph
    costs  : dict {node_id: int cost}
    budget : max total cost allowed for the seed set

    Returns seed_set = set() of node ids.
    """
    # ---- initialisation (lines 1-2) ----
    N = {}        # N(v): neighbors still in U
    delta = {}    # delta(v): current degree in the remaining graph
    k = {}        # k(v): remaining threshold

    for NI in graph.Nodes():
        v = NI.GetId()
        nbrs = {NI.GetNbrNId(i) for i in range(NI.GetDeg())}
        nbrs.discard(v)                          # ignore self-loops
        N[v] = nbrs
        delta[v] = len(nbrs)
        k[v] = math.ceil(delta[v] / 2)           # t(v) = ceil(d(v)/2)

    U = set(N)
    S = []                                       # seeds, in selection order

    case1 = {v for v in U if k[v] == 0}
    case2 = {v for v in U if delta[v] < k[v]}

    heap = []

    def push(v):
        # Case 3 score; delta(v) > 0 is guaranteed whenever v is
        # eligible for Case 3 (otherwise it would be Case 1 or Case 2).
        if delta[v] > 0:
            score = costs[v] * k[v] / (delta[v] * (delta[v] + 1))
            heapq.heappush(heap, (-score, v, delta[v], k[v]))

    for v in U:
        push(v)

    def refresh(u):
        """Re-classify u after its delta/k changed."""
        if k[u] == 0:
            case1.add(u)
            case2.discard(u)
        elif delta[u] < k[u]:
            case2.add(u)
        push(u)

    # ---- main loop (line 3) ----
    while U:
        if case1:
            # Case 1: v is activated by already-removed neighbors; free.
            v = case1.pop()
            if v not in U:
                continue
            influences = True
            dk = True            # lower neighbors' k (never below 0)
        elif case2:
            # Case 2: not enough neighbors left -> must be a seed.
            v = case2.pop()
            if v not in U or not (delta[v] < k[v]):
                continue
            S.append(v)
            influences = True
            dk = True
        else:
            # Case 3: greedy pick, assumed activated by neighbors in U.
            v = None
            while heap:
                _, u, d_u, k_u = heapq.heappop(heap)
                if u in U and delta[u] == d_u and k[u] == k_u:
                    v = u
                    break
            if v is None:        # safety net; should not happen
                break
            influences = False   # v is NOT active before its neighbors
            dk = False

        # ---- effect on neighbors (lines 8 / 14) ----
        if influences and dk:
            for u in N[v]:
                k[u] = max(0, k[u] - 1)

        # ---- remove v from the graph (lines 17-20) ----
        for u in N[v]:
            delta[u] -= 1
            N[u].discard(v)
            refresh(u)
        U.discard(v)
        case1.discard(v)
        case2.discard(v)

    # ---- enforce the budget ----
    seed_set = set()
    spent = 0
    for v in S:
        if spent + costs[v] > budget:
            break
        seed_set.add(v)
        spent += costs[v]

    return seed_set