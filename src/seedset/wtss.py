import heapq
import math


def wtss(graph, costs, budget):
    # 1. Initialization
    adjacency = {} # adjacency[v] = neighbors of v [(N(v))]
    degree = {} # degree[v] = current degree of v [(d(v))]
    residual_threshold = {} # residual_threshold[v] = residual number of seeded neighbors required to seed v [k(v)]

    for NI in graph.Nodes():
        v = NI.GetId()
        nbrs = {NI.GetNbrNId(i) for i in range(NI.GetDeg())}
        nbrs.discard(v) # Ignore self-loops so a vertex cannot count itself as a neighbor.
        adjacency[v] = nbrs
        degree[v] = len(nbrs)
        residual_threshold[v] = math.ceil(degree[v] / 2)

    universe = set(adjacency) # vertices yet to be evaluated
    seed_set_candidates = [] # candidate vertices to be in the seed set

    # 2. Pre-evaluate case 1, case 2 and case 3 candidates
    # Case 1: If there exists v in U such as k(v) = 0
    case_1_vertices = {v for v in universe if residual_threshold[v] == 0}

    # Case 2: If there exist v in U such as d[v] < k[v]
    case_2_vertices = {v for v in universe if degree[v] < residual_threshold[v]}

    # Case 3: Fallback, will select a v that maximizes the cost-effectiveness.
    case_3_vertices_heap = [] # Handled greedily with a heap for performance reasons.

    # Push function for the heap
    def push(v):
        if degree[v] > 0:
            score = (costs[v] * residual_threshold[v] / (degree[v] * (degree[v] + 1)))
            heapq.heappush(case_3_vertices_heap,(-score, v, degree[v], residual_threshold[v]))
            return

    # Refresh function to update u's case
    def refresh(u):
        if residual_threshold[u] == 0:
            case_1_vertices.add(u)
            case_2_vertices.discard(u)
        elif degree[u] < residual_threshold[u]:
            case_2_vertices.add(u)
        push(u)

    for v in universe: push(v) # heap initialization

    # 3. Main loop
    while universe:
        # Case 1: Remove a vertex that needs no further active neighbors
        if case_1_vertices:
            v = case_1_vertices.pop()
            if v not in universe:
                continue
            # Treat v as activated. Its removal reduces the residual thresholds of its remaining neighbors.
            v_is_seeded = True
        # Case 2: Select a vertex that cannot meet its residual threshold
        elif case_2_vertices:
            v = case_2_vertices.pop()
            # Recheck the conditions because the classification sets may stale entries following earlier updates
            if v not in universe or not (degree[v] < residual_threshold[v]):
                continue
            seed_set_candidates.append(v) # This vertex must be selected as a seed.
            # Treat the selected seed as activated when updating neighbors.
            v_is_seeded = True
        # Case 3: Select the remaining vertex with the best greedy cost-effectiveness
        else:
            v = None
            while case_3_vertices_heap:
                _, u, stored_degree, stored_residual_threshold = heapq.heappop(case_3_vertices_heap)
                # Lazy invalidation accepts the entry only if the vertex is still present and its degree and threshold have notchanged since this entry was created.
                if u in universe and degree[u] == stored_degree and residual_threshold[u] == stored_residual_threshold:
                    v = u
                    break
            # Exit if the queue is exhausted.
            if v is None:
                break
            # Unlike Cases 1 and 2, this greedy removal does not assume that v is already activated by the remaining vertices.
            # Consequently, it does not reduce neighbors' thresholds.
            v_is_seeded = False

        # When v is treated as activated, each remaining neighbor u gains one active neighbor.
        # Its residual threshold therefore decreases by one.
        if v_is_seeded:
            for u in adjacency[v]:
                residual_threshold[u] = max(0, residual_threshold[u] - 1)

        # Delete v from each neighbor's adjacency set and decrement that neighbor's current degree.
        for u in adjacency[v]:
            degree[u] -= 1
            adjacency[u].discard(v)
            refresh(u)

        universe.discard(v)
        case_1_vertices.discard(v)
        case_2_vertices.discard(v)

    # 4. Enforce the budget
    seed_set = set()
    spent = 0
    for v in seed_set_candidates:
        # Continue if the current seed would exceed the available budget. Later seeds could be cheaper.
        if spent + costs[v] > budget:
            continue
        seed_set.add(v)
        spent += costs[v]

    return seed_set
