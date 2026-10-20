import heapq
import math


def wtss(graph, costs, budget):
    """
    Compute a seed set using a greedy Weighted Target Set Selection (WTSS)
    heuristic on an undirected SNAP graph with majority thresholds.

    The algorithm progressively removes vertices from a working graph.
    During removal, it updates each remaining vertex's degree and residual
    activation threshold to determine whether that vertex:

      1. Can be activated without selecting it as a seed.
      2. Must be selected as a seed because too few neighbors remain.
      3. Can be selected using a greedy cost-based score.

    Vertices in Case 1 are removed without cost. Vertices in Case 2 are
    added to the seed-selection list. Vertices in Case 3 are selected
    greedily according to a score based on their cost, residual threshold,
    and current degree.

    After the removal process, the algorithm constructs a feasible seed
    set by accepting selected vertices in selection order until adding
    another vertex would exceed the budget.

    Parameters
    ----------
    graph : snap.TUNGraph
        Undirected input graph. Self-loops are ignored when constructing
        the working adjacency representation.

    costs : dict[int, int | float]
        Mapping from every node ID to its nonnegative selection cost.

    budget : int | float
        Maximum total cost permitted for the returned seed set.

    Returns
    -------
    set[int]
        The selected seed nodes, with total cost not exceeding budget.

    Notes
    -----
    The algorithm uses the majority threshold

        t(v) = ceil(d(v) / 2)

    where d(v) is the degree in the initial graph.

    During the algorithm, delta[v] tracks the degree of v in the remaining
    graph, while k[v] tracks its residual threshold after accounting for
    neighbors treated as already activated.

    The heap uses lazy invalidation: entries are accepted only if their
    stored degree and threshold still match the current values.

    The final budget-enforcement step truncates the selected sequence at
    the first vertex that would exceed the budget. It does not skip that
    vertex and continue considering later selections.

    This is a greedy heuristic; the returned set is budget-feasible, but
    global optimality is not guaranteed.
    """
    # 1. INITIALIZE THE WORKING GRAPH AND VERTEX THRESHOLDS

    N = {} # N[v] contains the neighbors of v that have not yet been removed.
    delta = {} # delta[v] is the current degree of v in the remaining graph.
    k = {} # k[v] is the residual number of active neighbors required by v. Initially, this is the majority threshold ceil(d(v) / 2).

    for NI in graph.Nodes():
        v = NI.GetId()

        # Convert SNAP's neighbor iterator into a Python set.
        nbrs = {NI.GetNbrNId(i) for i in range(NI.GetDeg())}

        # Ignore self-loops so a vertex cannot count itself as a neighbor.
        nbrs.discard(v)

        N[v] = nbrs
        delta[v] = len(nbrs)
        k[v] = math.ceil(delta[v] / 2)

    # U contains all vertices that have not yet been removed.
    U = set(N)

    # Store selected seeds in the order in which the algorithm selects them. Selection order is used later when enforcing the budget.
    S = []

    # 2. CLASSIFY VERTICES INTO SPECIAL CASES

    # Case 1: k[v] == 0.
    # The vertex requires no additional active neighbors and can be removed without selecting it as a seed.
    case1 = {v for v in U if k[v] == 0}

    # Case 2: delta[v] < k[v].
    # Even if every remaining neighbor were active, v could not reach its residual threshold. It must therefore be selected as a seed.
    case2 = {v for v in U if delta[v] < k[v]}

    # Remaining eligible vertices are handled greedily by Case 3. The heap stores negative scores because heapq is a min-heap.
    heap = []

    def push(v):
        """
        Insert a candidate into the Case 3 priority queue.

        The greedy score is

            score(v) = costs[v] * k[v] / (delta[v] * (delta[v] + 1))

        A lower score is preferred because the heap stores its negative.

        The current degree and residual threshold are stored with the entry.
        These values allow outdated entries to be detected when popped.

        Vertices with zero degree are not inserted. Under the intended
        classification, a zero-degree vertex belongs to Case 1 or Case 2.
        """
        if delta[v] > 0:
            score = (costs[v] * k[v] / (delta[v] * (delta[v] + 1)))
            heapq.heappush(heap,(-score, v, delta[v], k[v]))

    # Initially, all vertices are candidates for Case 3. Case 1 and Case 2 have priority over heap-based selection.
    for v in U:
        push(v)

    def refresh(u):
        """
        Update a vertex's classification and heap entry after its state
        changes.

        A vertex whose residual threshold reaches zero is placed in Case 1.
        Otherwise, if its current degree is below its residual threshold,
        it is placed in Case 2.

        A fresh heap entry is also inserted. Any older entries become stale
        and will be ignored when they are eventually popped.
        """
        if k[u] == 0:
            case1.add(u)
            case2.discard(u)
        elif delta[u] < k[u]:
            case2.add(u)
        push(u)

    # 3. ITERATIVELY REMOVE VERTICES

    while U:
        # Case 1: Remove a vertex that needs no further active neighbors
        if case1:
            v = case1.pop()

            # Ignore entries for vertices that have already been removed.
            if v not in U:
                continue

            # Treat v as activated. Its removal reduces the residual
            # thresholds of its remaining neighbors.
            influences = True
            dk = True


        # Case 2: Select a vertex that cannot meet its residual threshold
        elif case2:
            v = case2.pop()

            # Recheck the conditions because the classification sets may stale entries following earlier updates.
            if v not in U or not (delta[v] < k[v]):
                continue

            # This vertex must be selected as a seed.
            S.append(v)

            # Treat the selected seed as activated when updating neighbors.
            influences = True
            dk = True


        # Case 3: Select the remaining vertex with the best greedy score
        else:
            v = None

            while heap:
                _, u, stored_delta, stored_k = heapq.heappop(heap)

                # Lazy invalidation accepts the entry only if the vertex is still present and its degree and threshold have notchanged since this entry was created.
                if u in U and delta[u] == stored_delta and k[u] == stored_k:
                    v = u
                    break

            # Exit if the queue is exhausted.
            if v is None:
                break

            # Unlike Cases 1 and 2, this greedy removal does not assume that v is already activated by the remaining vertices.
            # Consequently, it does not reduce neighbors' thresholds.
            influences = False
            dk = False

        # 4. UPDATE RESIDUAL THRESHOLDS OF REMAINING NEIGHBORS

        # When v is treated as activated, each remaining neighbor u gains one active neighbor. Its residual threshold therefore decreases by one, but never becomes negative.
        if influences and dk:
            for u in N[v]:
                k[u] = max(0, k[u] - 1)

        # 5. REMOVE v AND UPDATE THE REMAINING GRAPH

        # Delete v from each neighbor's adjacency set and decrement that neighbor's current degree.
        for u in N[v]:
            delta[u] -= 1
            N[u].discard(v)

            # The updated degree and threshold may change u's case or score.
            refresh(u)

        # v has now been processed and must no longer be considered.
        U.discard(v)

        # Remove v from the special-case sets in case it was present.
        case1.discard(v)
        case2.discard(v)

    # 6. ENFORCE THE BUDGET

    # The removal procedure may select seeds whose combined cost exceeds the budget. Rebuild a feasible seed set in selection order.
    seed_set = set()
    spent = 0

    for v in S:
        # Stop at the first seed that would exceed the available budget. Later seeds are not considered, even if they are cheaper.
        if spent + costs[v] > budget:
            break

        seed_set.add(v)
        spent += costs[v]

    return seed_set
