def wtss(graph, budget, costs):
    """
    Implements the WTSS algorithm to find a maximal seed set S
    such that c(S) <= budget, using SNAP.py.

    Parameters:
    - Graph: snap.TNGraph or snap.TUNGraph, the social network graph
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
        for n_id in NI.GetOutNId():
            neighbors.add(n_id)
        N[v] = neighbors

    def get_current_cost(seed_set):
        return sum(costs.get(v, 0) for v in seed_set)

    # Step 3: Loop while U is not empty
    while U:
        # Case 1: Check if there exists v in U such that threshold[v] == 0
        case1_v = None
        for v in U:
            if threshold[v] == 0:
                case1_v = v
                break

        if case1_v is not None:
            v = case1_v
            for u in N[v]:
                if u in U:
                    threshold[u] = max(0, threshold[u] - 1)
            U.remove(v)

        else:
            # Case 2: Check if there exists v in U such that delta[v] < threshold[v]
            case2_v = None
            for v in U:
                if delta[v] < threshold[v]:
                    case2_v = v
                    break

            if case2_v is not None:
                v = case2_v
                # Check budget constraint before adding to S
                potential_cost = get_current_cost(S) + costs.get(v, 0)
                if potential_cost <= budget:
                    S.add(v)
                else:
                    # Budget exceeded for maximal seed set, stop selection
                    break

                for u in N[v]:
                    if u in U:
                        threshold[u] = threshold[u] - 1
                U.remove(v)

            else:
                # Case 3: Greedy selection on remaining vertices in U
                if not U:
                    break

                best_v = None
                max_val = -1.0
                for u in U:
                    d_u = delta[u]
                    k_u = threshold[u]
                    c_u = costs.get(u, 0)

                    if d_u * (d_u + 1) == 0:
                        val = 0.0
                    else:
                        val = (c_u * k_u) / float(d_u * (d_u + 1))

                    if val > max_val:
                        max_val = val
                        best_v = u

                if best_v is None:
                    break

                v = best_v
                # Remove v from the graph and update neighbor degrees/sets
                for u in N[v]:
                    if u in U:
                        delta[u] = delta[u] - 1
                        if v in N[u]:
                            N[u].remove(v)
                U.remove(v)

    return S