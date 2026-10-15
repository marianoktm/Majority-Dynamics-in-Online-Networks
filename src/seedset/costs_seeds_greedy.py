import math
from simulate_cascade import simulate_cascade


def calculate_influence(G, seed_set, func_type="f1"):
    total_influence = 0
    for v in G.Nodes():
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


def costs_seed_strategy(G, budget, costs):

    types = ["f1", "f2", "f3"]
    best_seed_set = set()
    max_cascade_impact = -1
    best_strategy_name = ""

    for func_type in types:

        seed_set = set()
        current_cost = 0

        # generazione seed set per ogni funzione di influenza [f1, f2, f3]
        while True:
            best_node = None
            best_ratio = -1
            current_inf = calculate_influence(G, seed_set, func_type)

            for v in G.Nodes():
                node_id = v.GetId()
                if node_id not in seed_set:
                    seed_set.add(node_id)
                    new_inf = calculate_influence(G, seed_set, func_type)
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

        # vediamo quale delle 3 strategia ha attivato piu' nodi---
        final_active_nodes = simulate_cascade(G, seed_set)
        cascade_impact = len(final_active_nodes)  # Quante persone sono state attivate in totale

        # confronto e scelta del migliore
        if cascade_impact > max_cascade_impact:
            max_cascade_impact = cascade_impact
            best_seed_set = seed_set
            best_strategy_name = func_type

    print(f"Migliore Strategia basata su Diffusione del Seed Set: {best_strategy_name} (Persone attivate: {max_cascade_impact})")
    return best_seed_set