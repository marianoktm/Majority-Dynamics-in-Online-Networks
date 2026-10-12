import math

#########################################
##### ALGORITHMS 1 #####
#########################################

# --- PARTE 1: Funzione di Influenza ---
def calculate_influence(G, seed_set, func_type="f1"):
    total_influence = 0
    for v in G.Nodes():
        node_id = v.GetId()
        neighbors = set(v.GetOutEdges()) # num adj
        active_neighbors = neighbors.intersection(seed_set) # adj in seed set
        num_active = len(active_neighbors) # num adj in seed set
        degree = len(neighbors) # degree tot
        threshold = math.ceil(degree / 2) # threshold algorithm

        if func_type == "f1":
            total_influence += min(num_active, threshold)

        elif func_type == "f2":
            # num adj active submodulare
            for i in range(1, num_active + 1):
                total_influence += max(threshold - i + 1, 0)

        elif func_type == "f3":
            for i in range(1, num_active + 1):
                # Nota: il denominatore è il numero di amici che mancano per la soglia
                # In questa formula semplificata usiamo il grado totale per la normalizzazione
                total_influence += max((threshold - i + 1) / (degree - i + 1), 0)

    return total_influence


# --- PARTE 2: Algoritmo Greedy ---
def cost_seeds_greedy(G, k, costs, func_type="f1"):
    seed_set = set()
    current_cost = 0

    while True:
        best_node = None
        best_ratio = -1

        # Calcoliamo l'influenza attuale
        current_inf = calculate_influence(G, seed_set, func_type)

        for v in G.Nodes():
            node_id = v.GetId()
            if node_id not in seed_set:
                # Proviamo ad aggiungere il nodo v
                seed_set.add(node_id)
                new_inf = calculate_influence(G, seed_set, func_type)
                seed_set.remove(node_id)

                marginal_gain = new_inf - current_inf

                # Calcoliamo il rapporto Guadagno/Costo
                if current_cost + costs[node_id] <= k:
                    ratio = marginal_gain / costs[node_id]
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_node = node_id

        if best_node is not None:
            seed_set.add(best_node)
            current_cost += costs[best_node]
        else:
            break

    return seed_set


