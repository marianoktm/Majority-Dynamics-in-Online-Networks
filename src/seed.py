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

# Best between f1, f2 and f3
def select_best_strategy(results):

    best_strategy = None
    max_influenced = -1
    min_cost = float('inf')

    for t, data in results.items():
        influenced = data["total_influenced"]
        cost = data["spent_cost"]

        # Criterio 1: Massimizzare l'influenza
        if influenced > max_influenced:
            max_influenced = influenced
            min_cost = cost
            best_strategy = t

        # Criterio 2: A parità di influenza, minimizzare il costo
        elif influenced == max_influenced:
            if cost < min_cost:
                min_cost = cost
                best_strategy = t

    return best_strategy, results[best_strategy]

#########################################
##### ALGORITHMS 3 Degree Centrality #####
#########################################

def centrality_cost_seeds_greedy(G, budget, costs):

    # controllo esistenza nodi
    n_nodes = G.GetNodes()
    if n_nodes <= 1:
        return set()

    seed_set = set()
    current_cost = 0

    # calcolo score (Centrality / Costo) per TUTTI i nodi in anticipo
    node_scores = []
    for v in G.Nodes():
        node_id = v.GetId()
        deg = v.GetDeg()
        cost = costs[node_id]  # Costo del nodo (default 1 per sicurezza)

        if cost <= 0:
            cost = 1  # Evita divisioni per zero

        # Calcolo Degree Centrality
        deg_centrality = deg / (n_nodes - 1)

        # Rapporto richiesto
        score = deg_centrality / cost

        node_scores.append((node_id, score, cost))

    # key=lambda x: x[1] definisce l'ordinamento in base al secondo elemento della tripla (score)
    # reverse=True --> ordinamento decrescente
    node_scores.sort(key=lambda x: x[1], reverse=True)

    # scegliamo i nodi finché c'è budget (Logica Greedy)
    for node_id, score, cost in node_scores:
        if current_cost + cost <= budget:
            seed_set.add(node_id)
            current_cost += cost
        # Se un nodo sfora il budget rimanente, il ciclo continua
        # provando a prendere il nodo successivo più efficiente

    return seed_set





















