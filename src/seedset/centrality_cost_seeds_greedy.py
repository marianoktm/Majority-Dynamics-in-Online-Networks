import snap


def centrality_cost_seeds_greedy(graph, budget, costs):
    # controllo esistenza nodi
    n_nodes = graph.GetNodes()
    if n_nodes <= 1:
        return set()

    seed_set = set()
    current_cost = 0

    # calcolo score (Centrality / Costo) per TUTTI i nodi in anticipo
    node_scores = []
    for v in graph.Nodes():
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