"""
Centrality-based greedy seed selection.

Baseline seed-selection algorithm that ranks nodes by degree centrality divided by
cost and greedily picks the best-ranked nodes that fit in the budget.
"""

import snap


def centrality_cost_seeds_greedy(graph, budget, costs):
    """
    Function: centrality_cost_seeds_greedy

    Greedy seed selection by degree centrality / cost. Nodes are ranked by score; each is
    added if it fits in the remaining budget, otherwise skipped.

    Args:
        graph (snap.TUNGraph): Undirected graph.
        budget (int): Maximum total cost allowed for the seed set.
        costs (dict): Maps each node ID (int) to its cost (int).

    Returns:
        set: Node IDs of the seed set (empty if the graph has 1 node or fewer).
    """
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
