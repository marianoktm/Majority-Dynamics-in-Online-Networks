
# function for diffusion with seed_set
def simulate_cascade(G, seed_set):
    active_nodes = set(seed_set)  # 1. Partiamo dal seed set
    new_activations = True  # 2. Flag per gestire il ciclo

    while new_activations:
        new_activations = False  # 3. Reset del flag a ogni giro
        nodes_to_activate = []  # 4. Lista per i "nuovi convertiti"

        for v in G.Nodes():
            node_id = v.GetId()
            if node_id not in active_nodes:  # 5. Guarda solo chi non è ancora attivo
                neighbors = set(v.GetOutEdges()) # chi sono adj

                if not neighbors:
                    continue

                active_neighbors = neighbors.intersection(active_nodes) # num adj active

                # 6. La Regola della Maggioranza
                if len(active_neighbors) > len(neighbors) / 2: # regola di attivazione
                    nodes_to_activate.append(node_id)

        # 7. Aggiornamento della "Onda"
        if nodes_to_activate:
            for node in nodes_to_activate:
                active_nodes.add(node)
            new_activations = True  # Se qualcuno si è attivato, dobbiamo rifare un giro!

    return active_nodes