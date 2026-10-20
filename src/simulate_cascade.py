import math


def simulate_cascade(graph, seed_set):
    active_nodes = set(seed_set)  # 1. Start from the seed set
    new_activations = True  # 2. Flag to manage the loop

    while new_activations:
        new_activations = False  # 3. Reset flag at each iteration
        nodes_to_activate = []  # 4. List for newly activated nodes

        for v in graph.Nodes():
            node_id = v.GetId()
            if node_id not in active_nodes:  # 5. Check only inactive nodes
                neighbors = set(v.GetOutEdges())  # Get neighboring nodes

                if not neighbors:
                    continue

                active_neighbors = neighbors.intersection(active_nodes)  # Count active neighbors

                # 6. Majority Rule: |N(v) ∩ Inf| >= ceil(d(v) / 2)[cite: 4, 5]
                deg = len(neighbors)
                threshold = math.ceil(deg / 2)

                if len(active_neighbors) >= threshold:
                    nodes_to_activate.append(node_id)

        # 7. Update the cascade wave
        if nodes_to_activate:
            for node in nodes_to_activate:
                active_nodes.add(node)
            new_activations = True  # Someone activated, another round is needed!

    return active_nodes



def simulate_cascade2(graph, seed_set):
    active_nodes = set(seed_set)
    new_activations = True

    while new_activations:
        new_activations = False
        nodes_to_activate = []
        for v in graph.Nodes():
            node_id = v.GetId()
            if node_id not in active_nodes: neighbors = set(v.GetOutEdges())

