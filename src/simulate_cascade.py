import math


def simulate_cascade(graph, seed_set):
    active_nodes = set(seed_set) # Initialize the active set with the nodes selected as seeds.
    new_activations = True # Continue while at least one new node was activated in the previous round.

    # Main loop
    while new_activations:
        new_activations = False # The flag becomes True only if this round activates at least one previously inactive node.
        nodes_to_activate = [] # Collect activations separately so that all nodes in this round are evaluated using the same active set.

        # Evaluate every node in the graph.
        for NI in graph.Nodes():
            v = NI.GetId()

            if v in active_nodes:
                continue

            # Obtain the node's neighbors and remove duplicate IDs, if any. For an undirected graph, these represent its adjacent nodes.
            neighbors = set(NI.GetOutEdges())

            if not neighbors:
                continue

            active_neighbors = neighbors & active_nodes # Count how many neighbors are active at the start of this round.
            threshold = math.ceil(len(neighbors) / 2) # Compute the majority threshold using the current neighborhood.

            # Schedule the node for activation if enough neighbors are active.
            if len(active_neighbors) >= threshold:
                nodes_to_activate.append(v)

        # Apply all activations simultaneously at the end of the round.
        if nodes_to_activate:
            active_nodes.update(nodes_to_activate)
            new_activations = True

    return active_nodes


