import math


def simulate_cascade(graph, seed_set):
    """
    Simulate a majority-threshold cascade on an undirected SNAP graph.

    The cascade starts with a set of initially active seed nodes. At each
    iteration, every inactive node is checked against the majority rule:
    a node becomes active if at least half of its neighbors are active,
    rounded up to the nearest integer.

    Activations occur in synchronous rounds. A node activated during a
    round cannot influence other nodes until the next round. The process
    terminates when a complete round produces no new activations.

    Parameters
    ----------
    graph : snap.TUNGraph
        Undirected input graph. Neighbor relationships are obtained from
        the SNAP node interface.

    seed_set : set[int] or iterable[int]
        Node IDs that are active at the beginning of the simulation.
        These nodes remain active throughout the cascade.

    Returns
    -------
    set[int]
        The set of all active node IDs at termination, including the
        original seeds and every node activated during the cascade.

    Notes
    -----
    For each node v, the activation threshold is

        threshold(v) = ceil(degree(v) / 2)

    where degree(v) is the number of distinct neighbors returned by the
    graph interface.

    Inactive isolated nodes are never activated because they have no
    neighbors. Seed nodes remain active regardless of their degree.

    The function assumes that seed_set contains valid node IDs from graph.
    """
    # Initialize the active set with the nodes selected as seeds. Converting to a set also ensures that each node is represented once.
    active_nodes = set(seed_set)

    # Continue while at least one new node was activated in the previous round. Initially, the first round must always be evaluated.
    new_activations = True

    while new_activations:
        # Reset the flag. It becomes True again only if this round activates at least one previously inactive node.
        new_activations = False

        # Collect activations separately so that all nodes in this round are evaluated using the same active set.
        nodes_to_activate = []

        # Evaluate every node in the graph.
        for node in graph.Nodes():
            node_id = node.GetId()

            # Active nodes cannot become inactive, so they need no further activation checks.
            if node_id in active_nodes:
                continue

            # Obtain the node's neighbors and remove duplicate IDs, if any. For an undirected graph, these represent its adjacent nodes.
            neighbors = set(node.GetOutEdges())

            # An isolated inactive node has no active neighbors and cannot be activated by the majority rule.
            if not neighbors:
                continue

            # Count how many neighbors are active at the start of this round.
            active_neighbors = neighbors & active_nodes

            # Compute the majority threshold using the current neighborhood.
            degree = len(neighbors)
            threshold = math.ceil(degree / 2)

            # Schedule the node for activation if enough neighbors are active. Do not update active_nodes here: activations are synchronous.
            if len(active_neighbors) >= threshold:
                nodes_to_activate.append(node_id)

        # Apply all activations simultaneously at the end of the round.
        if nodes_to_activate:
            active_nodes.update(nodes_to_activate)

            # Another round is needed because the newly activated nodes may cause additional inactive nodes to meet their thresholds.
            new_activations = True

    # Return the final active set once the cascade reaches a fixed point.
    return active_nodes


