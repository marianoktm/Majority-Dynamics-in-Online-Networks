import collections
import random
import snap
from matplotlib import pyplot
import networkx


def read_snap_graph(file_path, graph_type='directed', format_type='edgelist'):
    """Reads a SNAP graph from a file using the Snap.py library.

    Parameters:
    - file_path (str): Path to the graph file.
    - graph_type (str): 'directed' (TNGraph), 'undirected' (TUNGraph), or
                        'network' (TNEANet with attributes).
    - format_type (str):
        - 'edgelist': Text file with integer node IDs (space/tab separated).
        - 'edgelist_str': Text file with arbitrary string node IDs.
        - 'binary': SNAP's compact native binary format (.graph).

    Returns:
    - A Snap.py graph or network object.
    """
    # Map text identifiers to respective Snap.py graph classes
    graph_classes = {
        'directed': snap.TNGraph,
        'undirected': snap.TUNGraph,
        'network': snap.TNEANet,
    }

    if graph_type not in graph_classes:
        raise ValueError(
            "Invalid graph_type. Choose from 'directed', 'undirected', or 'network'."
        )

    g_cls = graph_classes[graph_type]

    # Load graph depending on the specified format
    if format_type == 'edgelist':
        # Standard text edge list (Source ID, Destination ID)
        return snap.LoadEdgeList(g_cls, file_path, 0, 1)

    elif format_type == 'edgelist_str':
        # Text edge list where node IDs are strings (e.g., names, strings)
        return snap.LoadEdgeListStr(g_cls, file_path, 0, 1)

    elif format_type == 'binary':
        # SNAP binary stream format
        f_in = snap.TFIn(file_path)
        return g_cls.Load(f_in)

    else:
        raise ValueError("Invalid format_type. Choose from 'edgelist', 'edgelist_str', or binary'.")


def plot_snap_undirected_graph(snap_graph, mult=1.0):
  """Converts a SNAP undirected graph (TUNGraph) to a NetworkX graph

  and renders it using Matplotlib with a larger canvas, thin/transparent edges,
  and wider node spacing.
  """
  # Initialize an empty NetworkX undirected graph
  nx_graph = networkx.Graph()

  # Add nodes from the SNAP graph
  for ni in snap_graph.Nodes():
    nx_graph.add_node(ni.GetId())

  # Add edges from the SNAP graph
  for ei in snap_graph.Edges():
    nx_graph.add_edge(ei.GetSrcNId(), ei.GetDstNId())

  # Bigger image canvas size
  pyplot.figure(figsize=(20, 20))

  # Compute node positions with expanded spacing
  num_nodes = nx_graph.number_of_nodes()
  # Default k is 1 / sqrt(n). We multiply by mult to increase the target spacing.
  k_val = (mult / (num_nodes**0.5)) if num_nodes > 0 else None

  pos = networkx.spring_layout(
      nx_graph,
      seed=42,
      k=k_val,  # Higher value = nodes push further apart
      iterations=200,  # More steps for the layout to expand and settle
      #scale=2.0,  # Expands the outer coordinate boundaries
  )

  # Draw nodes (fully opaque)
  networkx.draw_networkx_nodes(
      nx_graph, pos, node_color='blue', node_size=22, alpha=1.0
  )

  # Draw edges (thin and transparent)
  nx_graph_edges = networkx.draw_networkx_edges(
      nx_graph, pos, edge_color='gray', width=0.4, alpha=0.50
  )

  # Add title with graph statistics
  num_edges = nx_graph.number_of_edges()
  pyplot.title(
      f'SNAP Undirected Graph Visualization\nNodes: {num_nodes}, Edges:'
      f' {num_edges}',
      fontsize=14,
  )

  # Clean up layout and hide axis ticks
  pyplot.axis('off')
  pyplot.tight_layout()
  pyplot.show()


def get_connected_random_subgraph(graph, sample_size=50):
  """Given a SNAP graph, returns a smaller *connected* subgraph

  induced by a breadth-first search (BFS) expansion from a random start node.

  Parameters:
  - graph: A Snap.py graph (TNGraph, TUNGraph, or TNEANet).
  - sample_size:
      - If int: exact number of nodes to sample.
      - If float (0.0 < x <= 1.0): fraction of total nodes to sample.

  Returns:
  - A connected SNAP subgraph of the same type.
  """
  node_ids = [ni.GetId() for ni in graph.Nodes()]
  total_nodes = len(node_ids)

  if not node_ids:
    raise ValueError('The input graph is empty.')

  # Handle fraction vs absolute count for sample size
  if isinstance(sample_size, float) and 0.0 < sample_size <= 1.0:
    k = int(total_nodes * sample_size)
  elif isinstance(sample_size, int):
    k = sample_size
  else:
    raise ValueError(
        "sample_size must be an integer count or a float fraction (0.0, 1.0]."
    )

  k = max(1, min(k, total_nodes))

  best_nodes = []

  # Try up to 10 random start nodes to find a sufficiently large component
  for _ in range(10):
    start_nid = random.choice(node_ids)
    visited = [start_nid]
    visited_set = {start_nid}
    queue = collections.deque([start_nid])

    while queue and len(visited) < k:
      curr_nid = queue.popleft()
      node_i = graph.GetNI(curr_nid)

      nbr_ids = []
      try:
        # For TUNGraph (undirected)
        for i in range(node_i.GetDeg()):
          nbr_ids.append(node_i.GetNbrNId(i))
      except AttributeError:
        # For TNGraph and TNEANet (directed/network)
        for i in range(node_i.GetOutDeg()):
          nbr_ids.append(node_i.GetOutNId(i))
        for i in range(node_i.GetInDeg()):
          nbr_ids.append(node_i.GetInNId(i))

      # Shuffle neighbors to randomize expansion path
      random.shuffle(nbr_ids)

      for nbr in nbr_ids:
        if nbr not in visited_set:
          visited_set.add(nbr)
          visited.append(nbr)
          queue.append(nbr)
          if len(visited) >= k:
            break

    if len(visited) >= len(best_nodes):
      best_nodes = visited
      if len(best_nodes) >= k:
        break

  # Extract and return the connected induced subgraph
  return graph.GetSubGraph(best_nodes)