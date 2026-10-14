import math
import random


def random_cost(graph, min_val=1, max_val=10):
  """Generates a random integer cost within a fixed range for each node.

  Satisfies the project requirement for random node costs.

  Args:
      graph: A SNAP graph object.
      min_val (int): The minimum cost value (inclusive). Defaults to 1.
      max_val (int): The maximum cost value (inclusive). Defaults to 10.

  Returns:
      dict: A dictionary mapping each node ID to its generated integer cost.
  """
  costs = {}
  for ni in graph.Nodes():
    costs[ni.GetId()] = random.randint(min_val, max_val)
  return costs


def degree_ceil_cost(graph):
  """Computes the degree-based cost function c(u) = ceil(d(u) / 2) for each node.

  Satisfies the project requirement for degree-based node costs.

  Args:
      graph: A SNAP graph object.

  Returns:
      dict: A dictionary mapping each node ID to half its degree rounded up.
  """
  costs = {}
  for ni in graph.Nodes():
    costs[ni.GetId()] = math.ceil(ni.GetDeg() / 2)
  return costs


def avg_neighbor_degree_cost(graph):
  """Custom Cost Function: Average Degree of Neighbors.

  Computes the average degree of a node's immediate neighbors,
  rounded up to the nearest integer to satisfy integer cost requirements[cite:
  6, 16].

  Args:
      graph: A SNAP graph object.

  Returns:
      dict: A dictionary mapping each node ID to its neighbor-average-based cost.
  """
  costs = {}
  for ni in graph.Nodes():
    node_id = ni.GetId()
    deg = ni.GetDeg()

    # Handle isolated nodes (degree 0) to avoid division by zero
    if deg == 0:
      costs[node_id] = 1
      continue

    # Sum the degrees of all neighbors
    neighbor_degrees_sum = 0
    for i in range(deg):
      nbr_id = ni.GetNbrNId(i)
      nbr_node = graph.GetNI(nbr_id)
      neighbor_degrees_sum += nbr_node.GetDeg()

    # Compute average neighbor degree
    avg_neighbor_deg = neighbor_degrees_sum / deg

    # Assign integer cost (minimum cost of 1, rounded up)
    costs[node_id] = max(1, math.ceil(avg_neighbor_deg))

  return costs