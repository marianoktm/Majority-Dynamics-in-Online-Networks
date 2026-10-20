import math
import random


def random_cost(graph, min_val=1, max_val=10):
  costs = {}
  for ni in graph.Nodes():
    costs[ni.GetId()] = random.randint(min_val, max_val)
  return costs


def degree_cost(graph):
  costs = {}
  for ni in graph.Nodes():
    costs[ni.GetId()] = math.ceil(ni.GetDeg() / 2)
  return costs


def avg_neighbor_degree_cost(graph):
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


def average_cost(costs):
  """Calculates the average cost across all nodes.

  Args:
    costs (dict): A dictionary mapping node IDs to their costs.

  Returns:
    int: The average cost (ceil-ed), or 0 if the dictionary is empty.
  """
  if not costs:
    return 0
  return math.ceil(sum(costs.values()) / len(costs))