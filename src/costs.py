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


def custom_cost(graph):
  """Placeholder for your custom user-defined cost function.

  Satisfies the project requirement to design a custom cost function
  suitable for the problem.

  Args:
      graph: A SNAP graph object.

  Returns:
      dict: A dictionary mapping each node ID to its custom-calculated cost.
  """
  costs = {}
  for ni in graph.Nodes():
    # TODO: Implement your custom logic here (e.g., based on centrality, clustering, etc.)
    costs[ni.GetId()] = 1
  return costs