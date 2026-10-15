import random
from itertools import product

import snap
import networkx
import matplotlib.pyplot as pyplot
import argparse
import math

from src import graph_utils, costs
from src.experiments.run_experiment import run_experiment
from src.seedset import cost_seeds_greedy, centrality_cost_seeds_greedy, wtss
from src.experiments import budget_variation, edge_removal, vertex_removal

from src.seedset import cost_seeds_greedy, wtss, centrality_cost_seeds_greedy

# Parser
parser = argparse.ArgumentParser()

# Dataset input argument.
parser.add_argument('-d',
                    '--dataset', type=str,
                    help='Path of the graph to be loaded')

# Test mode argument (uses a subgraph for performance reasons).
parser.add_argument('--test',
                    action='store_true',
                    help='Enable test mode to run on a smaller random subgraph for performance')

# Budget input argument (if not set, calculates and uses the maximum budget).
parser.add_argument('-b',
                    '--budget',
                    type=int,
                    default=None,
                    help='Budget available for seed selection. If not available, calculates a maximum budget')


def main():
    ##############################
    ### PRELIMINARY OPERATIONS ###
    ##############################
    args = parser.parse_args()

    # Getting input graph path from command line.
    data_path = args.dataset
    print("Data Path: ", data_path)

    # Loading the graph with SNAP as an edgelist.
    print('Loading Graph Data...')
    graph = graph_utils.read_snap_graph(file_path=data_path, graph_type='undirected', format_type='edgelist')
    print(f'SNAP Graph Nodes: {graph.GetNodes()}, Snap Graph Edges: {graph.GetEdges()}')

    print('\n====================================================================\n')

    # Check whether test mode is enabled. If yes, convert the graph into a random subgraph.
    test_mode = args.test
    if test_mode:
        print('Test mode enabled!')
        print("Converting graph to subgraph (10% size)...")
        subgraph = graph_utils.get_connected_random_subgraph(graph=graph, sample_size=math.ceil(graph.GetNodes()*0.1))
        graph = subgraph
        print(f'SNAP SubGraph Nodes: {graph.GetNodes()}, Snap Graph Edges: {graph.GetEdges()}')

    # Plot the graph (if the graph is big it may take a while...)
    print('Drawing the graph...')
    if graph.GetNodes() > 1000:
        print('Warning: Graph has an huge number of nodes! It might take a while...')
    graph_utils.plot_snap_undirected_graph(graph)

    print('\n====================================================================\n')

    ##############################################
    ### PRELIMINARY OPERATIONS FOR EXPERIMENTS ###
    ##############################################

    random_costs = costs.random_cost(graph, 1, 10)
    print("Random Costs: ", random_costs)

    degree_ceil_costs = costs.degree_ceil_cost(graph)
    print("Degree Ceil Costs: ", degree_ceil_costs)

    avg_neighbor_degree_costs = costs.avg_neighbor_degree_cost(graph)
    print("Average Neighbor Degree Costs: ", avg_neighbor_degree_costs)

    print('\n====================================================================\n')

    # Collections of the computed costs
    graph_costs = [random_costs, degree_ceil_costs, avg_neighbor_degree_costs]

    # Collection of the seed set functions
    seed_set_functions = [cost_seeds_greedy, wtss, centrality_cost_seeds_greedy]

    # Pairs of seed_set_functions and graph_costs to run the experiments easily
    experiment_pairs = list(product(seed_set_functions, graph_costs))

    ##########################################
    ### EXPERIMENT 1: MODIFYING THE BUDGET ###
    ##########################################
    run_experiment(experiment_function=budget_variation.budget_variation_experiment,
                   graph=graph,
                   experiment_pairs=experiment_pairs)

    print('\n====================================================================\n')

    ####################################
    ### EXPERIMENT 2: REMOVING EDGES ###
    ####################################
    run_experiment(experiment_function=edge_removal.edge_removal_experiment,
                   graph=graph,
                   experiment_pairs=experiment_pairs)

    print('\n====================================================================\n')

    #####################################################
    ### EXPERIMENT 3: REMOVING VERTEX (AND NEIGHBORS) ###
    #####################################################
    run_experiment(experiment_function=vertex_removal.vertex_removal_experiment,
                   graph=graph,
                   experiment_pairs=experiment_pairs)

    print('\n====================================================================\n')

    ############################
    ### PLOTTING THE RESULTS ###
    ############################


if __name__ == '__main__':
   main()
