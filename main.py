import random
import snap
import networkx
import matplotlib.pyplot as pyplot
import argparse
import math

from src import graph_utils


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

    ##########################################
    ### EXPERIMENT 1: MODIFYING THE BUDGET ###
    ##########################################

    ####################################
    ### EXPERIMENT 2: REMOVING EDGES ###
    ####################################

    #####################################################
    ### EXPERIMENT 3: REMOVING VERTEX (AND NEIGHBORS) ###
    #####################################################

    ############################
    ### PLOTTING THE RESULTS ###
    ############################


if __name__ == '__main__':
   main()
