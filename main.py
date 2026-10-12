import random
import snap
import networkx as nx
import matplotlib.pyplot as plt
import argparse
import src.graph_utils as graph_utils
import src.costs as costs

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, help='Path of the graph to be loaded')
parser.add_argument('--test', action='store_true', help='Enable test mode to run on a smaller random subgraph for performance')

def main():
    # Read dataset path argument and testing options
    args = parser.parse_args()

    dataset_path = args.dataset
    print(f'Dataset: {dataset_path}')

    is_test_mode = args.test
    print(f'is_test_mode: {is_test_mode}')

    ######################################
    ##### PHASE 0: READING THE GRAPH #####
    ######################################
    # Read Snap Graph
    snapGraph = graph_utils.read_snap_graph(dataset_path, graph_type='undirected', format_type='edgelist')
    print(f'Snap Graph Nodes: {snapGraph.GetNodes()}, Snap Graph Edges: {snapGraph.GetEdges()}')

    # Using a Subgraph for performance improvements during testing
    if is_test_mode:
        snapSubgraph = graph_utils.get_connected_random_subgraph(snapGraph, 50)
        snapGraph = snapSubgraph
        print(f'Snap SubGraph Nodes: {snapGraph.GetNodes()}, Snap Graph Edges: {snapGraph.GetEdges()}')

    # Plot Graph
    graph_utils.plot_snap_undirected_graph2(snap_graph=snapGraph)

    #########################################
    ##### PHASE 1: COMPUTING NODE COSTS #####
    #########################################

    # Generating random costs dictionary
    randomCosts = costs.random_cost(snapGraph)
    print(f'Random Costs: {randomCosts}')

    # Generating degree-related costs dictionary
    degreeCeilCosts = costs.degree_ceil_cost(snapGraph)
    print(f'Degree Ceil Costs: {degreeCeilCosts}')

    # Generating custom costs dictionary
    avgNeighborDegreeCosts = costs.avg_neighbor_degree_cost(snapGraph)
    print(f'Average Neighbor Degree Costs: {avgNeighborDegreeCosts}')


if __name__ == "__main__":
    main()