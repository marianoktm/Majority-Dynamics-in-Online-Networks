import random
import snap
import networkx as nx
import matplotlib.pyplot as plt
import argparse



import diffusion
import src.graph_utils as graph_utils
import src.costs as costs
import seed
import plotting

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
        snapSubgraph = graph_utils.get_connected_random_subgraph(snapGraph, 75)
        snapGraph = snapSubgraph
        print(f'Snap SubGraph Nodes: {snapGraph.GetNodes()}, Snap Graph Edges: {snapGraph.GetEdges()}')

    # Plot Graph
    # graph_utils.plot_snap_undirected_graph2(snap_graph=snapGraph)

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

    #########################################
    ##### PHASE 2: SEEDSET ALGORITHMS #####
    #########################################

    G = snapGraph
    costsNode = randomCosts
    costsTot = 0
    seed_setCost = 0

# budget bilanced
    for v in G.Nodes():
        node_id = v.GetId()
        c = costsNode[node_id]
        costsTot += c
    # print(f'Total Costs: {costsTot}')
    #budget = costsTot / 2
    budget = 100

    ### Algorithm 1 ###
    types = ["f1", "f2", "f3"] # tipologie funzioni di influenza
    results = {}

    print(f"Budget: {budget} | Costi assegnati a {snapGraph.GetNodes()} nodi.\n")


    for t in types:
        # Trova il seed set ottimale per quella funzione
        seed_set = seed.cost_seeds_greedy(G, budget, costsNode, func_type=t)
        for v in seed_set:
            seed_setCost += costsNode[v]
        print(f"Seed Set: {seed_setCost}")

        # Simula la cascata reale per quel seed set
        final_active_nodes = diffusion.simulate_cascade(G, seed_set)

        results[t] = {
            "seeds": seed_set,
            "total_influenced": len(final_active_nodes),
            "spent_cost": seed_setCost
        }

        print(f"Risultato con {t}:")
        print(f"  Seed Set: {seed_set}")
        print(f"  Costo speso: {seed_setCost} / Budget: {budget}")
        print(f"  Persone totali influenzate: {results[t]['total_influenced']}\n")

        #plot_diffusion_comparison(G, seed_set, final_active_nodes)
        plotting.plot_diffusion_costs(G, seed_set, final_active_nodes, costsNode)
        seed_setCost = 0

    best_name, best_data = seed.select_best_strategy(results)
    print("\n==================================================")
    print(f"   STRATEGIA OTTIMALE SCELTA: {best_name.upper()}")
    print(f"   Persone influenzate: {best_data['total_influenced']}")
    print(f"   Costo totale sostenuto: {best_data['spent_cost']}")
    print("==================================================")


    ### Algorithm 3 - Degree Centrality ###

    seed_set = seed.centrality_cost_seeds_greedy(G,budget,costsNode)
    degree_centrality_active_nodes = diffusion.simulate_cascade(G, seed_set)
    plotting.plot_diffusion_costs(G, seed_set, degree_centrality_active_nodes, costsNode)
    print("\n====================================================")
    print(f"Degree Centrality: {len(seed_set)}")


if __name__ == "__main__":
    main()