import random
import snap
import networkx as nx
import matplotlib.pyplot as plt
import argparse

import src_old.diffusion as diffusion
import src_old.graph_utils as graph_utils
import src_old.costs as costs
import src_old.seed as seed
import src_old.plotting as plotting
import src_old.utils as utils


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, help='Path of the graph to be loaded')
parser.add_argument('--test', action='store_true', help='Enable test mode to run on a smaller random subgraph for performance')
parser.add_argument('-b', '--budget', type=int, default=None, help='Budget available for seed selection. If not available, calculates a balanced budget based on random node costs')


def main():
    # Read and validate command line arguments
    args = parser.parse_args()

    dataset_path = args.dataset
    print(f'Dataset: {dataset_path}')

    is_test_mode = args.test
    print(f'is_test_mode: {is_test_mode}')

    if args.budget is not None and args.budget < 0:
        parser.error("Budget must be non-negative.")

    ######################################
    ##### PHASE 0: READING THE GRAPH #####
    ######################################
    # Read Snap Graph
    snap_graph = graph_utils.read_snap_graph(dataset_path, graph_type='undirected', format_type='edgelist')
    print(f'Snap Graph Nodes: {snap_graph.GetNodes()}, Snap Graph Edges: {snap_graph.GetEdges()}')

    # Using a Subgraph for performance improvements during testing
    if is_test_mode:
        snap_subgraph = graph_utils.get_connected_random_subgraph(snap_graph, 75)
        snap_graph = snap_subgraph
        print(f'Snap SubGraph Nodes: {snap_graph.GetNodes()}, Snap Graph Edges: {snap_graph.GetEdges()}')

    # Plot Graph
    # graph_utils.plot_snap_undirected_graph2(snap_graph=snap_graph)

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



    ###########################################################
    ### DA QUI IN POI NON GASA! non cancello perchè         ###
    ### potrebbe servire per implementare in modo sistemato ###
    ###########################################################


    #########################################
    ##### PHASE 1: COMPUTING NODE COSTS #####
    #########################################

    # 1. RANDOM COSTS
    random_costs = costs.random_cost(snap_graph)
    print(f'Random Costs: {random_costs}')

    # 2. DEGREE-BASED COSTS
    degree_ceil_costs = costs.degree_ceil_cost(snap_graph)
    print(f'Degree Ceil Costs: {degree_ceil_costs}')

    # 3. AVERAGE-NEIGHBOR-DEGREE-BASED COSTS
    avg_neighbor_degree_costs = costs.avg_neighbor_degree_cost(snap_graph)
    print(f'Average Neighbor Degree Costs: {avg_neighbor_degree_costs}')

    ##########################################
    ##### PHASE 2: SEEDSET DETERMINATION #####
    ##########################################

    # 1. COST-SEEDS-GREEDY with Cost Functions:
    # f1 -> random_costs()
    # f2 -> degree_ceil_costs()
    # f3 -> avg_neighbor_degree_costs()

    # 2. WTSS

    # 3. DEGREE CENTRALITY

    ###########################################################
    ##### PHASE 3: MAJORITY DYNAMICAL INFLUENCE DIFFUSION #####
    ###########################################################



    #################################################
    ##### PHASE 4: ONLINE NETWORK MODIFICATIONS #####
    #################################################

    # 1.

    ###################################################
    ##### PHASE 5: RESULTS AND DATA VISUALIZATION #####
    ###################################################

    # 1. BUDGET VARIATION PLOT
    # values of INF[G,S]
    # varying the budget
    # using the same seed set
    # across the three algorithms

    # 2. EDGE REMOVAL IMPACT
    # values of INF[G,S]
    # compared to INF[G',S']
    # while varying the number of removed edges

    # 3. NEIGHBORHOOD REMOVAL IMPACT
    # values of INF[G,S]
    # compared to INF[G',S']
    # while varying the number of removed nodes (and neighbors)

    # Use the provided budget, or calculate a balanced budget if omitted.
    costsTot = 0
    seed_setCost = 0
    if args.budget is not None:
        budget = args.budget
    else:
        budget = utils.balanced_budget(snap_graph, random_costs)
    print(f'Budget: {budget}')


    ### Algorithm 1 ###
    types = ["f1", "f2", "f3"] # tipologie funzioni di influenza
    results = {}

    print(f"Budget: {budget} | Costi assegnati a {snap_graph.GetNodes()} nodi.\n")

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