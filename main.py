import os
import random
from itertools import product
import snap
import networkx
import matplotlib.pyplot as pyplot
import argparse
import math

from src import graph_utils, costs_functions, plotting
from src.experiments.run_experiment import run_experiment
from src.experiments import budget_variation, edges_removal, vertices_removal

from src.seedset.wtss import wtss
from src.seedset.centrality_cost_seeds_greedy import centrality_cost_seeds_greedy

from src.seedset.cost_seeds_greedy_lazy import cost_seeds_greedy

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

# Budget input argument (if not set, calculates and uses the average budget based on random costs).
parser.add_argument('-b',
                    '--budget',
                    type=int,
                    default=None,
                    help='Budget available for seed selection. If not available, calculates the average budget based on random costs')

# Save path argument for saving plots figures (defaults to "results" in the current working directory)
parser.add_argument('-s',
                    '--save-path',
                    type=str,
                    default=os.path.join(os.getcwd(), 'results'),
                    help='Path to the results folder to save the plots (defaults to the "result" folder in the current working directory)')

def main():
    ##############################
    ### PRELIMINARY OPERATIONS ###
    ##############################
    args = parser.parse_args()

    # Getting input graph path from command line.
    data_path = args.dataset
    print("Data Path: ", data_path)

    # Ensure the save directory exists
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)

    # Loading the graph with SNAP as an edgelist.
    print('Loading Graph Data...')
    graph = graph_utils.read_snap_graph(file_path=data_path, graph_type='undirected', format_type='edgelist')
    print(f'SNAP Graph Nodes: {graph.GetNodes()}, Snap Graph Edges: {graph.GetEdges()}, Type: {type(graph)}')

    print('\n====================================================================\n')

    # Check whether test mode is enabled. If yes, convert the graph into a random subgraph.
    test_mode = args.test
    if test_mode:
        print('Test mode enabled!')
        print("Converting graph to subgraph (10% size)...")
        subgraph = graph_utils.get_connected_random_subgraph(graph=graph, sample_size=math.ceil(graph.GetNodes()*0.05))
        graph = subgraph
        print(f'SNAP SubGraph Nodes: {graph.GetNodes()}, Snap Graph Edges: {graph.GetEdges()}')

    # Plot the graph (if the graph is big it may take a while...)
    print('Drawing the graph...')
    if graph.GetNodes() > 1000:
        print('Warning: Graph has an huge number of nodes! It might take a while...')
    #graph_utils.plot_snap_undirected_graph(graph)

    print('\n====================================================================\n')

    ##############################################
    ### PRELIMINARY OPERATIONS FOR EXPERIMENTS ###
    ##############################################

    random_costs = costs_functions.random_cost(graph, 1, 10)
    print("Random Costs: ", random_costs)

    degree_costs = costs_functions.degree_cost(graph)
    print("Degree Costs: ", degree_costs)

    avg_neighbor_degree_costs = costs_functions.avg_neighbor_degree_cost(graph)
    print("Average Neighbor Degree Costs: ", avg_neighbor_degree_costs)

    print('\n====================================================================\n')

    # Tuples of seed_set_functions and graph_costs to run the experiments easily
    experiment_pairs_rc= dict()
    experiment_pairs_rc["random-costs-cost-seeds-greedy"] = (random_costs, cost_seeds_greedy)
    experiment_pairs_rc["random-costs-wtss"] = (random_costs, wtss)
    experiment_pairs_rc["random-costs-centrality-cost-seeds-greedy"] = (random_costs, centrality_cost_seeds_greedy)

    experiment_pairs_dc = dict()
    experiment_pairs_dc["degree-costs-cost-seeds-greedy"] = (degree_costs, cost_seeds_greedy)
    experiment_pairs_dc["degree-costs-wtss"] = (degree_costs, wtss)
    experiment_pairs_dc["degree-costs-centrality-cost-seeds-greedy"] = (degree_costs, centrality_cost_seeds_greedy)

    experiment_pairs_andc = dict()
    experiment_pairs_andc["avg-neighbor-degree-costs-cost-seeds-greedy"] = (avg_neighbor_degree_costs, cost_seeds_greedy)
    experiment_pairs_andc["avg-neighbor-degree-costs-wtss"] = (avg_neighbor_degree_costs, wtss)
    experiment_pairs_andc["avg-neighbor-degree-costs-centrality-cost-seeds-greedy"] = (avg_neighbor_degree_costs, centrality_cost_seeds_greedy)
    
    # Budget (if not set, calculates it based on average random cost of nodes)
    if args.budget is not None:
        budget = args.budget
    else:
        budget = math.ceil(costs_functions.average_cost(random_costs) * graph.GetNodes() * 0.20)
    print("Budget: ", budget)

    print('\n====================================================================\n')

    ##########################################
    ### EXPERIMENT 1: MODIFYING THE BUDGET ###
    ##########################################
    # RANDOM COSTS
    budget_results_rc = run_experiment(experiment_function=budget_variation.budget_variation_experiment,
                                               graph=graph,
                                               experiment_pairs=experiment_pairs_rc)
    print("Budget Experiment Results (Random Costs):")
    print(budget_results_rc)
    plotting.plot_experiment_results(data=budget_results_rc,
                                     xlabel='Budget',
                                     ylabel='|Inf[G,S]|',
                                     title='Budget Experiment Results (Random Costs)',
                                     save_path=save_path)

    print('\n====================================================================\n')

    # DEGREE COSTS
    budget_results_dc = run_experiment(experiment_function=budget_variation.budget_variation_experiment,
                                                  graph=graph,
                                                  experiment_pairs=experiment_pairs_dc)
    print("Budget Experiment Results (Degree Costs):")
    print(budget_results_dc)
    plotting.plot_experiment_results(data=budget_results_dc,
                                     xlabel='Budget',
                                     ylabel='|Inf[G,S]|',
                                     title='Budget Experiment Results (Degree Costs)',
                                     save_path=save_path)

    print('\n====================================================================\n')

    # AVERAGE NEIGHBOR DEGREE COSTS
    budget_results_andc = run_experiment(experiment_function=budget_variation.budget_variation_experiment,
                                                    graph=graph,
                                                    experiment_pairs=experiment_pairs_andc)
    print("Budget Experiment Results (Average Neighbor Degree Costs):")
    print(budget_results_andc)
    plotting.plot_experiment_results(data=budget_results_andc,
                                     xlabel='Budget',
                                     ylabel='|Inf[G,S]|',
                                     title='Budget Experiment Results (Average Neighbor Degree Costs)',
                                     save_path=save_path)

    print('\n====================================================================\n')

    ####################################
    ### EXPERIMENT 2: REMOVING EDGES ###
    ####################################

    # RANDOM COSTS
    edges_removal_results_rc = run_experiment(experiment_function=edges_removal.edges_removal_experiment,
                                              graph=graph,
                                              experiment_pairs=experiment_pairs_rc,
                                              budget=budget)
    print("Edges Removal Experiment Results (Random Costs):")
    print(edges_removal_results_rc)
    plotting.plot_experiment_results(data=edges_removal_results_rc,
                                     xlabel='Removed Edges',
                                     ylabel='|Inf[G\',S]|',
                                     title='Edges Removal Experiment Results (Random Costs)',
                                     save_path=save_path)

    print('\n====================================================================\n')

    # DEGREE COSTS
    edges_removal_results_dc = run_experiment(experiment_function=edges_removal.edges_removal_experiment,
                                              graph=graph,
                                              experiment_pairs=experiment_pairs_dc,
                                              budget=budget)
    print("Edges Removal Experiment Results (Degree Costs):")
    print(edges_removal_results_dc)
    plotting.plot_experiment_results(data=edges_removal_results_dc,
                                     xlabel='Removed Edges',
                                     ylabel='|Inf[G\',S]|',
                                     title='Edges Removal Experiment Results (Degree Costs)',
                                     save_path=save_path)

    print('\n====================================================================\n')

    # AVERAGE NEIGHBOR DEGREE COSTS
    edges_removal_results_andc = run_experiment(experiment_function=edges_removal.edges_removal_experiment,
                                                graph=graph,
                                                experiment_pairs=experiment_pairs_andc,
                                                budget=budget)
    print("Edges Removal Experiment Results (Average Neighbor Degree Costs):")
    print(edges_removal_results_andc)
    plotting.plot_experiment_results(data=edges_removal_results_andc,
                                     xlabel='Removed Edges',
                                     ylabel='|Inf[G\',S]|',
                                     title='Edges Removal Experiment Results (Average Neighbor Degree Costs)',
                                     save_path=save_path)

    print('\n====================================================================\n')

    #####################################################
    ### EXPERIMENT 3: REMOVING VERTEX (AND NEIGHBORS) ###
    #####################################################

    # RANDOM COSTS
    vertices_removal_results_rc = run_experiment(experiment_function=vertices_removal.vertices_removal_experiment,
                                              graph=graph,
                                              experiment_pairs=experiment_pairs_rc,
                                              budget=budget)
    print("Vertices Removal Experiment Results (Random Costs):")
    print(vertices_removal_results_rc)
    plotting.plot_experiment_results(data=vertices_removal_results_rc,
                                     xlabel='Removed Vertices',
                                     ylabel='|Inf[G\',S]|',
                                     title='Vertices Removal Experiment Results (Random Costs)',
                                     save_path=save_path)

    print('\n====================================================================\n')

    # DEGREE COSTS
    vertices_removal_results_dc = run_experiment(experiment_function=vertices_removal.vertices_removal_experiment,
                                              graph=graph,
                                              experiment_pairs=experiment_pairs_dc,
                                              budget=budget)
    print("Vertices Removal Experiment Results (Degree Costs):")
    print(vertices_removal_results_dc)
    plotting.plot_experiment_results(data=vertices_removal_results_dc,
                                     xlabel='Removed Vertices',
                                     ylabel='|Inf[G\',S]|',
                                     title='Vertices Removal Experiment Results (Degree Costs)',
                                     save_path=save_path)

    print('\n====================================================================\n')

    # AVERAGE NEIGHBOR DEGREE COSTS
    vertices_removal_results_andc = run_experiment(experiment_function=vertices_removal.vertices_removal_experiment,
                                                graph=graph,
                                                experiment_pairs=experiment_pairs_andc,
                                                budget=budget)
    print("Vertices Removal Experiment Results (Average Neighbor Degree Costs):")
    print(vertices_removal_results_andc)
    plotting.plot_experiment_results(data=vertices_removal_results_andc,
                                     xlabel='Removed Vertices',
                                     ylabel='|Inf[G\',S]|',
                                     title='Vertices Removal Experiment Results (Average Neighbor Degree Costs)',
                                     save_path=save_path)

    print('\n====================================================================\n')


if __name__ == '__main__':
   main()
