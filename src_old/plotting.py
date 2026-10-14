import networkx as nx
from matplotlib import pyplot as plt

def plot_diffusion_comparison(snapGraph, seed_set, final_active_nodes):
    # 1. Convertiamo il grafo SNAP in NetworkX per poterlo disegnare facilmente
    nx_g = nx.Graph()
    for NI in snapGraph.Nodes():
        nx_g.add_node(NI.GetId())
    for EI in snapGraph.Edges():
        nx_g.add_edge(EI.GetSrcNId(), EI.GetDstNId())

    # Fissiamo il layout (seed=42) così la posizione dei nodi sarà IDENTICA nei due grafici
    pos = nx.spring_layout(nx_g, seed=42) #

    # Creiamo una figura con due sotto-grafici affiancati
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    # --- PLOT 1: Solo i Seed Set ---
    # Rosso per i seed, azzurro chiaro per tutti gli altri
    colors_seed = ['red' if node in seed_set else 'lightblue' for node in nx_g.nodes()]
    nx.draw(nx_g, pos, ax=ax1, with_labels=True, node_color=colors_seed, node_size=500, font_size=8)
    ax1.set_title("1. Seed Set Selezionati (Greedy)")

    # --- PLOT 2: Diffusione Finale (Cascata) ---
    # Rosso = Seed, Verde chiaro = Attivati dalla cascata, Grigio = Inattivi
    def get_color(node):
        if node in seed_set:
            return 'red'
        elif node in final_active_nodes:
            return 'lightgreen'
        else:
            return 'lightgray'

    colors_cascade = [get_color(node) for node in nx_g.nodes()]
    nx.draw(nx_g, pos, ax=ax2, with_labels=True, node_color=colors_cascade, node_size=500, font_size=8)
    ax2.set_title(f"2. Diffusione Finale (Totale attivi: {len(final_active_nodes)})")

    plt.tight_layout()
    plt.show()


def plot_diffusion_costs(snapGraph, seed_set, final_active_nodes, costsNode):
    # 1. Convertiamo il grafo SNAP in NetworkX
    nx_g = nx.Graph()
    for NI in snapGraph.Nodes():
        nx_g.add_node(NI.GetId())
    for EI in snapGraph.Edges():
        nx_g.add_edge(EI.GetSrcNId(), EI.GetDstNId())

    # Fissiamo il layout (seed=42)
    pos = nx.spring_layout(nx_g, seed=42)

    # 2. CREIAMO IL DIZIONARIO DELLE ETICHETTE CON I COSTI
    node_labels = {node: costsNode.get(node, 0) for node in nx_g.nodes()}

    # Creiamo una figura con due sotto-grafici affiancati
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    # --- PLOT 1: Solo i Seed Set ---
    colors_seed = ['red' if node in seed_set else 'lightblue' for node in nx_g.nodes()]
    # NOTA: Aggiunto "labels=node_labels"
    nx.draw(nx_g, pos, ax=ax1, with_labels=True, labels=node_labels, node_color=colors_seed, node_size=600, font_size=8)
    ax1.set_title("1. Seed Set Selezionati (Costi)")

    # --- PLOT 2: Diffusione Finale (Cascata) ---
    def get_color(node):
        if node in seed_set:
            return 'red'
        elif node in final_active_nodes:
            return 'lightgreen'
        else:
            return 'lightgray'

    colors_cascade = [get_color(node) for node in nx_g.nodes()]
    # NOTA: Aggiunto "labels=node_labels" anche qui
    nx.draw(nx_g, pos, ax=ax2, with_labels=True, labels=node_labels, node_color=colors_cascade, node_size=600, font_size=8)
    ax2.set_title(f"2. Diffusione Finale - Totale attivi: {len(final_active_nodes)} (Costi)")

    plt.tight_layout()
    plt.show()