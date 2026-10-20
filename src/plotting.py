from datetime import datetime
import os

import matplotlib.pyplot as plt

timestamp = datetime.now().strftime("%Y-%m-%d at %H-%M-%S")

def plot_experiment_results(data, xlabel="xlabel", ylabel="ylabel",
                            title="title", figsize=(10, 6),
                            save_path=None):
    """
    Plots experiment results structured as {name: {x_value: y_value}}.

    Parameters:
    - data (dict): Nested dictionary of results {strategy_name: {x_val: y_val}}.
    - xlabel (str): Label for the X-axis.
    - ylabel (str): Label for the Y-axis.
    - title (str): Title of the plot.
    - figsize (tuple): Figure size dimensions (width, height).
    - save_path (str, optional): File path to save the plot (e.g., 'experiment_plot.png').
    """
    plt.figure(figsize=figsize)

    for name, results in data.items():
        # Sort keys (e.g., budgets) to ensure the line flows sequentially from left to right
        sorted_x = sorted(results.keys())
        sorted_y = [results[x] for x in sorted_x]

        plt.plot(sorted_x, sorted_y, marker='o', markersize=4, label=name, linewidth=2)

    # Styling
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    # Optional file export
    if save_path:
        filename = f"{title} {timestamp}.png"
        save_file_path = os.path.join(save_path, filename)
        plt.savefig(save_file_path, dpi=300, bbox_inches='tight')
        print(f"Plot \"{title}.png\" successfully saved to {save_path}")

    plt.show()