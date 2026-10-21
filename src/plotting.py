"""
Plotting utilities.

Provides a function that plots experiment results (one line per algorithm/cost
combination) and optionally saves the figure as a PNG inside a folder named with the
timestamp of the run.
"""

from datetime import datetime
import os

import matplotlib.pyplot as plt

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def plot_experiment_results(data, xlabel="xlabel", ylabel="ylabel", title="title", figsize=(10, 6), save_path=None):
    """
    Function: plot_experiment_results

    Plots one line per experiment result series, optionally saving the figure as PNG in a
    timestamped folder.

    Args:
        data (dict): Maps a label (str) to a dict {x: y}.
        xlabel (str, optional): X-axis label.
        ylabel (str, optional): Y-axis label.
        title (str, optional): Plot title, also used as file name.
        figsize (tuple, optional): Figure size. Default (10, 6).
        save_path (str, optional): Base directory where the plot is saved; not saved if None.

    Returns:
        None: The plot is shown (and possibly saved).
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
        timestamp_directory = os.path.join(save_path, timestamp)
        os.makedirs(timestamp_directory, exist_ok=True)

        filename = f"{title}.png"
        save_file_path = os.path.join(timestamp_directory, filename)
        plt.savefig(save_file_path, dpi=300, bbox_inches='tight')
        print(f"Plot \"{filename}\" successfully saved to {timestamp_directory}")

    plt.show()
