# Majority Dynamics in Online Network

Project for the **Social Network** course — University of Salerno, academic year 2025/26.

This project implements and compares several algorithms that select a **seed set** of nodes in a social network under a **budget constraint**, where every node has a cost. The quality of a seed set is measured by simulating a **majority-threshold cascade** and counting how many nodes end up influenced. The robustness of each seed set is also tested against edge and vertex failures.


## Table of Contents

- [Problem Statement](#problem-statement)
- [Implemented Algorithms](#implemented-algorithms)
- [Cost Functions](#cost-functions)
- [Experiments](#experiments)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)


## Problem Statement

Given an undirected graph `G = (V, E)`, a cost `c(v)` for every node and a budget `B`, find a seed set `S ⊆ V` with total cost `Σ c(v) ≤ B` that maximizes the number of nodes eventually influenced.

**Diffusion model (majority threshold).** A node `v` becomes active as soon as at least `⌈deg(v) / 2⌉` of its neighbors are active. Starting from the seed set, activations are applied in synchronous rounds until no new node becomes active. The size of the final active set is denoted `|Inf[G, S]|`.


## Implemented Algorithms

All algorithms share the same interface, `f(graph, costs, budget) -> set of node IDs`, so they can be swapped freely in the experiments.

### 1. Cost-Seeds-Greedy

Greedy algorithm taken from the course slides. Starting from an empty set, it repeatedly adds the vertex with the highest **cost-effectiveness** (marginal gain of a target function divided by the vertex cost) until the budget is reached; the vertex that makes the cost reach or exceed the budget is discarded. The algorithm is run with three target functions and the **largest resulting seed set** is returned.

Three implementations are provided, each one an optimization of the previous one and all returning equivalent seed sets:

| File | Description |
|---|---|
| `cost_seeds_greedy_slow.py` | Reference implementation, a direct translation of the slides: the target function is re-evaluated on the whole graph for every candidate at every step. |
| `cost_seeds_greedy_fast.py` | Computes the marginal gain **locally** from the candidate's neighbors (using a count of already-seeded neighbors) instead of re-evaluating the whole function. Every step still scans all candidates. |
| `cost_seeds_greedy_lazy.py` | Adds **lazy evaluation** on top of the fast version: candidates are kept in a priority queue and a score is recomputed only when it is stale. This works because marginal gains never increase as the seed set grows (diminishing returns). **This is the version used by default in the experiments.** |

### 2. WTSS — Weighted Target Set Selection

Heuristic that iteratively removes vertices from the graph, using majority thresholds `⌈deg(v)/2⌉`:

1. **Case 1** — a vertex with residual threshold 0 is removed and treated as activated (its neighbors' thresholds decrease).
2. **Case 2** — a vertex whose current degree is lower than its residual threshold cannot be activated by its neighbors: it becomes a **seed candidate**.
3. **Case 3** — otherwise, remove the vertex with the best greedy score `cost · threshold / (deg · (deg + 1))` (picked through a lazily-invalidated heap).

The candidates are finally filtered to respect the budget.

### 3. Centrality / Cost Greedy (baseline)

Ranks every node by `degree centrality / cost` and adds nodes in that order whenever they fit in the remaining budget.


## Cost Functions

Three ways of assigning an integer cost to the nodes (`src/costs_functions.py`):

| Name | Cost of node `v` |
|---|---|
| **Random** | Uniform random integer in `[1, 10]` |
| **Degree** | `⌈deg(v) / 2⌉` |
| **Average neighbor degree** | `max(1, ⌈average degree of v's neighbors⌉)` |


## Experiments

Each experiment is run for every combination of **cost function × algorithm** (3 × 3 = 9 curves per cost type, 9 plots in total).

1. **Budget variation** (`budget_variation.py`) — The budget grows in about 50 steps from 0 to the total cost of all nodes. For each budget the seed set is computed and the cascade is simulated. Stops early when every node is influenced.
2. **Edges removal** (`edges_removal.py`) — The seed set is computed **once** on the original graph. Then edges are removed in random order from a copy of the graph (about 50 checkpoints) and the influence of the fixed seed set is measured after each checkpoint.
3. **Vertices removal** (`vertices_removal.py`) — The seed set is computed once on the original graph. Then a random vertex **together with all its neighbors** is repeatedly removed from a copy of the graph; the influence of the surviving seeds is measured after every step.

For experiments 2 and 3 the default budget is `20% of |V| × average random cost`, unless specified with `-b`.


## Project Structure

```
Social Network Project/
├── main.py                       # Sequential entry point
├── main_parallel.py              # Parallel entry point (9 experiments in a process pool)
├── data/                         # Input graphs (edge lists)
├── results/                      # Generated plots (one sub-folder per run, named by timestamp)
└── src/
    ├── costs_functions.py        # Random / degree / avg-neighbor-degree costs
    ├── graph_utils.py            # Graph loading, plotting, random connected subgraph
    ├── plotting.py               # Experiment plots (shown and saved as PNG)
    ├── simulate_cascade.py       # Majority-threshold cascade simulation
    ├── experiments/
    │   ├── run_experiment.py     # Runs one experiment for every (costs, algorithm) pair
    │   ├── budget_variation.py
    │   ├── edges_removal.py
    │   └── vertices_removal.py
    └── seedset/
        ├── cost_seeds_greedy_slow.py
        ├── cost_seeds_greedy_fast.py
        ├── cost_seeds_greedy_lazy.py
        ├── centrality_cost_seeds_greedy.py
        └── wtss.py
```


## Installation

**Requirements**

- Python 3.9.X (later version cannot be used because `snap-stanford` isn't compatible)
- [`snap-stanford`](https://snap.stanford.edu/snappy/) (SNAP for Python)
- `networkx`
- `matplotlib`

```bash
git clone <repository-url>
cd "Social Network Project"
pip install snap-stanford networkx matplotlib
```

> `snap-stanford` ships prebuilt wheels only for some Python versions and platforms. If the installation fails, try an older Python version or check the [SNAP.py documentation](https://snap.stanford.edu/snappy/).


## Usage

Both scripts must be launched from the project root, because the code imports from the `src` package.

### Sequential version

```bash
python main.py -d data/<graph>.txt
```

### Parallel version

The nine experiments (3 types × 3 cost functions) are independent, so `main_parallel.py` distributes them over a pool of worker processes. Since SNAP graphs cannot be pickled, each worker loads the graph from file and builds its own cost dictionaries.

```bash
python main_parallel.py -d data/<graph>.txt -w 4
```

### Command-line arguments

| Argument | Description | Default |
|---|---|---|
| `-d`, `--dataset` | Path of the graph edge list to load | *(required)* |
| `--test` | Run on a random **connected subgraph with 5% of the nodes**, useful for quick runs | off |
| `-b`, `--budget` | Seed-selection budget | 20% of average cost |
| `-s`, `--save-path` | Folder where plots are saved | `./results` |
| `-w`, `--workers` | Max number of worker processes *(parallel version only)* | `9` |

### Examples

```bash
# Quick test on a small subgraph
python main.py -d data/graph.txt --test

# Full run with a fixed budget, saving plots in a custom folder
python main.py -d data/graph.txt -b 500 -s my_results

# Parallel test run with 4 workers
python main_parallel.py -d data/graph.txt --test -w 4
```

### Choosing the Cost-Seeds-Greedy implementation

The lazy version is imported by default. To use another one, change the import at the top of `main.py`:

```python
from src.seedset.cost_seeds_greedy_lazy import cost_seeds_greedy
# from src.seedset.cost_seeds_greedy_slow import cost_seeds_greedy
# from src.seedset.cost_seeds_greedy_fast import cost_seeds_greedy
```
