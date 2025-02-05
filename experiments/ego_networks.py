import pandas as pd
import os
import numpy as np
import torch
from time import time
import networkx as nx
import matplotlib.pyplot as plt

import sys
sys.path.append('../')

from preordering_problem.ilp_solver import Preorder
from preordering_problem.greedy_arc_insertion import greedy_arc_insertion
from preordering_problem.di_cut_approximation import greedy_di_cut
from preordering_problem.drawing import PreorderPlot
from preordering_problem.decompose_preorder import decompose_preorder


"""
Preordering Social networks from https://snap.stanford.edu/data/ego-Twitter.html and
https://snap.stanford.edu/data/ego-Gplus.html
"""


data_root = "../data"
t_limit = 500
max_num_nodes = np.inf


def load_ego_network(platform: str, ego_id: int):
    """
    Load edges of social network
    :param platform:
    :param ego_id:
    :return: list of pairs
    """
    edges = []
    with open(f"{data_root}/{platform}/{ego_id}.edges", "r") as f:
        for line in f.readlines():
            i, j = line.strip().split(" ")
            edges.append((int(i), int(j)))
    return edges


def edges_to_adjacency(edges: list) -> (np.ndarray, list):
    """
    Convert edge list to adjacency matrix
    :param edges: list of pairs
    :return: np.ndarray, list
    """
    id2node = {}
    node2id = []
    counter = 0
    for e in edges:
        for i in e:
            if i not in id2node:
                id2node[i] = counter
                node2id.append(i)
                counter += 1
    n = len(id2node)
    adjacency = np.eye(n, dtype=int)
    for i, j in edges:
        adjacency[id2node[i], id2node[j]] = 1
    return adjacency, node2id


def get_ego_ids_by_num_nodes(platform):
    ego_ids = []
    num_nodes = []
    for filename in os.listdir(f"{data_root}/{platform}"):
        ego_id, data_type = filename.strip().split(".")
        if data_type != "edges":
            continue
        ego_id = int(ego_id)
        ego_ids.append(ego_id)
        edges = load_ego_network(platform, ego_id)
        adjacency, node2id = edges_to_adjacency(edges)
        num_nodes.append(adjacency.shape[0])

    return sorted(zip(num_nodes, ego_ids))

def create_dataframe(platform):
    filename = f"results/{platform}.csv"
    if os.path.exists(filename):
        print(f"File '{filename}' already exists")
        return

    ego_ids_and_num = get_ego_ids_by_num_nodes(platform)
    ego_ids_and_num = [(num, ego_id) for num, ego_id in ego_ids_and_num if num <= max_num_nodes]
    ego_ids = [ego_id for num, ego_id in ego_ids_and_num]
    ilp_algorithms = ["Preordering ILP", "Clustering ILP", "Partial Ordering ILP", "Successive ILPs"]
    other_algorithms = ["LP", "LP+OCW", "GDC", "GAI", "GDC+GAI"]
    columns = ["|V|", "|E|"]
    for ilp in ilp_algorithms:
        columns += [ilp, f"{ilp} Gap", f"{ilp} T"]
    for algo in other_algorithms:
        columns += [algo, f"{algo} T"]
    df = pd.DataFrame(columns=columns, index=ego_ids)

    print(df.columns)

    for num, ego_id in ego_ids_and_num:
        df.loc[ego_id, "|V|"] = num
        df.loc[ego_id, "|E|"] = len(load_ego_network(platform, ego_id))
    df.to_csv(filename)
    print("Created dataframe")


def solve_preorder_ilp(platform, method="ILP"):
    print("Solving", method)
    filename = f"results/{platform}.csv"
    df = pd.read_csv(filename, index_col=0)

    for i, ego_id in enumerate(df.index):
        if not np.isnan(df.loc[ego_id, method]) and (df.loc[ego_id, f"{method} Gap"] < 1e-3 or
                                                     df.loc[ego_id, f"{method} T"] >= t_limit):
            continue
        if method != "Perodering" and df.loc[ego_id, "Preordering ILP Gap"] > 1e-3:
            continue

        if method == "Successive ILPs" and not df.loc[ego_id, "Clustering ILP Gap"] <= 1e-3:
            continue

        print("\r", i, ego_id, df.loc[ego_id, "|V|"], end=" ")

        edges = load_ego_network(platform, ego_id)
        adjacency, nodes = edges_to_adjacency(edges)
        cost = - np.ones_like(adjacency) + 2*adjacency
        cost[np.diag_indices_from(cost)] = 0

        if method == "Successive ILPs":
            t_0 = time()
            # Step 1: solve clustering
            clustering = Preorder(cost, binary=True, suppress_log=True)
            clustering.add_symmetric_constraints()
            cluster_obj = clustering.solve()
            if clustering.model.MIPGap > 1e-3:
                raise ValueError("Clustering not optimal!")
            _, clust = decompose_preorder(clustering.get_variable_values())
            # Step 2: contract clusters
            contracted_costs = np.zeros((len(clust), len(clust)), dtype=cost.dtype)
            for i in range(len(clust)):
                for j in range(len(clust)):
                    if i == j:
                        continue
                    contracted_costs[i, j] = cost[clust[i]][:, clust[j]].sum()
            # Step 3: solve partial ordering
            preorder = Preorder(contracted_costs, binary=True, suppress_log=True)
            preorder.add_anti_symmetric_constraints()
            preorder_obj = preorder.solve()
            if preorder.model.MIPGap > 1e-3:
                raise ValueError("Clustering not optimal!")
            obj = cluster_obj + preorder_obj
            t = time() - t_0
            gap = 0
        else:
            preorder = Preorder(cost, binary=True, suppress_log=True)
            if method == "Preordering ILP":
                gdc_obj, gdc_sol = greedy_di_cut(cost)
                gdc_ga_obj, gdc_ga_sol = greedy_arc_insertion(torch.Tensor(cost), torch.Tensor(gdc_sol))
                preorder.set_solution(gdc_ga_sol.numpy())
            elif method == "Clustering ILP":
                preorder.add_symmetric_constraints()
            elif method == "Partial Ordering ILP":
                preorder.add_anti_symmetric_constraints()
                gdc_obj, gdc_sol = greedy_di_cut(cost)
                preorder.set_solution(gdc_sol)
            else:
                raise ValueError(f"Unknown method: {method}")

            try:
                t_0 = time()
                obj = preorder.solve(time_limit=t_limit)
                t = time() - t_0
                gap = preorder.model.MIPGap
            except AttributeError as e:
                print(e)
                t = t_limit
                obj = 0
                gap = np.inf

        df.loc[ego_id, f"{method}"] = obj
        df.loc[ego_id, f"{method} T"] = t
        df.loc[ego_id, f"{method} Gap"] = gap
        print(obj, t, gap, end=" ")
        df.to_csv(filename)
    print()


def compute_lp_bounds(platform, ocw=False):
    print(f"Computing LP bounds (OCW = {ocw})")
    filename = f"results/{platform}.csv"
    method = "LP+OCW" if ocw else "LP"
    df = pd.read_csv(filename, index_col=0)
    for i, ego_id in enumerate(df.index):
        if not np.isnan(df.loc[ego_id, f"{method}"]):
            continue

        if method == "LP+OCW" and df.loc[ego_id, f"LP T"] >= t_limit:
            df.loc[ego_id, f"LP+OCW"] = df.loc[ego_id, f"LP"]
            df.loc[ego_id, f"LP+OCW T"] = df.loc[ego_id, f"LP T"]
            df.to_csv(filename)
            continue

        print("\r", i, ego_id, df.loc[ego_id, "|V|"], end=" ")

        edges = load_ego_network(platform, ego_id)
        adjacency, _ = edges_to_adjacency(edges)
        cost = - np.ones_like(adjacency) + 2 * adjacency
        cost[np.diag_indices_from(cost)] = 0

        preorder = Preorder(cost, binary=False, suppress_log=True)
        preorder.separate_odd_closed_walk = 9 if ocw else 0

        t_0 = time()
        obj = preorder.solve(t_limit)
        t = time() - t_0
        if obj == -np.inf and t > t_limit:
            obj = preorder.model.ObjBoundC
        if obj < 0:
            raise ValueError("Negative bound!!")
        df.loc[ego_id, f"{method}"] = obj
        df.loc[ego_id, f"{method} T"] = t
        print(obj, t, end="")
        df.to_csv(filename)
    print()


def solve_local_search(platform):
    filename = f"results/{platform}.csv"
    df = pd.read_csv(filename, index_col=0)

    print("Performing local search")

    for i, ego_id in enumerate(df.index):
        if df.loc[ego_id, "|V|"] == 0:
            continue

        if not np.any(df.loc[ego_id, ["GAI", "GDC", "GDC+GAI"]].isna()):
            continue

        print("\r", i, ego_id, end=" ")

        edges = load_ego_network(platform, ego_id)
        adjacency, _ = edges_to_adjacency(edges)
        costs = - np.ones_like(adjacency) + 2 * adjacency
        costs[np.diag_indices_from(costs)] = 0
        costs_torch = torch.Tensor(costs).to("cuda")

        if np.isnan(df.loc[ego_id, "GAI"]):
            t_0 = time()
            ga_obj, _ = greedy_arc_insertion(costs_torch)
            t = time() - t_0
            ga_obj = int(ga_obj.item())
            df.loc[ego_id, "GAI"] = ga_obj
            df.loc[ego_id, "GAI T"] = t
            print("GAI", ga_obj, t, end=" ")

        if np.isnan(df.loc[ego_id, "GDC"]):
            t_0 = time()
            gdc_obj, gdc_adjacency = greedy_di_cut(costs)
            t = time() - t_0
            df.loc[ego_id, "GDC T"] = t
            gdc_obj = int(gdc_obj)
            df.loc[ego_id, "GDC"] = gdc_obj
            print("GDC", gdc_obj, t, end=" ")

        if np.isnan(df.loc[ego_id, "GDC+GAI"]):
            t_0 = time()
            gdc_obj, gdc_adjacency = greedy_di_cut(costs)
            gdc_adjacency = torch.Tensor(gdc_adjacency).to("cuda")
            combined_obj, adjacency = greedy_arc_insertion(costs_torch, gdc_adjacency)
            t = time() - t_0
            combined_obj = int(combined_obj.item())
            df.loc[ego_id, "GDC+GAI"] = combined_obj
            df.loc[ego_id, "GDC+GAI T"] = t
            print("GDC", combined_obj, t, end=" ")

        df.to_csv(filename)
    print()


def plot_ego_network_results(platform, ego_id):
    edges = load_ego_network(platform, ego_id)
    adjacency, _ = edges_to_adjacency(edges)
    cost = - np.ones_like(adjacency) + 2 * adjacency
    cost[np.diag_indices_from(cost)] = 0

    results = []
    preorder = Preorder(cost, binary=True, suppress_log=True)
    preorder.solve()
    results.append(preorder.get_variable_values())

    clustering = Preorder(cost, binary=True, suppress_log=True)
    clustering.add_symmetric_constraints()
    clustering.solve()
    results.append(clustering.get_variable_values())

    partial_order = Preorder(cost, binary=True, suppress_log=True)
    partial_order.add_anti_symmetric_constraints()
    partial_order.solve()
    results.append(partial_order.get_variable_values())

    g = nx.DiGraph()
    for i, j in np.argwhere(adjacency == 1):
        if i != j:
            g.add_edge(i, j)
    pos = nx.nx_pydot.graphviz_layout(g, prog="dot")

    fig, ax = plt.subplots(1, 4, figsize=(20, 4))
    nx.draw(g, pos=pos, ax=ax[0], with_labels=True, node_color="white", edgecolors="black")

    plotters = []
    for i, adj in enumerate(results):
        plotter = PreorderPlot(adj, cost, ax=ax[i+1])
        plotter.plot()
        plotters.append(plotter)
    for a in ax:
        a.set_axis_off()
    fig.tight_layout()

    plt.show()


def plot_closed_gap(platform):
    filename = f"results/{platform}.csv"
    df = pd.read_csv(filename, index_col=0)

    opt_idx = df["Preordering ILP Gap"] < 1e-3
    idx = opt_idx & ((df["Preordering ILP"] - df["LP"]).abs() > 1e-3)
    df = df.loc[idx]

    closed_gap = (df["LP"] - df[f"LP+OCW"]) / (df["LP"] - df["Preordering ILP"]) * 100
    print(f"Num LP+OCW results:", (~np.isnan(closed_gap)).sum())
    mean = np.nanmean(closed_gap)
    print("Mean closed gap:", mean)

    fig, ax = plt.subplots(figsize=(5, 2))
    hist = ax.hist(closed_gap, bins=np.linspace(0, 100, 11))[0]
    ax.plot([mean, mean], [0, hist.max() + 5], color="tab:red")
    ax.set_xlabel("Closed Gap (%)")
    ax.set_ylabel("Count")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, hist.max() + 5)
    fig.tight_layout()
    plt.show()


def plot_clustering_vs_ordering_ilp(platform):
    df = pd.read_csv(f"results/{platform}.csv", index_col=0)
    cols = ["Preordering ILP", "Partial Ordering ILP", "Clustering ILP", "Successive ILPs"]

    idx = (df[[f"{col} Gap" for col in cols]] <= 1e-3).all(axis=1)
    df = df.loc[idx]

    print("Means:")
    print(df.mean())
    relative = df[cols] / df[["|E|"]].values
    print("Relative Objectives Means:")
    print(relative.mean())

    markers = ["+", "x", "1", "3"]
    color = ["tab:blue", "tab:red", "tab:green", "tab:orange"]
    fig, ax = plt.subplots(1, 2, figsize=(9, 2.5))
    for col, m, c in zip(cols, markers, color):
        ax[0].scatter(df["|V|"], df[f"{col} T"], label=f"{col}", alpha=0.3, color=c, marker=m)

    ax[0].set_xlabel(r"$|V|$")
    ax[0].set_ylabel("Runtime [s]")
    ax[0].set_yscale('log')
    leg = ax[0].legend()
    for lh in leg.legend_handles:
        lh.set_alpha(1)

    bplot = ax[1].boxplot(relative, tick_labels=[col.split(" ")[0] for col in cols], widths=0.8)
    for median, flyer, c, m in zip(bplot["medians"], bplot["fliers"], color, markers):
        median.set_color(c)
        flyer.set_markeredgecolor(c)
        flyer.set_marker(m)
    ax[1].set_ylabel(r"Objective / $|E|$")

    fig.tight_layout()
    plt.show()


def plot_lower_upper_bound(platform):
    df = pd.read_csv(f"results/{platform}.csv", index_col=0)
    upper = "LP"
    columns = ["GDC", "GAI", "GDC+GAI"]
    bound = df[["LP", upper]].min(axis=1)
    gaps = -(df[columns] - bound.values.reshape((-1, 1))) / df[["|E|"]].values
    print("Num NANs:")
    print(gaps.isna().sum())

    fig, ax = plt.subplots(3, figsize=(6, 5), sharex=True, sharey=True)
    lim = 0.3
    bins = np.linspace(0, lim, int(lim * 100) + 1)

    means = gaps.mean()
    print("Mean gaps:")
    print(means)

    max_val = 0
    for i, col in enumerate(columns):
        hist = ax[i].hist(gaps[col], bins=bins)[0].max()
        max_val = max(max_val, hist.max())
        ax[i].set_ylabel("Count")
        ax[i].set_title(col)
    for i, col in enumerate(columns):
        ax[i].plot([means[col], means[col]], [0, max_val*1.05], color="tab:red")

    ax[-1].set_xlim(0, lim)
    ax[-1].set_ylim(0, max_val*1.05)
    ax[-1].set_xlabel("T Gap")
    fig.tight_layout()
    plt.show()


def plot_local_search_runtime(platform):
    df = pd.read_csv(f"results/{platform}.csv")
    fig, ax = plt.subplots(1, figsize=(6, 3), sharex=True)
    markers = ["+", "x", "1", "x"]
    names = ["GDC", "GAI", "GDC+GAI"]
    color = ["tab:blue", "tab:red", "tab:green"]
    for i, name in enumerate(names):
        ax.scatter(df["|V|"]**2, df[f"{name} T"], label=name, alpha=0.5, marker=markers[i], color=color[i])
    ax.set_xscale("log")
    ax.legend()
    ax.set_yscale("log")
    ax.set_xlabel(r"$|V_2|$")
    ax.set_ylabel(r"time [s]")
    fig.tight_layout()

    transitivity_idx = df[names[:3]] / df[["|E|"]].values
    print("Mean transitivity index")
    print(transitivity_idx.mean())

    plt.show()


def main(platform):
    """
    Comment out some of the lines below to nur run all algorithms!
    """
    create_dataframe(platform)
    solve_preorder_ilp(platform, "Preordering ILP")
    solve_preorder_ilp(platform, "Clustering ILP")
    solve_preorder_ilp(platform, "Partial Ordering ILP")
    solve_preorder_ilp(platform, "Successive ILPs")
    compute_lp_bounds(platform, False)
    compute_lp_bounds(platform, True)
    solve_local_search(platform)

    plot_local_search_runtime(platform)
    plot_lower_upper_bound(platform)
    plot_clustering_vs_ordering_ilp(platform)
    plot_closed_gap(platform)


if __name__ == "__main__":
    if not os.path.isdir("results"):
        os.mkdir("results")
    # Run experiments
    max_num_nodes = 40
    main("twitter")
    # Plot results of one single network
    plot_ego_network_results("twitter", 215824411)
