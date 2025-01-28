import pandas as pd
import os
import numpy as np
import torch
from time import time
import networkx as nx
import matplotlib.pyplot as plt

from preordering_problem.ilp_solver import Preorder
from preordering_problem.greedy_arc_insertion import greedy_arc_insertion
from preordering_problem.di_cut_approximation import greedy_di_cut
from preordering_problem.drawing import PreorderPlot


data_root = "../../../../datasets/snap-stanford"
t_limit = 500


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


def get_ego_ids_by_num_nodes(platform: str = "twitter"):
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
    filename = f"{platform}_results.csv"
    if os.path.exists(filename):
        print(f"File '{filename}' already exists")
        return

    ego_ids_and_num = get_ego_ids_by_num_nodes(platform)
    ego_ids = [ego_id for num, ego_id in ego_ids_and_num]
    ilp_algorithms = ["Preordering ILP", "Clustering ILP", "Partial Ordering ILP"]
    other_algorithms = ["LP", "LP+OCW", "GDC", "GAI", "GDC+GAI"]
    columns = ["|V|", "|E|"]
    for ilp in ilp_algorithms:
        columns += [ilp, f"{ilp} Gap", f"{ilp} T"]
    for algo in other_algorithms:
        columns += [algo, f"{algo} T"]
    df = pd.DataFrame(columns=columns, index=ego_ids)

    for num, ego_id in ego_ids_and_num:
        df.loc[ego_id, "|V|"] = num
        df.loc[ego_id, "|E|"] = len(load_ego_network(platform, ego_id))
    df.to_csv(filename)


def solve_preorder_ilp(platform, method="ILP"):

    filename = f"{platform}_results.csv"
    df = pd.read_csv(filename, index_col=0)

    for i, ego_id in enumerate(df.index):
        if not np.isnan(df.loc[ego_id, method]) and (df.loc[ego_id, f"{method} Gap"] < 1e-3 or
                                                     df.loc[ego_id, f"{method} T"] >= t_limit):
            print(f"{method} solution for {i} {ego_id} already exists")
            continue

        print(i, ego_id, df.loc[ego_id, "|V|"], end=" ")

        edges = load_ego_network("twitter", ego_id)
        adjacency, nodes = edges_to_adjacency(edges)
        cost = - np.ones_like(adjacency) + 2*adjacency
        cost[np.diag_indices_from(cost)] = 0

        preorder = Preorder(cost, binary=True, suppress_log=True, lazy=True)

        if method == "Preordering ILP":
            gdc_obj, gdc_sol = greedy_di_cut(cost)
            gdc_ga_obj, gdc_ga_sol = greedy_arc_insertion(torch.Tensor(cost), torch.Tensor(gdc_sol))
            preorder.set_solution(gdc_ga_sol.numpy())
        elif method == "Clustering ILP":
            preorder.add_symmetric_constraints()
        elif method == "Partial Ordering ILP":
            preorder.add_symmetric_constraints()
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
        print(obj, t, gap)
        df.to_csv(filename)


def compute_lp_bounds(platform, ocw=False):
    filename = f"{platform}_results.csv"
    method = "LP+OCW" if ocw else "LP"
    df = pd.read_csv(filename, index_col=0)
    for i, ego_id in enumerate(df.index):
        if not np.isnan(df.loc[ego_id, f"{method}"]):
            print(f"{method} for {ego_id} already computed.")
            continue
        print(i, ego_id, df.loc[ego_id, "|V|"], end=" ")

        edges = load_ego_network("twitter", ego_id)
        adjacency, _ = edges_to_adjacency(edges)
        cost = - np.ones_like(adjacency) + 2 * adjacency
        cost[np.diag_indices_from(cost)] = 0

        preorder = Preorder(cost, binary=False, suppress_log=True, lazy=True)
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
        print(obj, t)
        df.to_csv(filename)


def solve_local_search(platform):
    filename = f"{platform}_results.csv"
    df = pd.read_csv(filename, index_col=0)

    for i, ego_id in enumerate(df.index):
        if df.loc[ego_id, "|V|"] == 0:
            continue

        print(i, ego_id)

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

        if np.isnan(df.loc[ego_id, "GDC"]):
            t_0 = time()
            gdc_obj, gdc_adjacency = greedy_di_cut(costs)
            t = time() - t_0
            df.loc[ego_id, "GDC T"] = t
            gdc_obj = int(gdc_obj)
            df.loc[ego_id, "GDC"] = gdc_obj

        if np.isnan(df.loc[ego_id, "GDC+GAI"]):
            t_0 = time()
            gdc_obj, gdc_adjacency = greedy_di_cut(costs)
            gdc_adjacency = torch.Tensor(gdc_adjacency).to("cuda")
            combined_obj, adjacency = greedy_arc_insertion(costs_torch, gdc_adjacency)
            t = time() - t_0
            df.loc[ego_id, "GDC+GAI"] = int(combined_obj.item())
            df.loc[ego_id, "GDC+GAI T"] = t

        df.to_csv(filename)


def plot_ego_network_results(platform, ego_id):
    edges = load_ego_network(platform, ego_id)
    adjacency, _ = edges_to_adjacency(edges)
    cost = - np.ones_like(adjacency) + 2 * adjacency
    cost[np.diag_indices_from(cost)] = 0

    results = []
    preorder = Preorder(cost, binary=True, lazy=True, suppress_log=True)
    preorder.solve()
    results.append(preorder.get_variable_values())

    clustering = Preorder(cost, binary=True, lazy=True, suppress_log=True)
    clustering.add_symmetric_constraints()
    clustering.solve()
    results.append(clustering.get_variable_values())

    partial_order = Preorder(cost, binary=True, lazy=True, suppress_log=True)
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
    filename = f"{platform}_results.csv"
    df = pd.read_csv(filename, index_col=0)

    opt_idx = df["Preordering ILP Gap"] < 1e-3
    idx = opt_idx & ((df["Preordering ILP"] - df["LP"]).abs() > 1e-3)
    df = df.loc[idx]

    closed_gap = (df["LP"] - df[f"LP+OCW"]) / (df["LP"] - df["Preordering ILP Gap"]) * 100
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
    fig.savefig("closed-gap.png", dpi=300)
    plt.show()


def main():
    """
    Un-comment the lines below to run the specific algorithms
    :return:
    """
    platform = "twitter"
    create_dataframe(platform)
    solve_preorder_ilp(platform, "Preordering ILP")
    # solve_preorder_ilp(platform, "Clustering ILP")
    # solve_preorder_ilp(platform, "Partial Ordering ILP")
    compute_lp_bounds(platform, False)
    compute_lp_bounds(platform, True)
    # solve_local_search(platform)
    plot_closed_gap(platform)

    # plot_ego_network_results("twitter", 215824411)


if __name__ == "__main__":
    main()

