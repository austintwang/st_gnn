import os
import random
import pickle
import numpy as np 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import snap

def load_graph(graph_dir):
    meta_path = os.path.join(graph_dir, "meta.pickle")
    with open(meta_path, "rb") as in_file:
        data = pickle.load(in_file)
    graph_path = os.path.join(graph_dir, "graph.txt")
    graph = snap.LoadEdgeList(snap.TUNGraph, graph_path)
    data["graph"] = graph
    return data

def get_egonet(graph, node, hop):
    egonet = graph.GetEgonetHop(node, hop)
    for i in egonet.Edges():
        print(i) ####
    # print(i for i in egonet.Edges()) ####
    nbrhood = nx.Graph(list(egonet.Edges()))
    return nbrhood

def get_annotations(subgraph, cells, cell_pos):
    nodelist = list(subgraph.nodes)
    xs = []
    ys = []
    zs = []
    for n in nodelist:
        cell = cells[n]
        x, y, z = cell_pos[cell]
        xs.append(x)
        ys.append(y)
        zs.append(z)

    xs = np.array(xs)
    ys = np.array(ys)
    xs -= np.min(xs)
    ys -= np.min(ys)
    xs /= np.max(xs)
    ys /= np.max(ys)

    pos = {}
    for n, x, y in zip(nodelist, xs, ys):
        pos[n] = (x, y)

    return nodelist, pos, zs

def plot_egonet(subgraph, nodelist, pos, z, title, result_path):
    nx.draw_networkx(subgraph, pos=pos, nodelist=nodelist, node_color=z)
    plt.title(title)
    plt.savefig(result_path, bbox_inches='tight')
    plt.clf()

def make_subgraphs(num_subgraphs, radii, hop, in_dir, out_dir):
    graphs = {}
    for r in radii:
        graph_dir = os.path.join(in_dir, f"adj_r_{r}")
        in_data = load_graph(graph_dir)
        graphs[r] = in_data
        # cells_ref = in_data["cells"]

    rnd = snap.TRnd(42)
    rnd.Randomize()
    graph_ref = in_data[min(radii)]

    for i in range(num_subgraphs):
        # ref = random.choice(cells_ref)
        ref = graph_ref["cells"][graph_ref["graph"].GetRndNId(rnd)]
        for r, graph_data in graphs.items():
            graph = graph_data["graph"]
            cell_map = graph_data["cell_map"]
            cells = graph_data["cells"]
            cell_pos = graph_data["cell_pos"]

            node = cell_map[ref]
            subgraph = get_egonet(graph, node, hop)
            nodelist, pos, z = get_annotations(subgraph, cells, cell_pos)

            out_path = os.path.join(out_dir, f"sub_{i}_r_{r}.svg")
            title = f"{r} μm Radius, {hop}-Hop, {ref}"
            plot_egonet(subgraph, nodelist, pos, z, title, out_path)


if __name__ == '__main__':
    num_subgraphs = 5
    radii = [10, 100, 1000]
    hop = 5

    data_path = "/dfs/user/atwang/data/spt_zhuang/"
    in_dir = os.path.join(data_path, "parsed", "adj_graphs_small", "mouse1")

    results_path = "/dfs/user/atwang/results/st_gnn_results"
    out_dir = os.path.join(results_path, "spt_zhuang", "adj_graphs", "subgraphs")

    make_subgraphs(num_subgraphs, radii, hop, in_dir, out_dir)

