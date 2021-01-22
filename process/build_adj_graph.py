import os
import glob
import numpy as np 
import snap
import tqdm
import pickle

class CellTable(object):
    def __init__(self, bucket_size, radius, precision=6):
        self.bucket_size = bucket_size
        self.radius = radius
        self.radsq = radius**2
        self.scale = 10**precision
        self.bsize_scaled = self.bucket_size * self.scale

        self.buckets = {}
        self.cell_pos = {}

        # self.edges = set()
        self.graph = snap.TUNGraph.New()
        self.nodes = []
        self.node_map = {}
        self.counter = 0

    def add_cell(self, cell_id, x, y, z, report=False):
        # if cell_id in self.cell_pos: ####
        #     print(x,y,z)
        #     print(self.cell_pos[cell_id]) ####
        self.graph.AddNode(self.counter)
        self.node_map[cell_id] = self.counter
        self.nodes.append(cell_id)
        self.counter += 1

        coords = (x, y, z)
        self.cell_pos[cell_id] = coords
        x_lh, y_lh, z_lh = (int((i - self.radius) * self.scale) // self.bsize_scaled for i in coords)
        x_uh, y_uh, z_uh = (int((i + self.radius) * self.scale) // self.bsize_scaled for i in coords)

        # adj = set()
        for i in range(x_lh, x_uh + 1):
            for j in range(y_lh, y_uh + 1):
                for k in range(z_lh, z_uh + 1):
                    bidx = (i,j,k)
                    # print(bidx) ####
                    bucket = self.buckets.setdefault(bidx, [])
                    # if report:
                    #     print(len(bucket))
                    # print(bucket) ####
                    for c in bucket:
                        # print(len(c)) ####
                        xc, yc, zc = self.cell_pos[c]
                        if (xc - x)**2 + (yc - y)**2 + (zc - z)**2 <= self.radsq:
                            self.graph.AddEdge(self.node_map[cell_id], self.node_map[c])
                            # adj.add(c)

                    bucket.append(cell_id)

        # for c in adj:
        #     self.graph.AddEdge(self.nodes[cell_id], self.nodes[c])
            # self.graph.setdefault(c, set()).add(cell_id`1)
            # self.graph.setdefault(cell_id, set()).add(c)

    def save_graph(self, out_dir):
        meta = {"cells": self.nodes, "cell_map": self.node_map, "cell_pos": self.cell_pos}
        os.makedirs(out_dir, exist_ok=True)
        metadata_path = os.path.join(out_dir, "meta.pickle")
        with open(metadata_path, "wb") as out_file:
            pickle.dump(meta, out_file)
        graph_path = os.path.join(out_dir, "bin.graph")
        # self.graph.SaveEdgeList(graph_path)
        FOut = snap.TFOut(graph_path)
        self.graph.Save(FOut)
        FOut.Flush()


def parse_cell(line):
    entries = line.split("\"")
    if len(entries) == 1:
        cell_id, x, y, slice_id = line.split(",")
        x = float(x)
        y = float(y)
        z = int(slice_id.split("_")[1][5:]) * 10
        # print(x, y) ####
        return cell_id, x, y, z
        
    cell_id, x_b, _, y_b, slice_id = line.split("\"")
    cell_id = cell_id.rstrip(",")

    x1 = np.fromstring(x_b, sep=",")
    y1 = np.fromstring(y_b, sep=",")
    x2 = np.roll(x1, -1)
    y2 = np.roll(y1, -1)
    q = x1 * y2 - x2 * y1
    a = np.sum(q)
    if a == 0:
        x = np.mean((x1 + x2)) / 2
        y = np.mean((y1 + y2)) / 2
        # print(x1, y1) ####
    else:
        x = np.sum((x1 + x2) * q) / (3 * a)
        y = np.sum((y1 + y2) * q) / (3 * a)

    z = int(slice_id.split("_")[1][5:]) * 10

    # print(x,y,z) ####
    # print(x_b[0], x_b[-1]) ####
    # print(y_b[0], y_b[-1]) ####
    # if np.isnan(x) or np.isnan(y):
    #     print(x1) ####
    #     print(y1) ####
    return cell_id, x, y, z

def load_file(tables, in_path):
    with open(in_path, "rb") as f:
        lines = 0
        buf_size = 1024 * 1024
        buf = f.raw.read(buf_size)
        while buf:
            lines += buf.count(b'\n')
            buf = f.raw.read(buf_size)

    with open(in_path) as f:
        next(f)
        for line in tqdm.tqdm(f, desc=os.path.basename(in_path), total=lines):
            cell_id, x, y, z = parse_cell(line)
            for t in tables.values():
                t.add_cell(cell_id, x, y, z)
            # return ####

def build_graphs(params, in_paths, out_dir):
    tables = {}
    for radius, bucket_size in params:
        tables[radius] = CellTable(bucket_size, radius)

    for p in in_paths:
        load_file(tables, p)

    for k, v in tables.items():
        graph_dir = os.path.join(out_dir, f"adj_r_{k}")
        v.save_graph(graph_dir)

if __name__ == '__main__':
    params = [
        (10, 100),
        (100, 1000),
        (1000, 10000)
    ]

    data_path = "/dfs/user/atwang/data/spt_zhuang/"
    in_dir = os.path.join(data_path, "source", "processed_data")

    indiv = ["mouse1"]
    for i in indiv:
        out_dir = os.path.join(data_path, "parsed", "adj_graphs_small", i)
        in_paths = glob.glob(os.path.join(in_dir, f"segmented_cells_{i}sample*.csv"))[:1]
        # print(in_paths) ####
        build_graphs(params, in_paths, out_dir)

    indiv = ["mouse1", "mouse2"]
    for i in indiv:
        out_dir = os.path.join(data_path, "parsed", "adj_graphs", i)
        in_paths = glob.glob(os.path.join(in_dir, f"segmented_cells_{i}sample*.csv"))
        build_graphs(params, in_paths, out_dir)

