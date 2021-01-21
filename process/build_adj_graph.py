import os
import glob
import numpy as np 
import tqdm

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
        self.graph = {}

    def add_cell(self, cell_id, x, y, z):
        coords = (x, y, z)
        self.cell_pos[cell_id] = coords
        x_lh, y_lh, z_lh = [int((i - radius) * self.scale) // self.bsize_scaled for i in coords]
        x_uh, y_uh, z_uh = [int((i + radius) * self.scale) // self.bsize_scaled for i in coords]

        adj = set()
        for i in range(x_lh, x_uh + 1):
            for j in range(y_lh, y_uh + 1):
                for k in range(z_lh, z_uh + 1):
                    bidx = (i,j,k)
                    bucket = self.buckets.setdefault(bidx, [])
                    for c in bucket:
                        xc, yc, zc = self.cell_pos[c]
                        if (xc - x)**2 + (yc - y)**2 + (zc - z)**2 <= radsq:
                            adj.add(c)

                    bucket.append(cell_id)

        for c in adj:
            self.graph.setdefault(c, set()).add(cell_id)
            self.graph.setdefault(cell_id, set()).add(c)

    def get_graph(self):
        return {"cells": self.cell_pos, "graph": self.graph}

def parse_cell(line):
    cell_id, x_b, _, y_b, slice_id = line.split("\"")
    cell_id = cell_id.rstrip(",")

    x_b = np.fromstring(x_b, sep=",")
    y_b = np.fromstring(y_b, sep=",")
    x1 = x_b[:-1]
    x2 = x_b[1:]
    y1 = y_b[:-1]
    y2 = y_b[1:]
    q = x1 * y2 - x2 * y1
    a = np.sum(q)
    x = np.sum((x1 + x2) * q) / (3 * a)
    y = np.sum((y1 + y2) * q) / (3 * a)

    z = int(slice_id.split("_")[1][5:]) * 10

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

def build_graphs(params, in_paths, out_dir):
    tables = {}
    for radius, bucket_size in params:
        tables[radius] = CellTable(bucket_size, radius)

    for p in in_paths:
        load_file(tables, p)

    os.makedirs(out_dir, exist_ok=True)
    for k, v in tables.items():
        out_path = os.path.join(out_dir, f"adj_r_{k}.pickle")
        pickle.dump(v.get_graph(), out_path)

if __name__ == '__main__':
    params = [
        (10, 20),
        (100, 200),
        (1000, 2000)
    ]

    data_path = "/dfs/user/atwang/data/spt_zhuang/"
    in_dir = os.path.join(data_path, "source", "processed_data")

    indiv = ["mouse1", "mouse2"]
    for i in indiv:
        out_dir = os.path.join(data_path, "parsed", "adj_graphs", indiv)
        in_paths = glob.glob(os.path.join(in_dir, f"segmented_cells_{i}sample*.csv"))
        build_graphs(params, in_paths, out_dir)

