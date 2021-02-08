import os
import glob
import numpy as np 
import snap
import tqdm
import pickle

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

    return cell_id, x, y, z

def load_file(coords, cells, in_path):
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
            coords[cell_id] = [x, y, z]
            cells.append(cell_id)

def build_coords(in_paths, out_dir):
    coords = {}
    cells = {}
    for org, paths in in_paths.items():
        cells_org = cells.setdefault(org, [])
        for p in paths:
            load_file(coords, cells_org, p)

    out_path_coords = os.path.join(out_dir, "cell_coords.pickle")
    out_path_coords = os.path.join(out_dir, "cell_orgs.pickle")
    with open(out_path_coords, "wb") as out_file:
        pickle.dump(coords)
    with open(out_path_cells, "wb") as out_file:
        pickle.dump(cells)

if __name__ == '__main__':

    data_path = "/dfs/user/atwang/data/spt_zhuang/"
    in_dir = os.path.join(data_path, "source", "processed_data")
    out_dir = os.path.join(data_path, "parsed")

    indiv = ["mouse1", "mouse2"]
    in_paths = {}
    for i in indiv:
        in_paths[i] = glob.glob(os.path.join(in_dir, f"segmented_cells_{i}sample*.csv"))
    
    build_coords(in_paths, out_dir)

