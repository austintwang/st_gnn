import sys
import os
import pickle
import numpy as np 
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
import seaborn as sns

import model.models as models
from model.loaders import ZhuangBasicCellFFiltered

def load_state(data_dir, name, exp, device):
    target_dir = os.path.join(data_dir, name, exp)
    params_path = os.path.join(target_dir, "params.pickle")
    with open(params_path, "rb") as f:
        params = pickle.load(f)
    params["clear_cache"] = False
    params["device"] = device
    best_path = os.path.join(target_dir, "best_model_epoch.txt")
    with open(best_path, "r") as f:
        best_epoch = f.read().strip()
    model_path = os.path.join(target_dir, "model", f"ckpt_epoch_{best_epoch}.pt")
    model_state = torch.load(model_path)

    return params, model_state

def load_model(vae_model_cls, loader, components, params, model_state):
    m = vae_model_cls(loader.node_in_channels, components, **params)
    m.load_state_dict(model_state)
    m.eval()
    return m

def load_loader(loader_cls, params, clusters_path, cells_per_cluster):
    loader_params = params.copy()
    loader_params["st_clusters_path"] = clusters_path
    loader_params["num_cells_per_cluster"] = cells_per_cluster
    loader_params["batch_size"] = 400
    loader_params["saint_num_steps"] = {"train": 500, "val": 500, "test": 500}
    # loader_params["clear_cache"] = True ####

    loader = loader_cls(**loader_params)
    return loader

def sample_model(loader, vae_model, num_select, num_total, device, mode):
    data_lst = []
    count = 0

    # print(len(loader.val_part), len(loader.test_part))
    print((loader.val_part & loader.test_part))

    if mode == "val":
        sampler = loader.val_sampler
        node_map = loader.val_maps["node_to_id"]
    elif mode == "test":
        sampler = loader.test_sampler
        node_map = loader.test_maps["node_to_id"]
    
    vae_model.to(device)
    enough = False
    while not enough:
        for data in sampler:
            data.to(device)
            num_samples = data.x.shape[0]
            out = vae_model.sample_coords(data)
            out_coords = out["coords"]

            cell_mask = data.cell_mask.detach().cpu().numpy()
            # exp = data.x.detach().cpu().numpy()[cell_mask]
            # coords_true = data.pos.detach().cpu().numpy()[cell_mask]
            cell_indices = data.node_index_orig.detach().cpu().numpy()[cell_mask]
            coords_pred = out_coords.detach().cpu().numpy()

            clusters_df = loader.aux_data[0]

            num_samples = coords_pred.shape[0]
            for ind in range(num_samples):
                x, y, z = coords_pred[ind]
                cell = node_map[cell_indices[ind]]
                cluster = clusters_df.loc[cell, "label"]

                entry = {
                    "x": x, 
                    "y": y, 
                    "z": z, 
                    "cell": cell,
                    "cluster": cluster,
                }

                data_lst.append(entry)
            
            count += num_samples
            if count >= num_total:
                enough = True
                break

    # print(count) ####
    data_df = pd.DataFrame.from_records(data_lst)

    df_sampled = data_df.groupby("cell").sample(n=num_select)
    df_sampled = df_sampled.sample(frac=1)

    return df_sampled

def plt_scatter_3d(df, model_name, exp, mode, out_dir):
    # sns.set()
    print(df.max(axis=0)) ####
    print(df.min(axis=0)) ####

    clusters = df.groupby("cluster")
    for name, cluster in clusters:
        # print(cluster) ####
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')

        # cells = cluster.groupby("cell")
        codes, uniques = pd.factorize(cluster["cell"])
        num_cells = len(uniques)
        x = cluster["x"]
        y = cluster["y"]
        z = cluster["z"]
        c = codes

        cmap = ListedColormap(sns.color_palette("husl", num_cells).as_hex())

        ax.scatter(x, y, z, c=c, s=1, cmap=cmap)

        ax.set_xlabel('X (Microns)')
        ax.set_ylabel('Y (Microns)')
        ax.set_zlabel('Z (Microns)')

        ax.set_xlim(-8500, 5000)
        ax.set_ylim(-2000, 4000)
        ax.set_zlim(0, 2000)

        plt.title(f"Cluster {name}, {num_cells} Cells")
        for ii in range(0, 360, 30):
            res_dir = os.path.join(out_dir, model_name, exp, mode, str(ii))
            ax.view_init(azim=ii)
            os.makedirs(res_dir, exist_ok=True)
            plt.savefig(os.path.join(res_dir, f"samples_{name}.svg"), bbox_inches='tight')

        plt.clf()
        plt.close()

def vis_vae(loader_cls, vae_model_cls, components, dname, name, exp, cells_per_cluster, num_samples, num_total, data_dir, out_dir, clusters_path):
    device = dname if dname == "cpu" else f"cuda:{dname}" 
    params, model_state = load_state(data_dir, name, exp, device)
    loader = load_loader(loader_cls, params, clusters_path, cells_per_cluster)

    modes = ["val", "test"]
    for mode in modes:
        vae_model = load_model(vae_model_cls, loader, components, params, model_state)
        df = sample_model(loader, vae_model, num_samples, num_total, device, mode)
        plt_scatter_3d(df, name, exp, mode, out_dir)

if __name__ == '__main__':
    data_dir = "/dfs/user/atwang/data/analyses/st_gnn"
    out_dir = "/dfs/user/atwang/results/st_gnn_results/spt_zhuang/vae/"
    clusters_path = "/dfs/user/atwang/data/spt_zhuang/parsed/cell_labels.csv"

    loader_cls = ZhuangBasicCellFFiltered
    vae_model_cls = models.SupCVAE

    components = {
        "emb": models.EmbMLP,
        "struct": models.StructCoords,
        "struct_enc": models.AuxStructEncMLP,
        "exp_dec": models.AuxExpDecMLP,
    }

    num_samples = 10
    num_total = 1e7
    cells_per_cluster = 1000

    dname = sys.argv[1]

    name = "vb2"

    exps = ["0005"]
    for exp in exps:
        vis_vae(loader_cls, vae_model_cls, components, dname, name, exp, cells_per_cluster, num_samples, num_total, data_dir, out_dir, clusters_path)
