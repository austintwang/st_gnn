import sys
import os
import pickle
import numpy as np 
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
    loader_params["st_clusters"] = clusters_path
    loader_params["cells_per_cluster"] = cells_per_cluster
    loader_params["batch_size"] = 400
    loader_params["saint_num_steps"] = {"train": 500, "val": 500, "test":500}

    loader = loader_cls(**loader_params)
    return loader

def sample_model(loader, vae_model, num_select, num_total, device, mode):
    data_lst = []
    count = 0

    if mode == "val":
        sampler = loader.val_sampler
    elif mode == "test":
        sampler = loader.test_sampler
    
    vae_model.to(device)
    for data in sampler:
        data.to(device)
        num_samples = data.x.shape[0]
        out = vae_model.sample_coords(data)
        out_coords = out["coords"]

        cell_mask = data.cell_mask.detach().cpu().numpy()
        # exp = data.x.detach().cpu().numpy()[cell_mask]
        # coords_true = data.pos.detach().cpu().numpy()[cell_mask]
        cell_indices = data.node_indices_orig.detach().cpu().numpy()[cell_mask]
        coords_pred = out_coords.detach().cpu().numpy()

        # print(latent_exp) ####

        num_samples = coords_pred.shape[0]
        for ind in range(num_samples):
            x, y, z = coords_pred[ind]
            cell = loader.val_maps[cell_indices[pred]]
            cluster = loader.clusters[cell, "label"]

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
            break

    data_df = pd.DataFrame.from_records(data_lst)

    df_sampled = data_df.groupby("cell").sample(n=num_select)
    df_sampled = df_sampled.sample(frac=1)

    return df_sampled

def plt_scatter_3d(df, model_name, exp, mode, out_dir):
    # sns.set()

    clusters = df.groupby("cluster")
    for name, cluster in clusters:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')

        # cells = cluster.groupby("cell")
        num_cells = len(pd.unique(cluster["cell"]))
        x = cluster["x"]
        y = cluster["y"]
        z = cluster["z"]
        c = cluster["cell"]

        ax.scatter(x, y, z, c=c)

        ax.set_xlabel('X (Microns)')
        ax.set_ylabel('Y (Microns)')
        ax.set_zlabel('Z (Microns)')

        plt.title(f"Cluster {name}, {num_cells} Cells")
        res_dir = os.path.join(out_dir, model_name, exp, mode)
        os.makedirs(res_dir, exist_ok=True)
        plt.savefig(os.path.join(res_dir, f"samples_{name}.svg"), bbox_inches='tight')

        plt.clf()

    plt.close()

def vis_vae(loader_cls, vae_model_cls, components, dname, name, exp, num_samples, num_total, data_dir, out_dir):
    device = dname if dname == "cpu" else f"cuda:{dname}" 
    params, model_state = load_state(data_dir, name, exp, device)
    loader = load_loader(loader_cls, params, clusters_path, cells_per_cluster)

    mode = ["val", "test"]
    for model_state in model_states:
        vae_model = load_model(vae_model_cls, loader, components, params, mode)

        df = sample_model(loader, vae_model, num_samples, num_total, device)
        plt_scatter(df, name, exp, mode, out_dir)

if __name__ == '__main__':
    data_dir = "/dfs/user/atwang/data/analyses/st_gnn"
    out_dir = "/dfs/user/atwang/results/st_gnn_results/spt_zhuang/vae/"

    loader_cls = ZhuangBasicCellFFiltered
    vae_model_cls = models.SupCVAE

    components = {
        "emb": models.EmbMLP,
        "struct": models.StructCoords,
        "struct_enc": models.AuxStructEncMLP,
        "exp_dec": models.AuxExpDecMLP,
    }

    num_samples = 100
    num_total = 100000

    dname = sys.argv[1]

    name = "vb2"

    exps = ["0005"]
    for exp in exps:
        vis_vae(loader_cls, vae_model_cls, components, dname, name, exp, num_samples, data_dir, out_dir)