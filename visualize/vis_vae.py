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
from model.loaders import Synth3Layer

def load_state(data_dir, name, exp):
    target_dir = os.path.join(data_dir, name, exp)
    params_path = os.path.join(target_dir, "params.pickle")
    with open(params_path, "rb") as f:
        params = pickle.load(f)
    params["clear_cache"] = False
    best_path = os.path.join(target_dir, "best_model_epoch.txt")
    with open(best_path, "r") as f:
        best_epoch = f.read().strip()
    model_path = os.path.join(target_dir, "model", f"ckpt_epoch_{best_epoch}.pt")
    model_state = torch.load(model_path)

    return params, model_state

def load_model(vae_model_cls, loader, components, params, model_state):
    m = vae_model_cls(loader.node_in_channels, components, **params)
    m.load_state_dict(model_state)
    return m

def sample_model(loader, vae_model, device):
    data_lst = []

    vae_model.to(device)
    for data in loader.test_sampler:
        data.to(device)
        num_samples = data.x.shape[0]
        out = vae_model.sample_coords(data)
        out_coords = out["coords"]

        cell_mask = data.cell_mask.detach().cpu().numpy()
        exp = data.x.detach().cpu().numpy()[cell_mask]
        coords_true = data.pos.detach().cpu().numpy()[cell_mask]
        coords_pred = out_coords.detach().cpu().numpy()

        num_samples = exp.shape[0]
        for ind in range(num_samples):
            if exp[ind, 3] == 1:
                ctype = "left"
            elif exp[ind, 4] == 1:
                ctype = "middle"
            elif exp[ind, 5] == 1:
                ctype = "right"

            x, y, _ = coords_pred[ind]

            entry = {"x": x, "y": y, "input": ctype}
            data_lst.append(entry)

    data_df = pd.DataFrame.from_records(data_lst)

    return data_df

def plt_scatter(df, name, exp, out_dir):
    sns.set()
    sns.scatterplot(data=df, x="x", y="y", hue="input")
    plt.title(f"Latent Distribution Samples By Input")
    res_dir = os.path.join(out_dir, name, "samples")
    os.makedirs(res_dir, exist_ok=True)
    plt.savefig(os.path.join(res_dir, f"{exp}_samples.svg"), bbox_inches='tight')
    plt.clf()

def vis_vae(loader_cls, vae_model_cls, components, dname, name, exp, data_dir, out_dir):
    device = dname if dname == "cpu" else f"cuda:{dname}" 
    params, model_state = load_state(data_dir, name, exp)
    loader = loader_cls(**params)
    vae_model = load_model(vae_model_cls, loader, components, params, model_state)

    df = sample_model(loader, vae_model, device)
    plt_scatter(df, name, exp, out_dir)

if __name__ == '__main__':
    data_dir = "/dfs/user/atwang/data/analyses/st_gnn"
    out_dir = "/dfs/user/atwang/results/st_gnn_results/spt_zhuang/vae/"

    loader_cls = Synth3Layer
    vae_model_cls = models.SupCVAE

    components = {
        "emb": models.EmbMLP,
        "struct": models.StructCoords,
        "struct_enc": models.AuxStructEncMLP,
        "exp_dec": models.AuxExpDecMLP,
    }

    dname = sys.argv[1]

    name = "vs"

    exps = ["0012", "0013"]
    for exp in exps:
        vis_vae(loader_cls, vae_model_cls, components, dname, name, exp, data_dir, out_dir)
