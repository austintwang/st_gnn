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

def sample_model(loader, vae_model, num_select, device):
    data_lst = []

    vae_model.to(device)
    for data in loader.test_sampler:
        data.to(device)
        num_samples = data.x.shape[0]
        out = vae_model(data)
        out_coords = out["coords_from_exp"]
        out_coords_ae = out["coords"]
        emb_mean = out["emb_mean"]
        aux_enc_mean = out["aux_enc_mean"]
        emb_sample = out["emb_sample"]
        aux_enc_sample = out["aux_enc_sample"]


        cell_mask = data.cell_mask.detach().cpu().numpy()
        exp = data.x.detach().cpu().numpy()[cell_mask]
        coords_true = data.pos.detach().cpu().numpy()[cell_mask]
        coords_pred = out_coords.detach().cpu().numpy()
        coords_pred_ae = out_coords_ae.detach().cpu().numpy()
        latent_exp = emb_mean.detach().cpu().numpy()
        latent_struct = aux_enc_mean.detach().cpu().numpy()
        latent_exp_sample = emb_sample.detach().cpu().numpy()
        latent_struct_sample = aux_enc_sample.detach().cpu().numpy()

        # print(latent_exp) ####

        num_samples = exp.shape[0]
        for ind in range(num_samples):
            if exp[ind, 3] == 1:
                ctype = "left"
            elif exp[ind, 4] == 1:
                ctype = "middle"
            elif exp[ind, 5] == 1:
                ctype = "right"

            x, y, z = coords_pred[ind]
            x_ae, y_ae, z_ae = coords_pred_ae[ind]
            h1_e, h2_e, h3_e = latent_exp[ind]
            h1_s, h2_s, h3_s = latent_struct[ind]
            h1_esp, h2_esp, h3_esp = latent_exp_sample[ind]
            h1_ssp, h2_ssp, h3_ssp = latent_struct_sample[ind]

            entry = {
                "x": x, 
                "y": y, 
                "z": z, 
                "x_ae": x_ae,
                "y_ae": y_ae,
                "z_ae": z_ae,
                "h1_e": h1_e,
                "h2_e": h2_e,
                "h3_e": h3_e,
                "h1_s": h1_s,
                "h2_s": h2_s,
                "h3_s": h3_s,
                "h1_esp": h1_esp,
                "h2_esp": h2_esp,
                "h3_esp": h3_esp,
                "h1_ssp": h1_ssp,
                "h2_ssp": h2_ssp,
                "h3_ssp": h3_ssp,
                "input": ctype
            }
            data_lst.append(entry)

    data_df = pd.DataFrame.from_records(data_lst)
    # print(data_df) ####
    # print(data_df.groupby("input").count()) ####

    df_sampled = data_df.groupby("input").sample(n=num_select)
    # print(df_sampled) ####

    return df_sampled

def plt_scatter(df, name, exp, out_dir):

    sns.set()
    sns.scatterplot(data=df, x="x", y="y", hue="input")
    plt.title(f"Latent Distribution Samples By Input")
    res_dir = os.path.join(out_dir, name, "samples")
    os.makedirs(res_dir, exist_ok=True)
    plt.savefig(os.path.join(res_dir, f"{exp}_samples.svg"), bbox_inches='tight')
    plt.clf()

    sns.set()
    sns.scatterplot(data=df, x="x_ae", y="y_ae")
    plt.title(f"Structure Autoencoder Samples")
    res_dir = os.path.join(out_dir, name, "samples")
    os.makedirs(res_dir, exist_ok=True)
    plt.savefig(os.path.join(res_dir, f"{exp}_samples_ae.svg"), bbox_inches='tight')
    plt.clf()

    sns.set()
    g = sns.pairplot(data=df, vars=["h1_e", "h2_e", "h3_e"], hue="input", diag_kind="hist")
    g.fig.suptitle(f"Expression Encoder Latent Means", y=1.08)
    res_dir = os.path.join(out_dir, name, "samples")
    os.makedirs(res_dir, exist_ok=True)
    plt.savefig(os.path.join(res_dir, f"{exp}_latent_exp.svg"), bbox_inches='tight')
    plt.clf()

    sns.set()
    g = sns.pairplot(data=df, vars=["h1_s", "h2_s", "h3_s"], hue="input", diag_kind="hist")
    g.fig.suptitle(f"Structure Encoder Latent Means", y=1.08)
    res_dir = os.path.join(out_dir, name, "samples")
    os.makedirs(res_dir, exist_ok=True)
    plt.savefig(os.path.join(res_dir, f"{exp}_latent_struct.svg"), bbox_inches='tight')
    plt.clf()

    sns.set()
    g = sns.pairplot(data=df, vars=["h1_esp", "h2_esp", "h3_esp"], hue="input", diag_kind="hist")
    g.fig.suptitle(f"Expression Encoder Latent Samples", y=1.08)
    res_dir = os.path.join(out_dir, name, "samples")
    os.makedirs(res_dir, exist_ok=True)
    plt.savefig(os.path.join(res_dir, f"{exp}_latent_exp_samples.svg"), bbox_inches='tight')
    plt.clf()

    sns.set()
    g = sns.pairplot(data=df, vars=["h1_ssp", "h2_ssp", "h3_ssp"], hue="input", diag_kind="hist")
    g.fig.suptitle(f"Structure Encoder Latent Samples", y=1.08)
    res_dir = os.path.join(out_dir, name, "samples")
    os.makedirs(res_dir, exist_ok=True)
    plt.savefig(os.path.join(res_dir, f"{exp}_latent_struct_samples.svg"), bbox_inches='tight')
    plt.clf()

    plt.close()

def vis_vae(loader_cls, vae_model_cls, components, dname, name, exp, num_samples, data_dir, out_dir):
    device = dname if dname == "cpu" else f"cuda:{dname}" 
    params, model_state = load_state(data_dir, name, exp, device)
    loader = loader_cls(**params)
    vae_model = load_model(vae_model_cls, loader, components, params, model_state)

    df = sample_model(loader, vae_model, num_samples, device)
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

    num_samples = 1000

    dname = sys.argv[1]

    name = "vs"

    exps = ["0022"]
    for exp in exps:
        vis_vae(loader_cls, vae_model_cls, components, dname, name, exp, num_samples, data_dir, out_dir)
