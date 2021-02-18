import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def load_data(data_dir, names):
    data = []
    for name, exp in names:
        data_path = os.path.join(data_dir, name, exp, "stats.pickle")
        with open(data_path, "rb") as f:
            data_exp = pickle.load(f)
        dv = data_exp["val"]
        for i in dv:
            i.setdefault("name", name)
        data.extend(dv)
    
    data_df = pd.DataFrame.from_records(data)
    return data_df

def plot_training(df, metric, result_dir, constraints, max_epochs):
    sns.set()
    sns.lineplot(data=df, x="epoch", y=metric, hue="name")
    plt.title(f"Validation {metric}")
    if max_epochs is not None:
        plt.xlim(right=max_epochs)
    if metric in constraints:
        plt.ylim(**constraints[metric])
    plt.savefig(os.path.join(result_dir, f"{metric}_val.svg"), bbox_inches='tight')
    plt.clf()

def vis_training(data_dir, result_dir, names, metrics, constraints, max_epochs=None):
    df = load_data(data_dir, names)
    os.makedirs(result_dir, exist_ok=True)
    for metric in metrics:
        plot_training(df, metric, result_dir, constraints, max_epochs)

if __name__ == '__main__':
    data_dir = "/dfs/user/atwang/data/analyses/st_gnn"
    result_dir = "/dfs/user/atwang/results/st_gnn_results/spt_zhuang/sup/training"

    names = [
        ("sg2", "0002"), 
        ("sgc2", "0002"), 
        ("sb2", "0002")
    ]
    metrics = [
        "loss", 
        "gaussian_nll", 
        "spearman", 
        "mean_pred_mean", 
        "mean_pred_std", 
        "mse",
        "mse_lt_100",
        "mse_100_500",
        "mse_gt_500",
        "mse_log",
        "mean_chisq",
        "spearman"
    ]
    constraints = {
        "loss": {"top": 2000},
        "gaussian_nll": {"top": 0.2}
    }
    # vis_training(data_dir, result_dir, names, metrics, constraints, max_epochs=400)


    names = [
        ("sg2bin50", "0002"), 
        ("sgc2bin50", "0002"), 
        ("sb2bin50", "0002"),
        ("sg2bin100", "0001"), 
        ("sgc2bin100", "0001"), 
        ("sb2bin100", "0002"),
        ("sg2bin500", "0001"), 
        ("sgc2bin500", "0001"), 
        ("sb2bin500", "0001")
    ]
    metrics = [
        "loss", 
        "acc", 
        "f1", 
        "mcc", 
    ]
    constraints = {
    }
    vis_training(data_dir, result_dir, names, metrics, constraints, max_epochs=400)