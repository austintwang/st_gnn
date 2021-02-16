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
        data.append(data_exp["val"])
    
    data_df = pd.DataFrame.from_records(data)

def plot_training(df, metric, result_dir):
    sns.set()
    sns.lineplot(data=df, x="epoch", y=metric, hue="name")
    plt.title(f"Validation {metric}")
    plt.savefig(os.path.join(result_dir, f"{metric}_val.svg"), bbox_inches='tight')
    plt.clf()

def vis_training(data_dir, result_dir, names, metrics):
    df = load_data(data_dir, names)
    os.makedirs(result_dir, exist_ok=True)
    for metric in metrics:
        plot_training(df, metrics, result_dir)

if __name__ == '__main__':
    data_dir = "/dfs/user/atwang/data/analyses/st_gnn"
    result_dir = "/dfs/user/atwang/results/st_gnn_results/spt_zhuang/sup/training"
    names = [
        ("sg2", "0001"), 
        ("sgc2", "0001"), 
        ("sb2", "0001")
    ]
    metrics = ["loss", "gaussian_nll", "spearman"]
    vis_training(data_dir, result_dir, names, metrics)