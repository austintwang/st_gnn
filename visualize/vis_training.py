import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(data_dir, names):
    data = []
    for name in names:
        data_path = os.path.join(data_dir, name, "stats.pickle")
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
    for metric in metrics:
        plot_training(df, metrics, result_dir)

if __name__ == '__main__':
    names = ["sg2", "sgc2", "sb2"]
    metrics = ["loss", "gaussian_nll", "spearman"]
    vis_training(data_dir, result_dir, names, metrics)