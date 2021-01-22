import scanpy as sc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

data_path = "/dfs/user/atwang/data/spt_zhuang/source/processed_data/counts.h5ad"
result_path = "/dfs/user/atwang/results/st_gnn_results/spt_zhuang/expression/hist.svg"

adata = sc.read(data_path)
sns.set()
sns.distplot(adata.obs["total_counts"], kde=False, ax=axs[0])
plt.savefig(result_path, bbox_inches='tight')
plt.clf()