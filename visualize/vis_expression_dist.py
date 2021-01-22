import os
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

data_path = "/dfs/user/atwang/data/spt_zhuang/source/processed_data/counts.h5ad"
result_dir = "/dfs/user/atwang/results/st_gnn_results/spt_zhuang/expression"
os.makedirs(result_dir, exist_ok=True)

adata = sc.read(data_path)
# sc.pp.calculate_qc_metrics(adata)
# sc.pp.normalize_total(adata, target_sum=1e4)
# sc.pp.log1p(adata)
# sc.pl.highly_variable_genes(adata)
# plt.savefig(os.path.join(result_dir, "var_genes.svg"), bbox_inches='tight')
# plt.clf()

sns.set()
sns.distplot(log10(adata.X.flatten() + 1), kde=False)
plt.savefig(os.path.join(result_dir, "hist.svg"), bbox_inches='tight')
plt.clf()