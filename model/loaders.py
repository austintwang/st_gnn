import os
import random
import math
import pickle
import shutil
from tqdm import tqdm
import numpy as np
from scipy import sparse
import pandas as pd
import scanpy as sc
import anndata as ad
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from torch_geometric.data import Data, GraphSAINTRandomWalkSampler
from torch_geometric.utils import from_scipy_sparse_matrix

class GraphSAINTSamplerFixed(GraphSAINTRandomWalkSampler):
    def __compute_norm__(self):
        node_count = torch.zeros(self.N, dtype=torch.float)
        edge_count = torch.zeros(self.E, dtype=torch.float)

        loader = torch.utils.data.DataLoader(self, batch_size=200,
                                             collate_fn=lambda x: x,
                                             num_workers=self.num_workers)

        if self.log:  # pragma: no cover
            pbar = tqdm(total=self.N * self.sample_coverage)
            pbar.set_description('Compute GraphSAINT normalization')

        num_samples = total_sampled_nodes = 0
        while total_sampled_nodes < self.N * self.sample_coverage:
            for data in loader:
                for node_idx, adj in data:
                    edge_idx = adj.storage.value()
                    node_count[node_idx] += 1
                    edge_count[edge_idx] += 1
                    total_sampled_nodes += node_idx.size(0)

                    if self.log:  # pragma: no cover
                        pbar.update(node_idx.size(0))
            num_samples += self.num_steps
 
        if self.log:  # pragma: no cover
            pbar.close()

        row, _, edge_idx = self.adj.coo()
        t = torch.empty_like(edge_count).scatter_(0, edge_idx, node_count[row])
        edge_norm = (t / edge_count).clamp_(0, 1e4)
        edge_norm[torch.isnan(edge_norm)] = 0.1

        node_count[node_count == 0] = 0.1
        node_norm = num_samples / node_count / self.N

        return node_norm, edge_norm

class Loader(object):
    def __init__(self, **kwargs):
        self.params = kwargs
        self.cache_dir = os.path.join(self.params["loader_cache_dir"], self.__class__.__name__)
        os.makedirs(self.cache_dir, exist_ok=True)

        cache_path = os.path.join(self.cache_dir, "imports.pickle")
        if self.params.get("clear_cache", False) or not os.path.exists(cache_path):
            imports = self._import_data()
            with open(cache_path, "wb") as cache_file:
                pickle.dump(imports, cache_file)
        else:
            imports = pd.read_pickle(cache_path)

        in_data, partitions = imports[:2]
        self.aux_data = imports[2:]

        self.train_part, self.val_part, self.test_part = partitions

        self.train_data, self.train_maps, self.node_in_channels = self._build_graph(in_data, self.train_part)
        self.val_data, self.val_maps, _ = self._build_graph(in_data, self.val_part)
        self.test_data, self.test_maps, _ = self._build_graph(in_data, self.test_part)

        self.train_sampler = self._build_sampler(self.train_data, "train")
        self.val_sampler = self._build_sampler(self.val_data, "val")

    def _import_data(self):
        raise NotImplementedError

    def _build_graph(self, in_data, partition):
        raise NotImplementedError

    def _build_sampler(self, data, group):
        raise NotImplementedError


class SaintRWLoader(Loader):
    def _build_sampler(self, data, group):
        sampler_cache_dir = os.path.join(self.cache_dir, group)
        os.makedirs(sampler_cache_dir, exist_ok=True)
        if self.params.get("clear_cache", False):
            try:
                shutil.rmtree(sampler_cache_dir)
            except FileNotFoundError:
                pass
            os.makedirs(sampler_cache_dir)

        sampler = GraphSAINTSamplerFixed(
            data, 
            batch_size=self.params["batch_size"],
            walk_length=self.params["saint_walk_length"],
            num_steps=self.params["saint_num_steps"][group], 
            sample_coverage=self.params["saint_sample_coverage"],
            save_dir=sampler_cache_dir,
            num_workers=self.params["num_workers"]
        )
        return sampler

# class SaintRWTestLoader(Loader):
#     def _build_sampler(self, data, group):
#         sampler_cache_dir = os.path.join(self.cache_dir, group)
#         os.makedirs(sampler_cache_dir, exist_ok=True)
#         if self.params.get("clear_cache", False):
#             try:
#                 shutil.rmtree(sampler_cache_dir)
#             except FileNotFoundError:
#                 pass
#             os.makedirs(sampler_cache_dir)

#         sampler = GraphSAINTTestSampler(
#             self.train_data, 
#             batch_size=self.params["batch_size"],
#             walk_length=self.params["saint_walk_length"],
#             num_steps=self.params["saint_num_steps"][group], 
#             sample_coverage=self.params["saint_sample_coverage"],
#             save_dir=sampler_cache_dir,
#             num_workers=self.params["num_workers"]
#         )
#         print(sampler.node_norm) ####
#         return sampler


class ZhuangBasic(SaintRWLoader):
    def _import_data(self):
        anndata = self.params.get("st_anndata", sc.read(self.params["st_exp_path"]))
        coords = self.params.get("st_coords", pd.read_pickle(self.params["st_coords_path"]))
        organisms = self.params.get("st_organisms", pd.read_pickle(self.params["st_organisms_path"]))

        m1 = organisms["mouse1"]
        random.shuffle(m1)
        num_train = int(self.params["train_prop"] * len(m1))
        train = set(m1[:num_train])
        val = set(m1[num_train:])
        test = set(organisms["mouse2"])

        in_data = (anndata, coords)
        partitions = (train, val, test)

        return in_data, partitions

    def _build_graph(self, in_data, partition):
        st_anndata, st_coords = in_data

        genes = np.array(st_anndata.var_names)
        gene_to_node = {val: ind for ind, val in enumerate(genes)}
        num_genes = len(genes)

        part_mask = np.array([i in partition for i in st_anndata.obs_names])
        cells = np.array(st_anndata.obs_names[part_mask])
        cell_to_node = {val: ind + num_genes for ind, val in enumerate(cells)}
        num_cells = len(cells)
        coords = torch.tensor([st_coords[i] for i in cells])
        coords_dims = coords.shape[1]
        coords_pad = torch.cat((torch.full((num_genes, coords_dims), 0), coords), 0)
        cell_mask = torch.cat((torch.full((num_genes,), False), torch.full((num_cells,), True)), 0)

        node_in_channels = num_genes + 1
        x = torch.zeros(num_genes + num_cells, num_genes + 1)
        x[:num_genes,:num_genes].fill_diagonal_(1.)
        x[num_genes:,-1].fill_(1.)
        node_to_id = np.concatenate((genes, cells))

        expr = np.vstack((np.zeros((num_genes, num_genes),), np.log(st_anndata.X[part_mask,:] + 1)),)
        expr_sparse_cg = sparse.coo_matrix(np.nan_to_num(expr / expr.sum(axis=0, keepdims=1)))
        edges_cg, edge_features_cg = from_scipy_sparse_matrix(expr_sparse_cg)
        expr_sparse_gc = sparse.coo_matrix(np.nan_to_num((expr / expr.sum(axis=1, keepdims=1)).T))
        edges_gc, edge_features_gc = from_scipy_sparse_matrix(expr_sparse_gc)

        edges = torch.cat((edges_cg, edges_gc), 1)
        edge_attr = torch.cat((edge_features_cg, edge_features_gc), 0).float()
        edge_type = torch.cat(
            (torch.zeros_like(edge_features_cg, dtype=torch.long), torch.ones_like(edge_features_gc, dtype=torch.long)), 
            0
        )

        data = Data(x=x, edge_index=edges, edge_attr=edge_attr, edge_type=edge_type, pos=coords_pad, cell_mask=cell_mask)
        print(data) ####
        maps = {
            "gene_to_node": gene_to_node,
            "cell_to_node": cell_to_node,
            "node_to_id": node_to_id,
        }

        return data, maps, node_in_channels


class ZhuangBasicCellF(ZhuangBasic):
    def _build_graph(self, in_data, partition):
        st_anndata, st_coords = in_data

        genes = np.array(st_anndata.var_names)
        gene_to_node = {val: ind for ind, val in enumerate(genes)}
        num_genes = len(genes)

        part_mask = np.array([i in partition for i in st_anndata.obs_names])
        cells = np.array(st_anndata.obs_names[part_mask])
        cell_to_node = {val: ind + num_genes for ind, val in enumerate(cells)}
        num_cells = len(cells)
        coords = torch.tensor([st_coords[i] for i in cells])
        # print(coords) ####
        # print(partition) ####
        # print(st_anndata.obs_names) ####
        coords_dims = coords.shape[1]
        coords_pad = torch.cat((torch.full((num_genes, coords_dims), 0), coords), 0).float()
        cell_mask = torch.cat((torch.full((num_genes,), False), torch.full((num_cells,), True)), 0)

        expr_orig = np.log(st_anndata.X[part_mask,:] + 1)
        expr = np.vstack((np.zeros((num_genes, num_genes),), expr_orig),)
        expr_sparse_cg = sparse.coo_matrix(np.nan_to_num(expr / expr.sum(axis=0, keepdims=1)))
        edges_cg, edge_features_cg = from_scipy_sparse_matrix(expr_sparse_cg)
        expr_sparse_gc = sparse.coo_matrix(np.nan_to_num((expr / expr.sum(axis=1, keepdims=1)).T))
        edges_gc, edge_features_gc = from_scipy_sparse_matrix(expr_sparse_gc)

        node_in_channels = 2 * num_genes 
        x = torch.zeros(num_genes + num_cells, num_genes + num_genes)
        x[:num_genes,:num_genes].fill_diagonal_(1.)
        x[num_genes:,num_genes:] = torch.tensor(expr_orig).float()
        node_to_id = np.concatenate((genes, cells))

        edges = torch.cat((edges_cg, edges_gc), 1)
        edge_attr = torch.cat((edge_features_cg, edge_features_gc), 0).float()
        edge_type = torch.cat(
            (torch.zeros_like(edge_features_cg, dtype=torch.long), torch.ones_like(edge_features_gc, dtype=torch.long)), 
            0
        )

        node_index_orig = torch.arange(x.shape[0])

        data = Data(x=x, edge_index=edges, edge_attr=edge_attr, edge_type=edge_type, pos=coords_pad, cell_mask=cell_mask, node_index_orig=node_index_orig)
        print(data) ####
        maps = {
            "gene_to_node": gene_to_node,
            "cell_to_node": cell_to_node,
            "node_to_id": node_to_id,
        }

        return data, maps, node_in_channels


class Synth3Layer(ZhuangBasicCellF):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.test_sampler = self._build_sampler(self.test_data, "test")

    def _import_data(self):
        num_pts = self.params["synth_num_points"]
        num_pts_total = num_pts * 2
        num_train = int(self.params["train_prop"] * num_pts)

        x = np.random.uniform(size=(num_pts_total),)
        y = np.random.uniform(size=(num_pts_total),)
        z = np.zeros_like(x)
        coords_arr = np.stack([x, y, z], axis=1)
        coords = {str(ind): i for ind, i in enumerate(coords_arr)}

        type1 = (x < 1/3).astype(float)
        type2 = ((1/3 < x) & (x < 2/3)).astype(float)
        type3 = (x >= 2/3).astype(float)
        exp = np.stack([type1, type2, type3], axis=1) * (np.e - 1)
        # var = np.arange(exp.shape[1])
        # obs = np.arange(exp.shape[0])
        anndata = ad.AnnData(X=exp, var=None, obs=None)

        shf = np.random.permutation(num_pts_total)
        train = set(str(i) for i in shf[:num_train])
        val = set(str(i) for i in shf[num_train:num_pts])
        test = set(str(i) for i in shf[num_pts:])

        # print(coords) ####
        in_data = (anndata, coords)
        partitions = (train, val, test)

        return in_data, partitions

class ZhuangBasicCellFFiltered(ZhuangBasicCellF):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.test_sampler = self._build_sampler(self.test_data, "test")

    def _import_data(self):
        anndata = self.params.get("st_anndata", sc.read(self.params["st_exp_path"]))
        coords = self.params.get("st_coords", pd.read_pickle(self.params["st_coords_path"]))
        organisms = self.params.get("st_organisms", pd.read_pickle(self.params["st_organisms_path"]))
        clusters = self.params.get("st_clusters", pd.read_csv(self.params["st_clusters_path"], index_col=0))

        num_cells_per_cluster = self.params["num_cells_per_cluster"]

        m = clusters["slice_id"].str.split("_", n=1, expand=True) 
        mouse = m[0]
        # print(m) ####
        # print(coords) ####

        clusters_m1 = clusters[mouse == "mouse1"]
        select_m1_df = clusters_m1.groupby("label").sample(n=num_cells_per_cluster, replace=True)
        val = train = set(select_m1_df.index)

        clusters_m2 = clusters[mouse == "mouse2"]
        select_m2_df = clusters_m2.groupby("label").sample(n=num_cells_per_cluster, replace=True)
        test = set(select_m2_df.index)

        in_data = (anndata, coords)
        partitions = (train, val, test)

        print(len(val), len(test)) ####

        return in_data, partitions, clusters


if __name__ == '__main__':
    data_path = "/dfs/user/atwang/data/spt_zhuang/"
    coords_path = os.path.join(data_path, "parsed", "cell_coords.pickle")
    orgs_path = os.path.join(data_path, "parsed", "cell_orgs.pickle")
    exp_path = "/dfs/user/atwang/data/spt_zhuang/source/processed_data/counts.h5ad"

    cache_dir = "/dfs/user/atwang/data/spt_zhuang/cache/test"

    params = {
        "batch_size": 5000,
        "saint_walk_length": 2,
        "saint_num_steps": 50,
        "saint_sample_coverage": 10,
        "st_exp_path": exp_path,
        "st_coords_path": coords_path,
        "st_organisms_path": orgs_path,
        "train_prop": 0.1,
        "st_exp_threshold": 0.001,
        "num_workers": 8,
        "clear_cache": True,
        "loader_cache_dir": cache_dir
    }

    loader = ZhuangBasic(**params)
    for i in loader.train_sampler:
        print(i)

