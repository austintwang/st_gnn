import os
import random
import math
import numpy as np
from scipy import sparse
import pandas as pd
import scanpy as sc
import torch
from torch_geometric.data import Data, GraphSAINTRandomWalkSampler
from torch_geometric.utils import from_scipy_sparse_matrix

class Loader(object):
    def __init__(self, **kwargs):
        self.params = kwargs

        in_data, partitions = self._import_data()
        self.train_part, self.val_part, self.test_part = partitions

        self.train_data, self.train_maps = self._build_graph(in_data, self.train_part)
        self.val_data, self.val_maps = self._build_graph(in_data, self.val_part)
        self.test_data, self.test_maps = self._build_graph(in_data, self.test_part)

        self.train_sampler = self._build_sampler()

    # def _import_data(self):
    #     pass

    # def _build_graph(self, in_data, partition):
    #     pass

    # def _build_sampler(self):
    #     pass

class SaintRWLoader(Loader):
    def _build_sampler(self):
        sampler = GraphSAINTRandomWalkSampler(
            self.train_data, 
            batch_size=self.params["batch_size"],
            walk_length=self.params["saint_walk_length"],
            num_steps=self.params["saint_num_steps"], 
            sample_coverage=self.params["saint_sample_coverage"],
            num_workers=self.params["num_workers"]
        )
        return sampler

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
        coords_pad = torch.cat((torch.full((num_genes, coords_dims), np.nan), coords), 0)
        cell_mask = torch.cat((torch.full((num_genes,), False), torch.full((num_cells,), True)), 0)

        # print(partition) ####
        # print(st_anndata.obs_names[:5]) ####
        # print(partition.pop()) ####
        # print(num_cells) ####

        x = torch.zeros(num_genes + num_cells, num_genes + 1)
        x[:num_genes,:num_genes].fill_diagonal_(1.)
        x[num_genes:,-1].fill_(1.)
        node_to_id = np.concatenate((genes, cells))

        expr = np.log(st_anndata.X[part_mask,:] + 1)
        expr_sparse = sparse.coo_matrix(expr)
        edges, edge_features = from_scipy_sparse_matrix(expr_sparse)

        # edges_l = []
        # edge_features_l = []
        # threshold = self.params["st_exp_threshold"]
        # for index, x in np.ndenumerate(expr):
        #     if x >= threshold:
        #         cell, gene = index
        #         a = cell + num_genes
        #         b = gene
        #         edges_l.append([a, b])
        #         edges_l.append([b, a])
        #         edge_features_l.append(x)
        #         edge_features_l.append(x)

        # edges = torch.tensor(edges_l).transpose_(0, 1)
        # edge_features = torch.tensor(edge_features_l)

        data = Data(x=x, edge_index=edges, edge_attr=edge_features, pos=coords_pad, cell_mask=cell_mask)
        print(data) ####
        maps = {
            "gene_to_node": gene_to_node,
            "cell_to_node": cell_to_node,
            "node_to_id": node_to_id,
        }

        return data, maps


if __name__ == '__main__':
    data_path = "/dfs/user/atwang/data/spt_zhuang/"
    coords_path = os.path.join(data_path, "parsed", "cell_coords.pickle")
    orgs_path = os.path.join(data_path, "parsed", "cell_orgs.pickle")
    exp_path = "/dfs/user/atwang/data/spt_zhuang/source/processed_data/counts.h5ad"

    cache_path = "/dfs/user/atwang/data/spt_zhuang/cache/test/loader.pickle"
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    params = {
        "batch_size": 500,
        "saint_walk_length": 2,
        "saint_num_steps": 5,
        "saint_sample_coverage": 10,
        "st_exp_path": exp_path,
        "st_coords_path": coords_path,
        "st_organisms_path": orgs_path,
        "train_prop": 0.1,
        "st_exp_threshold": 0.001,
        "num_workers": 8,
    }
    loader = ZhuangBasic(**params)
    for i in loader.train_sampler:
        print(i)

    with open(cache_path, "wb") as cache_file:
        pickle.dump(loader, cache_file)

    with open(cache_path, "rb") as cache_file:
        loader = pickle.load(cache_file)

    print(i.val_data)
    for i in loader.train_sampler:
        print(i)

