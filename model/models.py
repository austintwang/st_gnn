
import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros

class WRGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_relations, **kwargs):
        super().__init__(aggr="add", node_dim=0, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations

        self.weight = Parameter(torch.Tensor(num_relations, in_channels, out_channels))
        self.root = Parameter(torch.Tensor(in_channels, out_channels))
        self.bias = Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.root)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_weight, edge_type):
        # print(x.shape) ####
        # print(self.weight) ####
        num_nodes = x.size(0)

        out = x @ self.root + self.bias

        for i in range(self.num_relations):
            mask = edge_type == i
            edges_masked = edge_index[:, mask]
            norm = edge_weight[mask].unsqueeze(1)
            # print(edges_masked.dtype, norm.dtype) ####
            h = self.propagate(edges_masked, x=x, size=(num_nodes, num_nodes), norm=norm)
            # print(out.dtype, h.dtype, self.weight[i].dtype) ####
            # print(out.shape, h.shape, self.weight[i].shape) ####
            # print((h.float() @ self.weight[i]).dtype) ####
            out += (h @ self.weight[i])

        # print(out) ####

        return out

    def message(self, x_j, norm):
        # print(x_j.shape) ####
        # print(norm.shape) ####
        return x_j * norm


# class SupNet(torch.nn.Module):
#     def __init__(self, in_channels, **kwargs):
#         super().__init__()

#         self.params = kwargs
#         emb_layers_out_chnls = self.params["emb_layers_out_chnls"]
#         struct_layers_out_chnls = self.params["struct_layers_out_chnls"]
#         self.dropout_prop = self.params["dropout_prop"]

#         self.emb_layers = self._get_gnn(in_channels, emb_layers_out_chnls)
#         self.batch_norm_layers = torch.nn.ModuleList(torch.nn.BatchNorm1d(o) for o in emb_layers_out_chnls)

#         emb_dim = sum(emb_layers_out_chnls) * 2
#         self.struct_layers = torch.nn.ModuleList()
#         prev = emb_dim
#         for i in struct_layers_out_chnls:
#             self.struct_layers.append(
#                 torch.nn.Conv2d(in_channels=prev, out_channels=i, kernel_size=1)
#             )
#             prev = i
#         self.final_dist_layer = torch.nn.Conv2d(in_channels=prev, out_channels=2, kernel_size=1)

#     def _get_gnn(self, in_channels, out_channels):
#         raise NotImplementedError

#     def forward(self, data):
#         z = self._gnn_fwd(data)
#         num_cells = z.shape[0]

#         rtile = z.unsqueeze(0).expand(num_cells, -1, -1)
#         ctile = z.unsqueeze(1).expand(-1, num_cells, -1)
#         pairs = torch.cat((rtile, ctile), dim=2)
#         pairs.unsqueeze_(0)

#         prev = pairs.permute(0, 3, 1, 2)
#         for i in self.struct_layers:
#             h = F.dropout(F.relu(i(prev)), p=self.dropout_prop, training=self.training)
#             prev = h
#         dists = self.final_dist_layer(prev)
#         # print(dists.shape) ####
#         dists = dists.permute(0, 2, 3, 1).squeeze_(dim=0)
#         # print(dists.shape) ####

#         return {"dists": dists}

#     def _gnn_fwd(self, data):
#         raise NotImplementedError


# class SupNetBin(torch.nn.Module):
#     def __init__(self, in_channels, **kwargs):
#         super().__init__()

#         self.params = kwargs
#         emb_layers_out_chnls = self.params["emb_layers_out_chnls"]
#         struct_layers_out_chnls = self.params["struct_layers_out_chnls"]
#         self.dropout_prop = self.params["dropout_prop"]

#         self.emb_layers = self._get_gnn(in_channels, emb_layers_out_chnls)
#         self.batch_norm_layers = torch.nn.ModuleList(torch.nn.BatchNorm1d(o) for o in emb_layers_out_chnls)

#         emb_dim = sum(emb_layers_out_chnls) * 2
#         self.struct_layers = torch.nn.ModuleList()
#         prev = emb_dim
#         for i in struct_layers_out_chnls:
#             self.struct_layers.append(
#                 torch.nn.Conv2d(in_channels=prev, out_channels=i, kernel_size=1)
#             )
#             prev = i
#         self.final_dist_layer = torch.nn.Conv2d(in_channels=prev, out_channels=1, kernel_size=1)

#     def _get_gnn(self, in_channels, out_channels):
#         raise NotImplementedError

#     def forward(self, data):
#         z = self._gnn_fwd(data)
#         num_cells = z.shape[0]

#         rtile = z.unsqueeze(0).expand(num_cells, -1, -1)
#         ctile = z.unsqueeze(1).expand(-1, num_cells, -1)
#         pairs = torch.cat((rtile, ctile), dim=2)
#         pairs.unsqueeze_(0)

#         prev = pairs.permute(0, 3, 1, 2)
#         for i in self.struct_layers:
#             h = F.dropout(F.relu(i(prev)), p=self.dropout_prop, training=self.training)
#             prev = h
#         dists = self.final_dist_layer(prev)
#         # print(dists.shape) ####
#         dists = dists.permute(0, 2, 3, 1).squeeze_(dim=0)

#         return {"logits": dists}

#     def _gnn_fwd(self, data):
#         raise NotImplementedError


# class SupNetS(torch.nn.Module):
#     def __init__(self, in_channels, **kwargs):
#         super().__init__()

#         self.params = kwargs
#         emb_layers_out_chnls = self.params["emb_layers_out_chnls"]
#         struct_layers_out_chnls = self.params["struct_layers_out_chnls"]
#         self.dropout_prop = self.params["dropout_prop"]

#         self.emb_layers = self._get_gnn(in_channels, emb_layers_out_chnls)
#         self.batch_norm_layers = torch.nn.ModuleList(torch.nn.BatchNorm1d(o) for o in emb_layers_out_chnls)

#         emb_dim = sum(emb_layers_out_chnls) * 2
#         self.struct_layers = torch.nn.ModuleList()
#         prev = emb_dim
#         for i in struct_layers_out_chnls:
#             self.struct_layers.append(
#                 torch.nn.Conv2d(in_channels=prev, out_channels=i, kernel_size=1)
#             )
#             prev = i
#         self.final_dist_layer = torch.nn.Conv2d(in_channels=prev, out_channels=1, kernel_size=1)

#     def _get_gnn(self, in_channels, out_channels):
#         raise NotImplementedError

#     def forward(self, data):
#         z = self._gnn_fwd(data)
#         num_cells = z.shape[0]

#         rtile = z.unsqueeze(0).expand(num_cells, -1, -1)
#         ctile = z.unsqueeze(1).expand(-1, num_cells, -1)
#         pairs = torch.cat((rtile, ctile), dim=2)
#         pairs.unsqueeze_(0)

#         prev = pairs.permute(0, 3, 1, 2)
#         for i in self.struct_layers:
#             h = F.dropout(F.relu(i(prev)), p=self.dropout_prop, training=self.training)
#             prev = h
#         dists = self.final_dist_layer(prev) ** 2
#         dists = dists.permute(0, 2, 3, 1).squeeze_(dim=0).squeeze_(dim=-1)

#         return {"dists": dists}

#     def _gnn_fwd(self, data):
#         raise NotImplementedError


# class SupNetLR(torch.nn.Module):
#     def __init__(self, in_channels, **kwargs):
#         super().__init__()

#         self.params = kwargs
#         emb_layers_out_chnls = self.params["emb_layers_out_chnls"]
#         struct_layers_out_chnls = self.params["struct_layers_out_chnls"]
#         self.dropout_prop = self.params["dropout_prop"]

#         self.emb_layers = self._get_gnn(in_channels, emb_layers_out_chnls)
#         self.batch_norm_layers = torch.nn.ModuleList(torch.nn.BatchNorm1d(o) for o in emb_layers_out_chnls)

#         emb_dim = sum(emb_layers_out_chnls)
#         self.struct_layers = torch.nn.ModuleList()
#         prev = emb_dim
#         for i in struct_layers_out_chnls:
#             self.struct_layers.append(
#                 torch.nn.Linear(in_features=prev, out_features=i)
#             )
#             prev = i
#         self.final_dist_layer = torch.nn.Linear(in_features=prev, out_features=3)

#     def _get_gnn(self, in_channels, out_channels):
#         raise NotImplementedError

#     def forward(self, data):
#         z = self._gnn_fwd(data)
#         num_cells = z.shape[0]

#         prev = z
#         for i in self.struct_layers:
#             h = F.dropout(F.relu(i(prev)), p=self.dropout_prop, training=self.training)
#             prev = h
#         coords = self.final_dist_layer(prev)

#         rtile = coords.unsqueeze(0).expand(num_cells, -1, -1)
#         ctile = coords.unsqueeze(1).expand(-1, num_cells, -1)
#         diffs = rtile - ctile
#         diffs_sq = diffs.abs().pow(2)
#         dists_sq = diffs_sq.sum(dim=2)
#         dists = dists_sq.sqrt()

#         # print(coords) ####

#         return {"dists": dists}

#     def _gnn_fwd(self, data):
#         raise NotImplementedError


class StructPairwiseLN(torch.nn.Module):
    def __init__(self, in_channels, **kwargs):
        super().__init__()

        self.params = kwargs
        struct_layers_out_chnls = self.params["struct_layers_out_chnls"]
        self.dropout_prop = self.params["dropout_prop"]

        emb_dim = in_channels * 2
        self.struct_layers = torch.nn.ModuleList()
        prev = emb_dim
        for i in struct_layers_out_chnls:
            self.struct_layers.append(
                torch.nn.Conv2d(in_channels=prev, out_channels=i, kernel_size=1)
            )
            prev = i
        self.final_dist_layer = torch.nn.Conv2d(in_channels=prev, out_channels=2, kernel_size=1)

    def forward(self, z):
        num_cells = z.shape[0]

        rtile = z.unsqueeze(0).expand(num_cells, -1, -1)
        ctile = z.unsqueeze(1).expand(-1, num_cells, -1)
        pairs = torch.cat((rtile, ctile), dim=2)
        pairs.unsqueeze_(0)

        prev = pairs.permute(0, 3, 1, 2)
        for i in self.struct_layers:
            h = F.dropout(F.relu(i(prev)), p=self.dropout_prop, training=self.training)
            prev = h
        dists = self.final_dist_layer(prev)
        # print(dists.shape) ####
        dists = dists.permute(0, 2, 3, 1).squeeze_(dim=0)
        # print(dists.shape) ####

        return {"dists": dists}


class StructPairwiseBin(torch.nn.Module):
    def __init__(self, in_channels, **kwargs):
        super().__init__()

        self.params = kwargs
        struct_layers_out_chnls = self.params["struct_layers_out_chnls"]
        self.dropout_prop = self.params["dropout_prop"]

        emb_dim = in_channels * 2
        self.struct_layers = torch.nn.ModuleList()
        prev = emb_dim
        for i in struct_layers_out_chnls:
            self.struct_layers.append(
                torch.nn.Conv2d(in_channels=prev, out_channels=i, kernel_size=1)
            )
            prev = i
        self.final_dist_layer = torch.nn.Conv2d(in_channels=prev, out_channels=1, kernel_size=1)

    def forward(self, z):
        num_cells = z.shape[0]

        rtile = z.unsqueeze(0).expand(num_cells, -1, -1)
        ctile = z.unsqueeze(1).expand(-1, num_cells, -1)
        pairs = torch.cat((rtile, ctile), dim=2)
        pairs.unsqueeze_(0)

        prev = pairs.permute(0, 3, 1, 2)
        for i in self.struct_layers:
            h = F.dropout(F.relu(i(prev)), p=self.dropout_prop, training=self.training)
            prev = h
        dists = self.final_dist_layer(prev)
        # print(dists.shape) ####
        dists = dists.permute(0, 2, 3, 1).squeeze_(dim=0)

        return {"logits": dists}


class StructPairwiseN(torch.nn.Module):
    def __init__(self, in_channels, **kwargs):
        super().__init__()

        self.params = kwargs
        struct_layers_out_chnls = self.params["struct_layers_out_chnls"]
        self.dropout_prop = self.params["dropout_prop"]

        emb_dim = in_channels * 2
        self.struct_layers = torch.nn.ModuleList()
        prev = emb_dim
        for i in struct_layers_out_chnls:
            self.struct_layers.append(
                torch.nn.Conv2d(in_channels=prev, out_channels=i, kernel_size=1)
            )
            prev = i
        self.final_dist_layer = torch.nn.Conv2d(in_channels=prev, out_channels=1, kernel_size=1)


    def forward(self, z):
        num_cells = z.shape[0]

        rtile = z.unsqueeze(0).expand(num_cells, -1, -1)
        ctile = z.unsqueeze(1).expand(-1, num_cells, -1)
        pairs = torch.cat((rtile, ctile), dim=2)
        pairs.unsqueeze_(0)

        prev = pairs.permute(0, 3, 1, 2)
        for i in self.struct_layers:
            h = F.dropout(F.relu(i(prev)), p=self.dropout_prop, training=self.training)
            prev = h
        dists = self.final_dist_layer(prev) ** 2
        dists = dists.permute(0, 2, 3, 1).squeeze_(dim=0).squeeze_(dim=-1)

        return {"dists": dists}


class StructPairwiseLR(torch.nn.Module):
    def __init__(self, in_channels, **kwargs):
        super().__init__()

        self.params = kwargs
        struct_layers_out_chnls = self.params["struct_layers_out_chnls"]
        self.dropout_prop = self.params["dropout_prop"]

        emb_dim = in_channels
        self.struct_layers = torch.nn.ModuleList()
        prev = emb_dim
        for i in struct_layers_out_chnls:
            self.struct_layers.append(
                torch.nn.Linear(in_features=prev, out_features=i)
            )
            prev = i
        self.final_dist_layer = torch.nn.Linear(in_features=prev, out_features=3)

    def forward(self, z):
        num_cells = z.shape[0]

        prev = z
        for i in self.struct_layers:
            h = F.dropout(F.relu(i(prev)), p=self.dropout_prop, training=self.training)
            prev = h
        coords = self.final_dist_layer(prev)

        rtile = coords.unsqueeze(0).expand(num_cells, -1, -1)
        ctile = coords.unsqueeze(1).expand(-1, num_cells, -1)
        diffs = rtile - ctile
        diffs_sq = diffs.abs().pow(2)
        dists_sq = diffs_sq.sum(dim=2)
        dists = dists_sq.sqrt()

        # print(coords) ####

        return {"dists": dists}


class StructCoords(torch.nn.Module):
    def __init__(self, in_channels, **kwargs):
        super().__init__()

        self.params = kwargs
        struct_layers_out_chnls = self.params["struct_layers_out_chnls"]
        self.dropout_prop = self.params["dropout_prop"]

        emb_dim = in_channels
        self.struct_layers = torch.nn.ModuleList()
        prev = emb_dim
        for i in struct_layers_out_chnls:
            self.struct_layers.append(
                torch.nn.Linear(in_features=prev, out_features=i)
            )
            prev = i
        self.final_dist_layer = torch.nn.Linear(in_features=prev, out_features=3)

    def forward(self, z):
        num_cells = z.shape[0]

        prev = z
        for i in self.struct_layers:
            h = F.dropout(F.relu(i(prev)), p=self.dropout_prop, training=self.training)
            prev = h
        coords = self.final_dist_layer(prev)

        return {"coords": coords}


class EmbRGCN(torch.nn.Module):
    def __init__(self, in_channels, **kwargs):
        super().__init__()

        self.params = kwargs
        out_channels = self.params["emb_layers_out_chnls"]
        self.dropout_prop = self.params["dropout_prop"]

        self.emb_layers = torch.nn.ModuleList()
        prev = in_channels
        for i in out_channels:
            emb_layers.append(
                WRGCNConv(in_channels=prev, out_channels=i, num_relations=2)
            )
            prev = i

        self.batch_norm_layers = torch.nn.ModuleList(torch.nn.BatchNorm1d(o) for o in out_channels)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_weight = data.edge_norm * data.edge_attr
        edge_type = data.edge_type
        cell_mask = data.cell_mask

        embs = []
        prev = x
        for i , j in zip(self.emb_layers, self.batch_norm_layers):
            h = F.relu(j(i(prev, edge_index, edge_weight, edge_type)))
            h = F.dropout(h, p=self.dropout_prop, training=self.training)
            embs.append(h)
            prev = h

        z = torch.cat(embs, dim=1)[cell_mask]

        return z


class EmbMLP(torch.nn.Module):
    def __init__(self, in_channels, **kwargs):
        super().__init__()

        self.params = kwargs
        out_channels = self.params["emb_layers_out_chnls"]
        self.dropout_prop = self.params["dropout_prop"]

        self.emb_layers = torch.nn.ModuleList()
        prev = in_channels
        for i in out_channels:
            self.emb_layers.append(
                torch.nn.Linear(prev, i)
            )
            prev = i

        self.batch_norm_layers = torch.nn.ModuleList(torch.nn.BatchNorm1d(o) for o in out_channels)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_weight = data.edge_norm * data.edge_attr
        edge_type = data.edge_type
        cell_mask = data.cell_mask

        embs = []
        prev = x[cell_mask]
        for i in self.emb_layers:
            h = F.relu(i(prev))
            h = F.dropout(h, p=self.dropout_prop, training=self.training)
            embs.append(h)
            prev = h

        z = torch.cat(embs, dim=1)

        return z


class SupFF(torch.nn.Module):
    def __init__(self, in_channels, components, **kwargs):
        super().__init__()

        self.params = kwargs
        emb_layers_out_chnls = self.params["emb_layers_out_chnls"]

        self.embedder = components["emb"](in_channels, **self.params)
        self.struct_module = components["struct"](sum(emb_layers_out_chnls), **self.params)

    def forward(self, data):
        embeddings = self.embedder(data)
        out = self.struct_module(embeddings)

        return out


class AuxExpDecMLP(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()

        self.params = kwargs
        dec_channels = self.params["aux_exp_dec_layers_out_chnls"]
        self.dropout_prop = self.params["dropout_prop"]

        self.dec_layers = torch.nn.ModuleList()
        prev = in_channels
        for i in dec_channels:
            self.dec_layers.append(
                torch.nn.Linear(prev, i)
            )
            prev = i

        self.final_dec_layer = torch.nn.Linear(in_features=prev, out_features=out_channels)

    def forward(self, z):
        prev = z
        for i in self.dec_layers:
            h = F.dropout(F.relu(i(prev)), p=self.dropout_prop, training=self.training)
            prev = h
        exp = self.final_dist_layer(prev)

        return {"exp": exp}


class AuxStructEncMLP(torch.nn.Module):
    def __init__(self, in_channels, **kwargs):
        super().__init__()

        self.params = kwargs
        out_channels = self.params["aux_struct_enc_layers_out_chnls"]
        self.dropout_prop = self.params["dropout_prop"]

        self.enc_layers = torch.nn.ModuleList()
        prev = in_channels
        for i in out_channels:
            self.enc_layers.append(
                torch.nn.Linear(prev, i)
            )
            prev = i

        self.final_dec_layer = torch.nn.Linear(in_features=prev, out_features=3)

    def forward(self, z):
        prev = z
        for i in self.enc_layers:
            h = F.dropout(F.relu(i(prev)), p=self.dropout_prop, training=self.training)
            prev = h
        coords = self.final_dist_layer(prev)

        return {"coords": coords}


class SupCVAE(torch.nn.Module):
    def __init__(self, in_channels, components, **kwargs):
        super().__init__()

        self.params = kwargs
        self.device = self.params["device"]
        self.dropout_prop = self.params["dropout_prop"]
        emb_layers_out_chnls = self.params["emb_layers_out_chnls"]
        aux_struct_enc_layers_out_chnls = self.params["aux_struct_enc_layers_out_chnls"]
        vae_enc_add_chnls = self.params["vae_enc_add_chnls"]
        self.vae_latent_dim = vae_latent_dim = self.params["vae_latent_dim"]

        self.embedder = components["emb"](in_channels, **self.params)
        self.emb_add_chnls = torch.nn.ModuleList()
        prev = sum(emb_layers_out_chnls)
        for i in vae_enc_add_chnls:
            self.emb_add_chnls.append(torch.nn.Linear(prev, i))
            prev = i
        self.emb_add_final_layer = torch.nn.Linear(in_features=prev, out_features=(2*vae_latent_dim))

        self.aux_struct_enc = components["struct_enc"](3, **self.params)
        self.aux_enc_add_chnls = torch.nn.ModuleList()
        prev = sum(aux_struct_enc_layers_out_chnls)
        for i in vae_enc_add_chnls:
            self.aux_enc_add_chnls.append(torch.nn.Linear(prev, i))
            prev = i
        self.aux_enc_add_final_layer = torch.nn.Linear(in_features=prev, out_features=(2*vae_latent_dim))

        self.struct_module = components["struct"](vae_latent_dim, **self.params)
        self.aux_exp_dec = components["exp_dec"](vae_latent_dim, in_channels, **self.params)

    def _sample_sn(self, size):
        return torch.normal(1, 1, size=size).to(self.device)

    def _sample_sn_like(self, tensor):
        return self._sample_sn(tensor.shape)

    def forward(self, data):
        emb = self.embedder(data)
        prev = emb
        for i in self.emb_add_chnls:
            h = F.dropout(F.relu(i(prev)), p=self.dropout_prop, training=self.training)
            prev = h
        dist = self.emb_add_final_layer(prev)
        print(dist) ####
        emb_mean = dist[:self.vae_latent_dim]
        emb_lstd = dist[self.vae_latent_dim:]
        emb_std = torch.exp(emb_lstd)
        emb_sample = self._sample_sn_like(emb_std) * emb_std + emb_mean

        aux_enc = self.embedder(data)
        prev = aux_enc
        for i in self.aux_enc_add_chnls:
            h = F.dropout(F.relu(i(prev)), p=self.dropout_prop, training=self.training)
            prev = h
        dist = self.aux_enc_add_final_layer(prev)
        aux_enc_mean = dist[:self.vae_latent_dim]
        aux_enc_lstd = dist[self.vae_latent_dim:]
        aux_enc_std = torch.exp(aux_enc_lstd)
        aux_enc_sample = _sample_sn_like(aux_enc_std) * aux_enc_std + aux_enc_mean

        out_coords = self.struct_module(aux_enc_sample)["coords"]
        out_exp = self.aux_exp_dec(emb_sample)["exp"]

        out = {
            "coords": out_coords,
            "exp": out_exp,
            "emb_mean": emb_mean,
            "emb_std": emb_std,
            "emb_lstd": emb_lstd,
            "aux_enc_mean": aux_enc_mean,
            "aux_enc_std": aux_enc_std,
            "aux_enc_lstd": aux_enc_lstd,
        }

        return out

    def sample_coords(self, data):
        emb = self.embedder(data)
        prev = emb
        for i in self.emb_add_chnls:
            h = F.dropout(F.relu(i(prev)), p=self.dropout_prop, training=self.training)
            prev = h
        dist = self.emb_add_final_layer(prev)
        emb_mean = dist[:self.vae_latent_dim]
        emb_lstd = dist[self.vae_latent_dim:]
        emb_std = torch.exp(emb_lstd)
        emb_sample = self._sample_sn_like(emb_std) * emb_std + emb_mean

        out_coords = self.struct_module(emb_sample)["coords"]

        out = {
            "coords": out_coords,
            "emb_mean": emb_mean,
            "emb_std": emb_std,
            "emb_lstd": emb_lstd,
        }


# class MixinRGCN(object):
#     def _get_gnn(self, in_channels, out_channels):
#         emb_layers = torch.nn.ModuleList()
#         prev = in_channels
#         for i in out_channels:
#             emb_layers.append(
#                 WRGCNConv(in_channels=prev, out_channels=i, num_relations=2)
#             )
#             prev = i

#         return emb_layers

#     def _gnn_fwd(self, data):
#         x = data.x
#         edge_index = data.edge_index
#         edge_weight = data.edge_norm * data.edge_attr
#         edge_type = data.edge_type
#         cell_mask = data.cell_mask

#         embs = []
#         prev = x
#         for i , j in zip(self.emb_layers, self.batch_norm_layers):
#             h = F.relu(j(i(prev, edge_index, edge_weight, edge_type)))
#             h = F.dropout(h, p=self.dropout_prop, training=self.training)
#             embs.append(h)
#             prev = h

#         z = torch.cat(embs, dim=1)[cell_mask]

#         return z


# class MixinMLP(object):
#     def _get_gnn(self, in_channels, out_channels):
#         emb_layers = torch.nn.ModuleList()
#         prev = in_channels
#         for i in out_channels:
#             emb_layers.append(
#                 torch.nn.Linear(prev, i)
#             )
#             prev = i

#         return emb_layers

#     def _gnn_fwd(self, data):
#         x = data.x
#         edge_index = data.edge_index
#         edge_weight = data.edge_norm * data.edge_attr
#         edge_type = data.edge_type
#         cell_mask = data.cell_mask

#         embs = []
#         prev = x[cell_mask]
#         for i in self.emb_layers:
#             h = F.relu(i(prev))
#             h = F.dropout(h, p=self.dropout_prop, training=self.training)
#             embs.append(h)
#             prev = h

#         z = torch.cat(embs, dim=1)

#         return z


# class SupRCGN(MixinRGCN, SupNet):
#     pass


# class SupMLP(MixinMLP, SupNet):
#     pass


# class SupBinRCGN(MixinRGCN, SupNetBin):
#     pass


# class SupBinMLP(MixinMLP, SupNetBin):
#     pass


# class SupSRCGN(MixinRGCN, SupNetS):
#     pass


# class SupSMLP(MixinMLP, SupNetS):
#     pass


# class SupLRRCGN(MixinRGCN, SupNetLR):
#     pass


# class SupLRMLP(MixinMLP, SupNetLR):
#     pass