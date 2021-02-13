
import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing

class WRGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_relations, **kwargs):
        super().__init__(aggr="add", node_dim=0, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations

        self.weight = Parameter(torch.Tensor(num_relations, in_channels, out_channels))
        self.root = Parameter(torch.Tensor(in_channels, out_channels))
        self.bias = Parameter(torch.Tensor(out_channels))

        print(self.weight) ####

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


class SupNet(torch.nn.Module):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.params = kwargs
        gnn_layers_out_chnls = self.params["gnn_layers_out_chnls"]
        dist_layers_out_chnls = self.params["dist_layers_out_chnls"]
        self.dropout_prop = self.params["dropout_prop"]

        self.gnn_layers = self._get_gnn(in_channels, gnn_layers_out_chnls)

        emb_dim = sum(gnn_layers_out_chnls) * 2
        self.dist_layers = torch.nn.ModuleList()
        prev = emb_dim
        for i in dist_layers_out_chnls:
            self.dist_layers.append(
                torch.nn.Conv2d(in_channels=prev, out_channels=i, kernel_size=1)
            )
            prev = i
        self.final_dist_layer = torch.nn.Conv2d(in_channels=prev, out_channels=2, kernel_size=1)

    def _get_gnn(self, in_channels, out_channels):
        raise NotImplementedError

    def forward(self, data):
        z = self._gnn_fwd(data)
        num_cells = z.shape[0]

        rtile = z.unsqueeze(0).expand(num_cells, -1, -1)
        ctile = z.unsqueeze(1).expand(-1, num_cells, -1)
        pairs = torch.cat((rtile, ctile), dim=2)
        pairs.unsqueeze_(0)

        prev = pairs.permute(0, 3, 1, 2)
        for i in self.dist_layers:
            h = F.dropout(F.relu(i(prev)), p=self.dropout_prop, training=self.training)
            prev = h
        dists = self.final_dist_layer(prev)
        # print(dists.shape) ####
        dists = dists.permute(0, 2, 3, 1).squeeze_(dim=0)
        # print(dists.shape) ####

        return {"dists": dists}

    def _gnn_fwd(self, data):
        raise NotImplementedError


class SupRCGN(SupNet):
    def _get_gnn(self, in_channels, out_channels):
        gnn_layers = torch.nn.ModuleList()
        prev = in_channels
        for i in out_channels:
            gnn_layers.append(
                WRGCNConv(in_channels=prev, out_channels=i, num_relations=2)
            )
            prev = i

        return gnn_layers

    def _gnn_fwd(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_weight = data.edge_norm * data.edge_attr
        edge_type = data.edge_type
        cell_mask = data.cell_mask
        # print(data.edge_norm.dtype, data.edge_attr.dtype) ####

        embs = []
        prev = x
        for i in self.gnn_layers:
            h = F.relu(i(prev, edge_index, edge_weight, edge_type))
            h = F.dropout(h, p=self.dropout_prop, training=self.training)
            embs.append(h)
            prev = h

        z = torch.cat(embs, dim=1)[cell_mask]
        # print(z) ####

        return z