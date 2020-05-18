import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class PAPAGCNChannel(torch.nn.Module):
    def __init__(self, num_steps, num_nodes, emb_dim, hidden_size, repr_dim, if_use_features):
        super(PAPAGCNChannel, self).__init__()
        self.num_steps = num_steps
        self.num_nodes = num_nodes
        self.if_use_features = if_use_features

        self.gcn_layers = torch.nn.ModuleList()
        self.gcn_layers.append(GCNConv(emb_dim, hidden_size))
        for i in range(num_steps - 2):
            self.gcn_layers.append(GCNConv(hidden_size, hidden_size))
        self.gcn_layers.append(GCNConv(hidden_size, repr_dim))

        self.reset_parameters()

    def reset_parameters(self):
        for module in self.gcn_layers:
            module.reset_parameters()

    def forward(self, edge_index_list, x):
        if len(edge_index_list) != self.num_steps:
            raise RuntimeError('Number of input adjacency matrices is not equal to step number!')

        for step_idx in range(self.num_steps - 1):
            x = F.relu(self.gcn_layers[step_idx](x, edge_index_list[step_idx]))
        x = self.gcn_layers[-1](x, edge_index_list[-1])
        x = F.normalize(x)
        return x


class PAGAGCN(torch.nn.Module):
    def __init__(self, meta_path_steps, num_nodes, emb_dim, hidden_size, repr_dim, if_use_features, aggr='concat'):
        super(PAGAGCN, self).__init__()
        self.meta_path_steps = meta_path_steps
        self.if_use_features = if_use_features
        self.aggr = aggr

        if not if_use_features:
            self.x = torch.nn.Embedding(num_nodes, emb_dim, max_norm=1)

        self.pagcn_channels = torch.nn.ModuleList()
        for num_steps in meta_path_steps:
            self.pagcn_channels.append(PAPAGCNChannel(num_steps, num_nodes, emb_dim, hidden_size, repr_dim, if_use_features))

    def forward(self, meta_path_edge_index_list, x=None):
        if not self.if_use_features:
            x = self.x.weight
        xs = [module(meta_path_edge_index_list[idx], x) for idx, module in enumerate(self.pagcn_channels)]
        if self.aggr == 'concat':
            x = torch.cat(xs, dim=1)
        else:
            raise NotImplemented('Other aggr methods not implemeted!')
        x = F.normalize(x)
        return x