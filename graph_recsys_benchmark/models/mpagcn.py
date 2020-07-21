import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from .base import GraphRecsysModel


class MPAGCNChannel(torch.nn.Module):
    def __init__(self, **kwargs):
        super(MPAGCNChannel, self).__init__()
        self.num_steps = kwargs['num_steps']
        self.num_nodes = kwargs['num_nodes']

        self.gcn_layers = torch.nn.ModuleList()
        if kwargs['num_steps'] >= 2:
            self.gcn_layers.append(GCNConv(kwargs['emb_dim'], kwargs['hidden_size']))
            for i in range(kwargs['num_steps'] - 2):
                self.gcn_layers.append(GCNConv(kwargs['hidden_size'], kwargs['hidden_size']))
            self.gcn_layers.append(GCNConv(kwargs['hidden_size'], kwargs['repr_dim']))
        else:
            self.gcn_layers.append(GCNConv(kwargs['emb_dim'], kwargs['repr_dim']))

        self.reset_parameters()

    def reset_parameters(self):
        for module in self.gcn_layers:
            module.reset_parameters()

    def forward(self, x, edge_index_list):
        if len(edge_index_list) != self.num_steps:
            raise RuntimeError('Number of input adjacency matrices is not equal to step number!')

        for step_idx in range(self.num_steps - 1):
            x = F.relu(self.gcn_layers[step_idx](x, edge_index_list[step_idx]))
        x = self.gcn_layers[-1](x, edge_index_list[-1])
        x = F.normalize(x)
        return x


class MPAGCNRecsysModel(GraphRecsysModel):
    def __init__(self, **kwargs):
        super(MPAGCNRecsysModel, self).__init__(**kwargs)

    def _init(self, **kwargs):
        self.meta_path_steps = kwargs['meta_path_steps']
        self.if_use_features = kwargs['if_use_features']
        self.aggr = kwargs['aggr']

        if not self.if_use_features:
            self.x = torch.nn.Embedding(kwargs['num_nodes'], kwargs['emb_dim'], max_norm=1).weight
        else:
            raise NotImplementedError('Feature not implemented!')
        self.update_graph_input(kwargs['dataset'])

        self.mpagcn_channels = torch.nn.ModuleList()
        for num_steps in kwargs['meta_path_steps']:
            kwargs_cpy = kwargs.copy()
            kwargs_cpy['num_steps'] = num_steps
            self.mpagcn_channels.append(MPAGCNChannel(**kwargs_cpy))

        self.fc1 = torch.nn.Linear(2 * len(kwargs['meta_path_steps']) * kwargs['repr_dim'], kwargs['repr_dim'])
        self.fc2 = torch.nn.Linear(kwargs['repr_dim'], 1)

    def reset_parameters(self):
        for module in self.mpagcn_channels:
            module.reset_parameters()

    def forward(self):
        xs = [module(self.x, self.meta_path_edge_index_list[idx]) for idx, module in enumerate(self.mpagcn_channels)]
        if self.aggr == 'concat':
            x = x.view(x.shape[0], -1)
        elif self.aggr == 'mean':
            x = x.mean(dim=-2)
        elif self.aggr == 'att':
            atts = F.softmax(self.att(x).squeeze(-1), dim=-1).unsqueeze(-1)
            x = torch.sum(x * atts, dim=-2)
        else:
            raise NotImplemented('Other aggr methods not implemeted!')
        x = F.normalize(x)
        return x
