import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from .base import GraphRecsysModel


class PAPAGATChannel(torch.nn.Module):
    def __init__(self, **kwargs):
        super(PAPAGATChannel, self).__init__()
        self.num_steps = kwargs['num_steps']
        self.num_nodes = kwargs['num_nodes']

        self.gat_layers = torch.nn.ModuleList()
        if kwargs['num_steps'] >= 2:
            self.gat_layers.append(GATConv(kwargs['emb_dim'], kwargs['hidden_size'], heads=kwargs['num_heads'], dropout=kwargs['dropout']))
            for i in range(kwargs['num_steps'] - 2):
                self.gat_layers.append(GATConv(kwargs['hidden_size'] * kwargs['num_heads'], kwargs['hidden_size'], heads=kwargs['num_heads'], dropout=kwargs['dropout']))
            self.gat_layers.append(GATConv(kwargs['hidden_size'] * kwargs['num_heads'], kwargs['repr_dim'], heads=1, dropout=kwargs['dropout']))
        else:
            self.gat_layers.append(GATConv(kwargs['emb_dim'], kwargs['repr_dim'], heads=1, dropout=kwargs['dropout']))

        self.reset_parameters()

    def reset_parameters(self):
        for module in self.gat_layers:
            module.reset_parameters()

    def forward(self, x, edge_index_list):
        if len(edge_index_list) != self.num_steps:
            raise RuntimeError('Number of input adjacency matrices is not equal to step number!')

        for step_idx in range(self.num_steps - 1):
            x = F.relu(self.gat_layers[step_idx](x, edge_index_list[step_idx]))
        x = self.gat_layers[-1](x, edge_index_list[-1])
        x = F.normalize(x)
        return x


class PAGAGATRecsysModel(GraphRecsysModel):
    def __init__(self, **kwargs):
        super(PAGAGATRecsysModel, self).__init__(**kwargs)

    def _init(self, **kwargs):
        self.meta_path_steps = kwargs['meta_path_steps']
        self.if_use_features = kwargs['if_use_features']
        self.aggr = kwargs['aggr']

        if not self.if_use_features:
            self.x = torch.nn.Embedding(kwargs['num_nodes'], kwargs['emb_dim'], max_norm=1).weight
        else:
            raise NotImplementedError('Feature not implemented!')
        self.x, self.meta_path_edge_index_list = self.update_graph_input(kwargs['dataset'])

        self.pagagat_channels = torch.nn.ModuleList()
        for num_steps in kwargs['meta_path_steps']:
            kwargs_cpy = kwargs.copy()
            kwargs_cpy['num_steps'] = num_steps
            self.pagagat_channels.append(PAPAGATChannel(**kwargs_cpy))

        self.fc1 = torch.nn.Linear(2 * len(kwargs['meta_path_steps']) * kwargs['repr_dim'], kwargs['repr_dim'])
        self.fc2 = torch.nn.Linear(kwargs['repr_dim'], 1)

    def reset_parameters(self):
        for module in self.pagagat_channels:
            module.reset_parameters()

    def forward(self):
        xs = [module(self.x, self.meta_path_edge_index_list[idx]) for idx, module in enumerate(self.pagagat_channels)]
        if self.aggr == 'concat':
            x = torch.cat(xs, dim=1)
        else:
            raise NotImplemented('Other aggr methods not implemeted!')
        x = F.normalize(x)
        return x
