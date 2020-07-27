import torch
import torch.nn.functional as F
from graph_recsys_benchmark.nn import MPAGATConv
from torch_geometric.nn.inits import glorot

from .base import GraphRecsysModel


class MPAGATChannel(torch.nn.Module):
    def __init__(self, **kwargs):
        super(MPAGATChannel, self).__init__()
        self.num_steps = kwargs['num_steps']
        self.num_nodes = kwargs['num_nodes']

        self.gat_layers = torch.nn.ModuleList()
        if kwargs['num_steps'] >= 2:
            self.gat_layers.append(MPAGATConv(kwargs['emb_dim'], kwargs['hidden_size'], heads=kwargs['num_heads'], dropout=kwargs['dropout']))
            for i in range(kwargs['num_steps'] - 2):
                self.gat_layers.append(MPAGATConv(kwargs['hidden_size'] * kwargs['num_heads'], kwargs['hidden_size'], heads=kwargs['num_heads'], dropout=kwargs['dropout']))
            self.gat_layers.append(MPAGATConv(kwargs['hidden_size'] * kwargs['num_heads'], kwargs['repr_dim'], heads=1, dropout=kwargs['dropout']))
        else:
            self.gat_layers.append(MPAGATConv(kwargs['emb_dim'], kwargs['repr_dim'], heads=1, dropout=kwargs['dropout']))

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


class MPAGATRecsysModel(GraphRecsysModel):
    def __init__(self, **kwargs):
        super(MPAGATRecsysModel, self).__init__(**kwargs)

    def _init(self, **kwargs):
        self.meta_path_steps = kwargs['meta_path_steps']
        self.if_use_features = kwargs['if_use_features']
        self.aggr = kwargs['aggr']

        if not self.if_use_features:
            self.x = torch.nn.Embedding(kwargs['num_nodes'], kwargs['emb_dim'], max_norm=1).weight
        else:
            raise NotImplementedError('Feature not implemented!')
        self.update_graph_input(kwargs['dataset'])

        self.mpagat_channels = torch.nn.ModuleList()
        for num_steps in kwargs['meta_path_steps']:
            kwargs_cpy = kwargs.copy()
            kwargs_cpy['num_steps'] = num_steps
            self.mpagat_channels.append(MPAGATChannel(**kwargs_cpy))

        if self.aggr == 'concat':
            self.fc1 = torch.nn.Linear(2 * len(kwargs['meta_path_steps']) * kwargs['repr_dim'], kwargs['repr_dim'])
        elif self.aggr == 'mean':
            self.fc1 = torch.nn.Linear(2 * kwargs['repr_dim'], kwargs['repr_dim'])
        elif self.aggr == 'att':
            self.att = torch.nn.Linear(kwargs['repr_dim'], 1)
            self.fc1 = torch.nn.Linear(2 * kwargs['repr_dim'], kwargs['repr_dim'])
        else:
            raise NotImplemented('Other aggr methods not implemeted!')
        self.fc2 = torch.nn.Linear(kwargs['repr_dim'], 1)

    def reset_parameters(self):
        for module in self.mpagat_channels:
            module.reset_parameters()
        glorot(self.fc1.weight)
        glorot(self.fc2.weight)
        if self.aggr == 'att':
            glorot(self.att.weight)

    def forward(self):
        x = [module(self.x, self.meta_path_edge_index_list[idx]).unsqueeze(-2) for idx, module in enumerate(self.mpagat_channels)]
        x = torch.cat(x, dim=-2)
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
