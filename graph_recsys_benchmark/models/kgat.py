import torch
import torch.nn.functional as F
from graph_recsys_benchmark.nn import KGATConv
from torch_geometric.nn.inits import glorot

from .base import GraphRecsysModel


class KGATRecsysModel(GraphRecsysModel):
    def __init__(self, **kwargs):
        super(KGATRecsysModel, self).__init__(**kwargs)

    def _init(self, **kwargs):
        self.if_use_features = kwargs['if_use_features']
        self.dropout = kwargs['dropout']

        if not self.if_use_features:
            # User and item representation are concatenation of three layers, so it should be divided by 3
            self.x = torch.nn.Embedding(kwargs['dataset']['num_nodes'], kwargs['emb_dim'], max_norm=1).weight
        else:
            raise NotImplementedError('Feature not implemented!')
        self.r = torch.nn.Embedding(kwargs['dataset'].num_edge_types, kwargs['emb_dim'], max_norm=1).weight
        self.proj_mat = torch.nn.Parameter(torch.Tensor(kwargs['emb_dim'], kwargs['emb_dim']))

        self.edge_index, self.edge_attr = self.update_graph_input(kwargs['dataset'])

        self.conv1 = KGATConv(
            kwargs['emb_dim'],
            kwargs['hidden_size'],
            self.x,
            self.proj_mat,
            self.r
        )
        self.conv2 = KGATConv(
            kwargs['hidden_size'],
            kwargs['hidden_size'] // 2,
            self.x,
            self.proj_mat,
            self.r
        )
        self.conv3 = KGATConv(
            kwargs['hidden_size'] // 2,
            kwargs['hidden_size'] // 4,
            self.x,
            self.proj_mat,
            self.r
        )

    def reset_parameters(self):
        if not self.if_use_features:
            glorot(self.x)
        glorot(self.r)
        glorot(self.proj_mat)
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()

    def forward(self):
        x, edge_index, edge_attr = self.x, self.edge_index, self.edge_attr
        x_1 = F.normalize(F.dropout(self.conv1(x, edge_index, edge_attr), p=self.dropout, training=self.training), p=2, dim=-1)
        x_2 = F.normalize(F.dropout(self.conv2(x_1, edge_index, edge_attr), p=self.dropout, training=self.training), p=2, dim=-1)
        x_3 = F.normalize(F.dropout(self.conv3(x_2, edge_index, edge_attr), p=self.dropout, training=self.training), p=2, dim=-1)
        return torch.cat([x_1, x_2, x_3], dim=-1)

    def predict(self, unids, inids):
        u_repr = self.cached_repr[unids]
        i_repr = self.cached_repr[inids]
        x = torch.sum(u_repr * i_repr, dim=-1)
        return x
