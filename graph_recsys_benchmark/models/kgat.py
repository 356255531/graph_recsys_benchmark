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
            self.x = torch.nn.Embedding(kwargs['dataset']['num_nodes'], kwargs['emb_dim'], max_norm=1).weight
        else:
            raise NotImplementedError('Feature not implemented!')
        self.edge_type_vec = torch.nn.Embedding(kwargs['dataset'].num_edge_types, kwargs['repr_dim'], max_norm=1).weight
        self.proj_mat = torch.nn.Parameter(torch.Tensor(kwargs['emb_dim'], kwargs['repr_dim']))
        self.update_graph_input(kwargs['dataset'])

        self.conv1 = KGATConv(
            kwargs['emb_dim'],
            kwargs['emb_dim'],
            dropout=kwargs['dropout']
        )
        self.conv2 = KGATConv(
            kwargs['emb_dim'],
            kwargs['emb_dim'],
            dropout=kwargs['dropout']
        )
        self.conv3 = KGATConv(
            kwargs['emb_dim'],
            kwargs['emb_dim'],
            dropout=kwargs['dropout']
        )

        self.fc1 = torch.nn.Linear(2 * kwargs['emb_dim'], kwargs['emb_dim'])
        self.fc2 = torch.nn.Linear(kwargs['emb_dim'], 1)

    def reset_parameters(self):
        if not self.if_use_features:
            glorot(self.x)
        glorot(self.edge_type_vec)
        glorot(self.proj_mat)
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        glorot(self.fc1.weight)
        glorot(self.fc2.weight)

    def forward(self):
        x, edge_index, edge_attr, edge_type_vec, proj_mat = self.x, self.edge_index, self.edge_attr, self.edge_type_vec, self.proj_mat
        x_1 = F.dropout(x, p=self.dropout, training=self.training)
        x_1 = self.conv1(x_1, edge_index, edge_attr, edge_type_vec, proj_mat)
        x_2 = F.dropout(F.elu(x_1), p=self.dropout, training=self.training)
        x_2 = self.conv2(x_2, edge_index, edge_attr, edge_type_vec, proj_mat)
        x_3 = F.dropout(F.elu(x_2), p=self.dropout, training=self.training)
        x_3 = self.conv3(x_3, edge_index, edge_attr, edge_type_vec, proj_mat)
        return torch.cat([x_1, x_2, x_3], dim=-1)

    def predict(self, unids, inids):
        u_repr = self.cached_repr[unids]
        i_repr = self.cached_repr[inids]
        x = torch.sum(u_repr * i_repr, dim=-1)
        return x
