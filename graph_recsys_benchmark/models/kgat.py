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
            self.x = torch.nn.Embedding(kwargs['dataset']['num_nodes'], kwargs['emb_dim'] // 3, max_norm=1).weight
        else:
            raise NotImplementedError('Feature not implemented!')
        self.edge_type_vec = torch.nn.Embedding(kwargs['dataset'].num_edge_types, kwargs['emb_dim'] // 3, max_norm=1).weight
        self.proj_mat = torch.nn.Parameter(torch.Tensor(kwargs['emb_dim'] // 3, kwargs['emb_dim'] // 3))
        self.update_graph_input(kwargs['dataset'])

        self.conv1 = KGATConv(
            kwargs['emb_dim'] // 3,
            kwargs['emb_dim'] // 3,
        )
        self.conv2 = KGATConv(
            kwargs['emb_dim'] // 3,
            kwargs['emb_dim'] // 3,
        )
        self.conv3 = KGATConv(
            kwargs['emb_dim'] // 3,
            kwargs['emb_dim'] // 3,
        )

    def reset_parameters(self):
        if not self.if_use_features:
            glorot(self.x)
        glorot(self.edge_type_vec)
        glorot(self.proj_mat)
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()

    def forward(self):
        x, edge_index, edge_attr, edge_type_vec, proj_mat = self.x, self.edge_index, self.edge_attr, self.edge_type_vec, self.proj_mat
        x_1 = F.normalize(F.dropout(self.conv1(x, edge_index, edge_attr, edge_type_vec, proj_mat), p=self.dropout, training=self.training), p=2, dim=-1)
        x_2 = F.normalize(F.dropout(self.conv2(x_1, edge_index, edge_attr, edge_type_vec, proj_mat), p=self.dropout, training=self.training), p=2, dim=-1)
        x_3 = F.normalize(F.dropout(self.conv3(x_2, edge_index, edge_attr, edge_type_vec, proj_mat), p=self.dropout, training=self.training), p=2, dim=-1)
        return torch.cat([x_1, x_2, x_3], dim=-1)

    def predict(self, unids, inids):
        u_repr = self.cached_repr[unids]
        i_repr = self.cached_repr[inids]
        x = torch.sum(u_repr * i_repr, dim=-1)
        return x
