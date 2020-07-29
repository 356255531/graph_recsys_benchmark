import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn.inits import glorot

from .base import GraphRecsysModel


class GATRecsysModel(GraphRecsysModel):
    def __init__(self, **kwargs):
        super(GATRecsysModel, self).__init__(**kwargs)

    def _init(self, **kwargs):
        self.if_use_features = kwargs['if_use_features']
        self.dropout = kwargs['dropout']

        if not self.if_use_features:
            self.x = torch.nn.Embedding(kwargs['dataset']['num_nodes'], kwargs['emb_dim'], max_norm=1).weight
        else:
            raise NotImplementedError('Feature not implemented!')
        self.edge_index = self.update_graph_input(kwargs['dataset'])

        self.conv1 = GATConv(
            kwargs['emb_dim'],
            kwargs['hidden_size'],
            heads=kwargs['num_heads'],
            dropout=kwargs['dropout']
        )
        self.conv2 = GATConv(
            kwargs['hidden_size'] * kwargs['num_heads'],
            kwargs['hidden_size'] // 2,
            heads=1,
            dropout=kwargs['dropout']
        )
        self.conv3 = GATConv(
            kwargs['hidden_size'] // 2 * kwargs['num_heads'],
            kwargs['hidden_size'] // 4,
            heads=1,
            dropout=kwargs['dropout']
        )

    def reset_parameters(self):
        if not self.if_use_features:
            glorot(self.x)
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()

    def forward(self):
        x, edge_index = self.x, self.edge_index
        x_1 = F.normalize(self.conv1(x, edge_index), p=2, dim=-1)
        x_2 = F.normalize(self.conv2(x_1, edge_index), p=2, dim=-1)
        x_3 = F.normalize(self.conv3(x_2, edge_index), p=2, dim=-1)
        return torch.cat([x_1, x_2, x_3], dim=-1)
