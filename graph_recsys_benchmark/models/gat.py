import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from .base import GraphRecsysModel


class GATRecsysModel(GraphRecsysModel):
    def __init__(self, **kwargs):
        super(GATRecsysModel, self).__init__(**kwargs)

    def _init(self, **kwargs):
        self.if_use_features = kwargs['if_use_features']
        self.dropout = kwargs['dropout']

        if not self.if_use_features:
            self.x = torch.nn.Embedding(kwargs['num_nodes'], kwargs['emb_dim'], max_norm=1)

        self.conv1 = GATConv(
            kwargs['emb_dim'],
            kwargs['hidden_size'],
            heads=kwargs['num_heads'],
            dropout=kwargs['dropout']
        )
        self.conv2 = GATConv(
            kwargs['hidden_size'] * kwargs['num_heads'],
            kwargs['repr_dim'],
            heads=1,
            dropout=kwargs['dropout']
        )

        self.reset_parameters()

    def reset_parameters(self):
        if not self.if_use_features:
            torch.nn.init.uniform_(self.x.weight, -1.0, 1.0)
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, edge_index, x=None):
        if not self.if_use_features:
            x = self.x.weight
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.normalize(x)
        return x
