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
        self.dropout = kwargs['propa_dropout']

        if not self.if_use_features:
            self.x = torch.nn.Embedding(kwargs['dataset']['num_nodes'], kwargs['emb_dim'], max_norm=1).weight
        else:
            raise NotImplementedError('Feature not implemented!')
        self.update_graph_input(kwargs['dataset'])

        self.conv1 = GATConv(
            kwargs['emb_dim'],
            kwargs['hidden_size'],
            heads=kwargs['num_heads'],
            dropout=kwargs['aggr_dropout']
        )
        self.conv2 = GATConv(
            kwargs['hidden_size'] * kwargs['num_heads'],
            kwargs['repr_dim'],
            heads=1,
            dropout=kwargs['aggr_dropout']
        )

        self.fc1 = torch.nn.Linear(2 * kwargs['repr_dim'], kwargs['repr_dim'])
        self.fc2 = torch.nn.Linear(kwargs['repr_dim'], 1)

    def reset_parameters(self):
        if not self.if_use_features:
            glorot(self.x)
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        glorot(self.fc1.weight)
        glorot(self.fc2.weight)

    def forward(self):
        x, edge_index = self.x, self.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x
