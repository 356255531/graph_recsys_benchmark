import torch
import torch.nn.functional as F
from graph_recsys_benchmark.nn import KGATConv

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
        self.update_graph_input(kwargs['dataset'])

        self.edge_type_vec = torch.nn.Embedding(len(list(kwargs['dataset'].edge_index_nps.values())), kwargs['emb_dim'], max_norm=1).weight

        self.conv1 = KGATConv(
            kwargs['emb_dim'],
            kwargs['repr_dim'],
            heads=kwargs['num_heads'],
            dropout=kwargs['dropout']
        )
        self.conv2 = KGATConv(
            kwargs['repr_dim'] * kwargs['num_heads'],
            kwargs['repr_dim'],
            heads=1,
            dropout=kwargs['dropout']
        )
        self.conv3 = KGATConv(
            kwargs['hidden_size'] * kwargs['num_heads'],
            kwargs['repr_dim'],
            heads=1,
            dropout=kwargs['dropout']
        )

        self.fc1 = torch.nn.Linear(2 * kwargs['repr_dim'], kwargs['repr_dim'])
        self.fc2 = torch.nn.Linear(kwargs['repr_dim'], 1)

    def reset_parameters(self):
        if not self.if_use_features:
            torch.nn.init.uniform_(self.x, -1.0, 1.0)
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        torch.nn.init.uniform_(self.fc1.weight, -1.0, 1.0)
        torch.nn.init.uniform_(self.fc2.weight, -1.0, 1.0)

    def forward(self):
        x_1 = F.relu(self.conv1(self.x, self.edge_index))
        x_1 = F.dropout(x_1, p=self.dropout, training=self.training)
        x_2 = F.relu(self.conv2(x_1, self.edge_index))
        x_2 = F.dropout(x_2, p=self.dropout, training=self.training)
        x_3 = F.relu(self.conv3(x_2, self.edge_index))
        x_3 = F.dropout(x_3, p=self.dropout, training=self.training)
        x = F.normalize(torch.cat([x_1, x_2, x_3], dim=-1))
        return x

    def predict(self, unids, inids):
        u_repr = self.cached_repr[unids]
        i_repr = self.cached_repr[inids]
        x = torch.sum(u_repr * i_repr, dim=-1)
        return x
