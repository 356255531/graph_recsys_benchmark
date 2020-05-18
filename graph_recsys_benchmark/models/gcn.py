import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


from .base import GraphRecsysModel


class GCNRecsysModel(GraphRecsysModel):
    def __init__(self, **kwargs):
        super(GCNRecsysModel, self).__init__(**kwargs)

    def _init(self, **kwargs):
        self.if_use_features = kwargs['if_use_features']
        self.dropout = kwargs['dropout']

        if not self.if_use_features:
            self.x = torch.nn.Embedding(kwargs['dataset']['num_nodes'], kwargs['emb_dim'], max_norm=1).weight
            self.edge_index = self.update_graph_input(kwargs['dataset'])
        else:
            raise NotImplementedError('Feature not implemented!')

        self.conv1 = GCNConv(kwargs['emb_dim'], kwargs['hidden_size'])
        self.conv2 = GCNConv(kwargs['hidden_size'], kwargs['repr_dim'])

    def reset_parameters(self):
        if not self.if_use_features:
            torch.nn.init.uniform_(self.x, -1.0, 1.0)
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.normalize(x)
        return x
