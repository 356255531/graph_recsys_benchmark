from .base import GraphRecsysModel
import torch


class Node2VecRecsysModel(GraphRecsysModel):
    def __init__(self, **kwargs):
        super(Node2VecRecsysModel, self).__init__(**kwargs)

    def _init(self, **kwargs):
        self.dropout = kwargs['dropout']

        self.random_walk_model = kwargs['random_walk_model']
        with torch.no_grad():
            self.random_walk_model.eval()
            self.cached_repr = self.random_walk_model()

    def eval(self):
        raise NotImplementedError
