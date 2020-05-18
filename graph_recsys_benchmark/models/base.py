import torch


class GraphRecsysModel(torch.nn.Module):
    def __init__(self, **kwargs):
        super(GraphRecsysModel, self).__init__()
        self._init(**kwargs)

        self.reset_parameters()

    def _init(self, **kwargs):
        raise NotImplementedError

    def reset_parameters(self):
        raise NotImplementedError

    def loss_func(self, pos_i_ratings, neg_i_ratings):
        raise NotImplementedError

    def update_graph_input(self, dataset):
        raise NotImplementedError

    def eval(self):
        super(GraphRecsysModel, self).eval()
        self.cached_repr = self.forward(self.x, self.edge_index)

    def predict(self, unids, inids):
        u_repr = self.cached_repr[unids]
        i_repr = self.cached_repr[inids]
        ratings = torch.sum(u_repr * i_repr, dim=1)
        return ratings

    def loss(self, pos_neg_pair_t):
        if self.training:
            self.cached_repr = self.forward(self.x, self.edge_index)
        pos_i_ratings = self.predict(pos_neg_pair_t[:, 0], pos_neg_pair_t[:, 1])
        neg_i_ratings = self.predict(pos_neg_pair_t[:, 0], pos_neg_pair_t[:, 2])
        return self.loss_func(pos_i_ratings, neg_i_ratings)


class MFRecsysModel(torch.nn.Module):
    def __init__(self, **kwargs):
        super(MFRecsysModel, self).__init__()
        self._init(**kwargs)

        self.reset_parameters()

    def _init(self, **kwargs):
        raise NotImplementedError

    def reset_parameters(self):
        raise NotImplementedError

    def loss_func(self, pos_i_ratings, neg_i_ratings):
        raise NotImplementedError

    def predict(self, unids, inids):
        return self.forward(unids, inids)

    def loss(self, pos_neg_pair_t):
        pos_i_ratings = self.predict(pos_neg_pair_t[:, 0], pos_neg_pair_t[:, 1])
        neg_i_ratings = self.predict(pos_neg_pair_t[:, 0], pos_neg_pair_t[:, 2])
        return self.loss_func(pos_i_ratings, neg_i_ratings)
