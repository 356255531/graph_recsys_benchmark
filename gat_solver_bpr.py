import argparse
import torch
import os
import numpy as np
import random as rd

from graph_recsys_benchmark.models import GATRecsysModel
from graph_recsys_benchmark.utils import get_folder_path
from graph_recsys_benchmark.solvers import BaseSolver

MODEL_TYPE = 'Graph'
LOSS_TYPE = 'BPR'
MODEL = 'GAT'

parser = argparse.ArgumentParser()

# Dataset params
parser.add_argument("--dataset", type=str, default='Movielens', help="")
parser.add_argument("--dataset_name", type=str, default='1m', help="")
parser.add_argument("--if_use_features", type=bool, default=False, help="")
parser.add_argument("--num_core", type=int, default=10, help="")
parser.add_argument("--num_feat_core", type=int, default=10, help="")

# Model params
parser.add_argument("--dropout", type=float, default=0, help="")
parser.add_argument("--emb_dim", type=int, default=64, help="")
parser.add_argument("--num_heads", type=int, default=1, help="")
parser.add_argument("--repr_dim", type=int, default=16, help="")
parser.add_argument("--hidden_size", type=int, default=64, help="")
# Train params
parser.add_argument("--init_eval", type=bool, default=True, help="")
parser.add_argument("--num_negative_samples", type=int, default=4, help="")
parser.add_argument("--num_neg_candidates", type=int, default=99, help="")

parser.add_argument("--device", type=str, default='cuda', help="")
parser.add_argument("--gpu_idx", type=str, default='0', help="")
parser.add_argument("--runs", type=int, default=100, help="")
parser.add_argument("--epochs", type=int, default=50, help="")
parser.add_argument("--batch_size", type=int, default=4096, help="")
parser.add_argument("--num_workers", type=int, default=4, help="")
parser.add_argument("--opt", type=str, default='adam', help="")
parser.add_argument("--lr", type=float, default=0.001, help="")
parser.add_argument("--weight_decay", type=float, default=0, help="")
parser.add_argument("--early_stopping", type=int, default=20, help="")
parser.add_argument("--save_epochs", type=list, default=[10, 15, 20], help="")
parser.add_argument("--save_every_epoch", type=int, default=20, help="")

args = parser.parse_args()


# Setup data and weights file path
data_folder, weights_folder, logger_folder = \
    get_folder_path(model=MODEL, dataset=args.dataset + args.dataset_name, loss_type=LOSS_TYPE)

# Setup device
if not torch.cuda.is_available() or args.device == 'cpu':
    device = 'cpu'
else:
    device = 'cuda:{}'.format(args.gpu_idx)

# Setup args
dataset_args = {
    'root': data_folder, 'dataset': args.dataset, 'name': args.dataset_name,
    'if_use_features': args.if_use_features, 'num_negative_samples': args.num_negative_samples,
    'num_core': args.num_core, 'num_feat_core': args.num_feat_core,
    'loss_type': LOSS_TYPE
}
model_args = {
    'model_type': MODEL_TYPE,
    'if_use_features': args.if_use_features,
    'emb_dim': args.emb_dim, 'hidden_size': args.hidden_size,
    'repr_dim': args.repr_dim, 'dropout': args.dropout,
    'num_heads': args.num_heads
}
train_args = {
    'init_eval': args.init_eval,
    'num_negative_samples': args.num_negative_samples, 'num_neg_candidates': args.num_neg_candidates,
    'opt': args.opt,
    'runs': args.runs, 'epochs': args.epochs, 'batch_size': args.batch_size,
    'num_workers': args.num_workers,
    'weight_decay': args.weight_decay, 'lr': args.lr, 'device': device,
    'weights_folder': os.path.join(weights_folder, str(model_args)),
    'logger_folder': os.path.join(logger_folder, str(model_args)),
    'save_epochs': args.save_epochs, 'save_every_epoch': args.save_every_epoch
}
print('dataset params: {}'.format(dataset_args))
print('task params: {}'.format(model_args))
print('train params: {}'.format(train_args))


def _negative_sampling(u_nid, num_negative_samples, train_splition, item_nid_occs):
    """
    The negative sampling methods used for generating the training batches
    :param u_nid:
    :return:
    """
    train_pos_unid_inid_map, test_pos_unid_inid_map, neg_unid_inid_map = train_splition
    # negative_inids = test_pos_unid_inid_map[u_nid] + neg_unid_inid_map[u_nid]
    # nid_occs = np.array([item_nid_occs[nid] for nid in negative_inids])
    # nid_occs = nid_occs / np.sum(nid_occs)
    # negative_inids = rd.choices(population=negative_inids, weights=nid_occs, k=num_negative_samples)
    # negative_inids = negative_inids

    negative_inids = test_pos_unid_inid_map[u_nid] + neg_unid_inid_map[u_nid]
    negative_inids = rd.choices(population=negative_inids, k=num_negative_samples)

    return negative_inids


class GATRecsysModel(GATRecsysModel):
    def loss(self, batch):
        if self.training:
            self.cached_repr = self.forward()
        pos_pred = self.predict(batch[:, 0], batch[:, 1])
        neg_pred = self.predict(batch[:, 0], batch[:, 2])

        loss = -(pos_pred - neg_pred).sigmoid().log().sum()

        return loss

    def update_graph_input(self, dataset):
        edge_index_np = np.hstack(list(dataset.edge_index_nps.values()))
        edge_index_np = np.hstack([edge_index_np, np.flip(edge_index_np, 0)])
        edge_index = torch.from_numpy(edge_index_np).long().to(train_args['device'])
        return edge_index


class GATSolver(BaseSolver):
    def __init__(self, model_class, dataset_args, model_args, train_args):
        super(GATSolver, self).__init__(model_class, dataset_args, model_args, train_args)

    def generate_candidates(self, dataset, u_nid):
        pos_i_nids = dataset.test_pos_unid_inid_map[u_nid]
        neg_i_nids = np.array(dataset.neg_unid_inid_map[u_nid])

        neg_i_nids_indices = np.array(rd.sample(range(neg_i_nids.shape[0]), train_args['num_neg_candidates']), dtype=int)

        return pos_i_nids, list(neg_i_nids[neg_i_nids_indices])


if __name__ == '__main__':
    dataset_args['_negative_sampling'] = _negative_sampling
    solver = GATSolver(GATRecsysModel, dataset_args, model_args, train_args)
    solver.run()
