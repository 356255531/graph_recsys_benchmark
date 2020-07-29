import os
from graph_recsys_benchmark.utils import *

MODEL = 'MPAGAT'


DATASET = 'Movielens'
DATASET_NAME = '1m'

CF_LOSS_TYPE = 'BPR'

model_args = {
    'model_type': 'Graph',
    'if_use_features': False,
    'emb_dim': 64, 'hidden_size': 128,
    'repr_dim': 16, 'dropout': 0
}

_, _, logger_folder = \
    get_folder_path(model=MODEL, dataset=DATASET + DATASET_NAME, loss_type=CF_LOSS_TYPE)


global_logger_file_path = os.path.join(logger_folder, str(model_args))

HRs_per_run_np, NDCGs_per_run_np, AUC_per_run_np, train_loss_per_run_np, eval_loss_per_run_np, last_run = \
            load_global_logger(global_logger_file_path)

print(NDCGs_per_run_np)
