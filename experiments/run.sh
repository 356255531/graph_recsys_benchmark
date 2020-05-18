screen -dmS gat_solver_bce python3.7 gat_solver_bce.py \
--dataset_name=Movielens --dataset_name=1m --if_use_features=false --num_core=10 --num_feat_core=10 \
--dropout=0 --emb_dim=64 --repr_dim=16 --hidden_size=64 \
--init_eval=true --num_negative_samples=4 --num_neg_candidates=99 \
--device=cuda --gpu_idx=0 --runs=10 --epochs=30 --batch_size=4096 \
--num_workers=8 --opt=adam --lr=0.001 --weight_decay=0 \
--save_epochs="15,20,25" --save_every_epoch=1

screen -dmS pagagat_solver_bpr python3.7 pagagat_solver_bpr.py \
--dataset_name=Movielens --dataset_name=1m --if_use_features=false --num_core=10 --num_feat_core=10 \
--dropout=0 --emb_dim=64 --num_heads=1 --repr_dim=16 --hidden_size=64 \
--meta_path_steps="2,2,2,2,2,2,2,2,2,2" --aggr=concat \
--init_eval=true --num_negative_samples=4 --num_neg_candidates=99 \
--device=cuda --gpu_idx=0 --runs=10 --epochs=30 --batch_size=4096 \
--num_workers=8 --opt=adam --lr=0.001 --weight_decay=0 \
--save_epochs="15,20,25" --save_every_epoch=1;
