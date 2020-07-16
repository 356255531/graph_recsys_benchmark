screen -dmS gat_solver_bpr python3.7 gat_solver_bpr.py --emb_dim=32 --hidden_size=64 --repr_dim=16 --epochs=50 --gpu_idx='0'
screen -dmS gcn_solver_bpr python3.7 gcn_solver_bpr.py --emb_dim=32 --hidden_size=64 --repr_dim=16 --epochs=50 --gpu_idx='1'
screen -dmS nmf_solver_bpr python3.7 nmf_solver_bpr.py --factor_num=16 --epochs=50 --gpu_idx='2'
screen -dmS nmf_solver_bce python3.7 nmf_solver_bce.py --factor_num=16 --epochs=50 --gpu_idx='3'
screen -dmS sage_solver_bpr python3.7 sage_solver_bpr.py --emb_dim=32 --hidden_size=64 --repr_dim=16 --epochs=50 --gpu_idx='4'

screen -dmS mpagat_solver_bpr python3.7 mpagat_solver_bpr.py --emb_dim=32 --hidden_size=64 --repr_dim=16  --aggr='att' --epochs=50 --gpu_idx='5'
screen -dmS mpagat_solver_bpr python3.7 mpagat_solver_bpr.py --emb_dim=32 --hidden_size=64 --repr_dim=16  --aggr='mean' --epochs=50 --gpu_idx='6'
screen -dmS mpagat_solver_bpr python3.7 mpagat_solver_bpr.py --emb_dim=32 --hidden_size=64 --repr_dim=16  --aggr='concat' --epochs=50 --gpu_idx='7'
