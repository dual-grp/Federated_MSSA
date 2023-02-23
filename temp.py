
python3 main.py --algorithm FedPG --learning_rate 0.005 --num_global_iters 50 --dim 100 --local_epochs 30 --ro 1 --dataset Traffic20 --window 100 --ro_auto 1 --missingVal 1
python3 main.py --algorithm FedPG --learning_rate 0.005 --num_global_iters 50 --dim 100 --local_epochs 30 --ro 1 --dataset Traffic20 --window 100 --ro_auto 1 --missingVal 0

python3 main.py --algorithm FedPG --learning_rate 0.005 --num_global_iters 50 --dim 80 --local_epochs 30 --ro 1 --dataset Elec20 --window 80 --ro_auto 1 --missingVal 1
python3 main.py --algorithm FedPG --learning_rate 0.005 --num_global_iters 50 --dim 80 --local_epochs 30 --ro 1 --dataset Elec20 --window 80 --ro_auto 1 --missingVal 0
