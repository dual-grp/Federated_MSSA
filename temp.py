
python3 main.py --algorithm FedPG --learning_rate 0.005 --num_global_iters 100 --dim 80 --local_epochs 30 --ro 1 --dataset Elec20 --window 80 --ro_auto 1
python3 main.py --algorithm FedPG --learning_rate 0.005 --num_global_iters 100 --dim 160 --local_epochs 30 --ro 1 --dataset Elec20 --window 160 --ro_auto 1

python3 main.py --algorithm FedPG --learning_rate 0.005 --num_global_iters 100 --dim 80 --local_epochs 30 --ro 1 --dataset Elec370 --window 80 --ro_auto 1
python3 main.py --algorithm FedPG --learning_rate 0.005 --num_global_iters 100 --dim 160 --local_epochs 30 --ro 1 --dataset Elec370 --window 160 --ro_auto 1

python3 main.py --algorithm FedPG --learning_rate 0.005 --num_global_iters 100 --dim 80 --local_epochs 30 --ro 1 --dataset Traffic20 --window 80 --ro_auto 1


python3 main.py --algorithm FedPG --learning_rate 0.005 --num_global_iters 10 --dim 3 --local_epochs 30 --ro 0.01 --dataset debug   