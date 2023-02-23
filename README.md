# Fed SSA: Federated Learning Singular-Spectrum Analysis
This repository is for the Experiment Section of the paper: "Federated Singular Spectrum Analysis for Spatio-Temporal Time
Series Modeling via Matrix Estimation"

Authors: Jiayu He, Tung-Anh Nguyen, Matloob Khushi, Nguyen H.Tran
# Software requirements:
- numpy, scipy, pytorch, Pillow, matplotlib.

- To download the dependencies: pip3 install -r requirements.txt

- The code can be run on any pc.
## Instruction to run the code for FedSSA

<pre></code>
python3 main.py --algorithm FedPG --learning_rate 0.005 --num_global_iters 50 --dim 100 --local_epochs 30 --ro 1 --dataset Traffic20 --window 100 --ro_auto 1 --missingVal 1
python3 main.py --algorithm FedPG --learning_rate 0.005 --num_global_iters 50 --dim 100 --local_epochs 30 --ro 1 --dataset Traffic20 --window 100 --ro_auto 1 --missingVal 0
python3 main.py --algorithm FedPG --learning_rate 0.005 --num_global_iters 50 --dim 80 --local_epochs 30 --ro 1 --dataset Elec20 --window 80 --ro_auto 1 --missingVal 1
python3 main.py --algorithm FedPG --learning_rate 0.005 --num_global_iters 50 --dim 80 --local_epochs 30 --ro 1 --dataset Elec20 --window 80 --ro_auto 1 --missingVal 0
<code></pre>

## Instruction to run the code for FedLSTM
<pre></code>
python3 main.py --dataset Imputed_Elec370 --algorithm FedLSTM --batch_size 64 --learning_rate 0.001 --subusers 0.1 --num_global_iters 100 --local_epochs 2
<code></pre>

## Dataset:
Electricity dataset
Traffic dataset
