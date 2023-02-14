# Fed SSA: Federated Learning Singular-Spectrum Analysis
This repository is for the Experiment Section of the paper: "Fed PCA: Federated Learning Singular-Spectrum Analysis"

Authors: Tung-Anh Nguyen, Nguyen H.Tran
# Software requirements:
- numpy, scipy, pytorch, Pillow, matplotlib.

- To download the dependencies: pip3 install -r requirements.txt

- The code can be run on any pc.
## Instruction to run the code for FedSSA

<pre></code>
python3 main.py --algorithm FedPG --learning_rate 0.00001 --dataset Elec20 --num_global_iters 100 --window 80 --dim 80 --subusers 0.1 --local_epochs 30
python3 main.py --algorithm FedPE --learning_rate 0.00001 --dataset Elec20 --num_global_iters 100 --window 80 --dim 80 --subusers 0.1 --local_epochs 30
<code></pre>

## Instruction to run the code for FedLSTM
<pre></code>
!python3 main.py --dataset Imputed_Elec370 --algorithm FedLSTM --batch_size 64 --learning_rate 0.001 --subusers 0.1 --num_global_iters 100 --local_epochs 2
<code></pre>

## Dataset:
Electricity dataset