# Fed SSA: Federated Learning Singular-Spectrum Analysis
This repository is for the Experiment Section of the paper: "Fed PCA: Federated Learning Singular-Spectrum Analysis"

Authors: Tung-Anh Nguyen, Nguyen H.Tran
# Software requirements:
- numpy, scipy, pytorch, Pillow, matplotlib.

- To download the dependencies: pip3 install -r requirements.txt

- The code can be run on any pc.
## Instruction to run the code

<pre></code>
python3 main.py --algorithm FedPG --learning_rate 0.00001 --dataset Elec20 --num_global_iters 100 --window 80 --dim 80 --subusers 0.1 --local_epochs 30
python3 main.py --algorithm FedPE --learning_rate 0.00001 --dataset Elec20 --num_global_iters 100 --window 80 --dim 80 --subusers 0.1 --local_epochs 30
<code></pre>


## Dataset:
https://www.kaggle.com/competitions/store-sales-time-series-forecasting