# Fed SSA: Federated Learning Singular-Spectrum Analysis
This repository is for the Experiment Section of the paper: "Federated Singular Spectrum Analysis for Spatio-Temporal Time
Series Modeling via Matrix Estimation"

Authors: Jiayu He, Tung-Anh Nguyen, Matloob Khushi, Nguyen H.Tran
# Software requirements:
- numpy, scipy, pytorch, Pillow, matplotlib.

- To download the dependencies: pip3 install -r requirements.txt

- The code can be run on any pc.
## Instruction to run the code for FedSSA univariate-time series on client
<pre></code>
python3 main.py --algorithm FedPG --learning_rate 0.005 --num_global_iters 50 --dim 100 --local_epochs 30 --ro 1 --dataset Traffic20 --window 100 --ro_auto 1 --missingVal 1
python3 main.py --algorithm FedPG --learning_rate 0.005 --num_global_iters 50 --dim 100 --local_epochs 30 --ro 1 --dataset Traffic20 --window 100 --ro_auto 1 --missingVal 0
python3 main.py --algorithm FedPG --learning_rate 0.005 --num_global_iters 50 --dim 80 --local_epochs 30 --ro 1 --dataset Elec20 --window 80 --ro_auto 1 --missingVal 1
python3 main.py --algorithm FedPG --learning_rate 0.005 --num_global_iters 50 --dim 80 --local_epochs 30 --ro 1 --dataset Elec20 --window 80 --ro_auto 1 --missingVal 0
<code></pre>

## Running command for FedSSA with missing percentage with univariate-time series on client
<pre></code>
python3 main.py --algorithm FedPG --learning_rate 0.005 --num_global_iters 50 --dim 100 --local_epochs 30 --ro 1 --dataset Traffic20 --window 100 --ro_auto 1 --missingVal 20
python3 main.py --algorithm FedPG --learning_rate 0.005 --num_global_iters 50 --dim 100 --local_epochs 30 --ro 1 --dataset Traffic20 --window 100 --ro_auto 1 --missingVal 40
<code></pre>
## Instruction to run the code for FedLSTM
<pre></code>
python3 main.py --dataset Imputed_Elec370 --algorithm FedLSTM --batch_size 64 --learning_rate 0.001 --subusers 0.1 --num_global_iters 100 --local_epochs 2

python3 main.py --dataset Imputed_Traff20 --algorithm FedLSTM --batch_size 64 --learning_rate 0.001 --subusers 0.1 --num_global_iters 100 --local_epochs 2
<code></pre>

## Running command for FedSSA with missing data in multivariate-time series on client for Electricity370 dataset
<pre></code>
python3 main.py --algorithm FedPG --learning_rate 0.005 --num_global_iters 50 --dim 80 --local_epochs 30 --ro 1 --dataset Elec370 --window 80 --ro_auto 1 --missingVal 0 --mulTS 1 --fac 1 --num_users 10

python3 main.py --algorithm FedPG --learning_rate 0.005 --num_global_iters 50 --dim 80 --local_epochs 30 --ro 1 --dataset Elec370 --window 80 --ro_auto 1 --missingVal 20 --mulTS 1 --fac 1 --num_users 10

python3 main.py --algorithm FedPG --learning_rate 0.005 --num_global_iters 50 --dim 80 --local_epochs 30 --ro 1 --dataset Elec370 --window 80 --ro_auto 1 --missingVal 40 --mulTS 1 --fac 1 --num_users 10

python3 main.py --algorithm FedPG --learning_rate 0.005 --num_global_iters 50 --dim 80 --local_epochs 30 --ro 1 --dataset Elec370 --window 80 --ro_auto 1 --missingVal 60 --mulTS 1 --fac 1 --num_users 10

python3 main.py --algorithm FedPG --learning_rate 0.005 --num_global_iters 50 --dim 80 --local_epochs 30 --ro 1 --dataset Elec370 --window 80 --ro_auto 1 --missingVal 80 --mulTS 1 --fac 1 --num_users 10

python3 main.py --algorithm FedPG --learning_rate 0.005 --num_global_iters 50 --dim 80 --local_epochs 30 --ro 1 --dataset Elec370 --window 80 --ro_auto 1 --missingVal 0 --mulTS 1 --fac 1 --num_users 18

python3 main.py --algorithm FedPG --learning_rate 0.005 --num_global_iters 50 --dim 80 --local_epochs 30 --ro 1 --dataset Elec370 --window 80 --ro_auto 1 --missingVal 0 --mulTS 1 --fac 1 --num_users 37

python3 main.py --algorithm FedPG --learning_rate 0.005 --num_global_iters 50 --dim 80 --local_epochs 30 --ro 1 --dataset Elec370 --window 80 --ro_auto 1 --missingVal 20 --mulTS 1 --fac 1 --num_users 37

python3 main.py --algorithm FedPG --learning_rate 0.005 --num_global_iters 50 --dim 80 --local_epochs 30 --ro 1 --dataset Elec370 --window 80 --ro_auto 1 --missingVal 40 --mulTS 1 --fac 1 --num_users 37

python3 main.py --algorithm FedPG --learning_rate 0.005 --num_global_iters 50 --dim 80 --local_epochs 30 --ro 1 --dataset Elec370 --window 80 --ro_auto 1 --missingVal 60 --mulTS 1 --fac 1 --num_users 37

python3 main.py --algorithm FedPG --learning_rate 0.005 --num_global_iters 50 --dim 80 --local_epochs 30 --ro 1 --dataset Elec370 --window 80 --ro_auto 1 --missingVal 0 --mulTS 1 --fac 1 --num_users 74

python3 main.py --algorithm FedPG --learning_rate 0.005 --num_global_iters 50 --dim 80 --local_epochs 30 --ro 1 --dataset Elec370 --window 80 --ro_auto 1 --missingVal 20 --mulTS 1 --fac 1 --num_users 74

python3 main.py --algorithm FedPG --learning_rate 0.005 --num_global_iters 50 --dim 80 --local_epochs 30 --ro 1 --dataset Elec370 --window 80 --ro_auto 1 --missingVal 40 --mulTS 1 --fac 1 --num_users 74

python3 main.py --algorithm FedPG --learning_rate 0.005 --num_global_iters 50 --dim 80 --local_epochs 30 --ro 1 --dataset Elec370 --window 80 --ro_auto 1 --missingVal 60 --mulTS 1 --fac 1 --num_users 74

python3 main.py --algorithm FedPG --learning_rate 0.005 --num_global_iters 50 --dim 80 --local_epochs 30 --ro 1 --dataset Elec370 --window 80 --ro_auto 1 --missingVal 80 --mulTS 1 --fac 1 --num_users 74

python3 main.py --algorithm FedPG --learning_rate 0.005 --num_global_iters 50 --dim 80 --local_epochs 30 --ro 1 --dataset Elec370 --window 80 --ro_auto 1 --missingVal 0 --mulTS 1 --fac 1 --num_users 148

python3 main.py --algorithm FedPG --learning_rate 0.005 --num_global_iters 50 --dim 80 --local_epochs 30 --ro 1 --dataset Elec370 --window 80 --ro_auto 1 --missingVal 20 --mulTS 1 --fac 1 --num_users 148

python3 main.py --algorithm FedPG --learning_rate 0.005 --num_global_iters 50 --dim 80 --local_epochs 30 --ro 1 --dataset Elec370 --window 80 --ro_auto 1 --missingVal 40 --mulTS 1 --fac 1 --num_users 148

python3 main.py --algorithm FedPG --learning_rate 0.005 --num_global_iters 50 --dim 80 --local_epochs 30 --ro 1 --dataset Elec370 --window 80 --ro_auto 1 --missingVal 60 --mulTS 1 --fac 1 --num_users 148

python3 main.py --algorithm FedPG --learning_rate 0.005 --num_global_iters 50 --dim 80 --local_epochs 30 --ro 1 --dataset Elec370 --window 80 --ro_auto 1 --missingVal 80 --mulTS 1 --fac 1 --num_users 148

python3 main.py --algorithm FedPG --learning_rate 0.005 --num_global_iters 5 --dim 80 --local_epochs 10 --ro 1 --dataset Elec370 --window 80 --ro_auto 1 --missingVal 0 --mulTS 1 --fac 1 --num_users 370

python3 main.py --algorithm FedPG --learning_rate 0.005 --num_global_iters 5 --dim 80 --local_epochs 10 --ro 1 --dataset Elec370 --window 80 --ro_auto 1 --missingVal 20 --mulTS 1 --fac 1 --num_users 370

python3 main.py --algorithm FedPG --learning_rate 0.005 --num_global_iters 5 --dim 80 --local_epochs 10 --ro 1 --dataset Elec370 --window 80 --ro_auto 1 --missingVal 40 --mulTS 1 --fac 1 --num_users 370

python3 main.py --algorithm FedPG --learning_rate 0.005 --num_global_iters 5 --dim 80 --local_epochs 10 --ro 1 --dataset Elec370 --window 80 --ro_auto 1 --missingVal 60 --mulTS 1 --fac 1 --num_users 370

python3 main.py --algorithm FedPG --learning_rate 0.005 --num_global_iters 5 --dim 80 --local_epochs 10 --ro 1 --dataset Elec370 --window 80 --ro_auto 1 --missingVal 80 --mulTS 1 --fac 1 --num_users 370
<code></pre>

## Running command for FedSSA with missing data in multivariate-time series on client for Traffic860 dataset
<pre></code>
python3 main.py --algorithm FedPG --learning_rate 0.005 --num_global_iters 5 --dim 100 --local_epochs 10 --ro 1 --dataset Traffic860 --window 100 --ro_auto 1 --missingVal 0 --mulTS 1 --fac 1 --num_users 10

python3 main.py --algorithm FedPG --learning_rate 0.005 --num_global_iters 5 --dim 100 --local_epochs 10 --ro 1 --dataset Traffic860 --window 100 --ro_auto 1 --missingVal 20 --mulTS 1 --fac 1 --num_users 10

python3 main.py --algorithm FedPG --learning_rate 0.005 --num_global_iters 5 --dim 100 --local_epochs 10 --ro 1 --dataset Traffic860 --window 100 --ro_auto 1 --missingVal 40 --mulTS 1 --fac 1 --num_users 10

python3 main.py --algorithm FedPG --learning_rate 0.005 --num_global_iters 5 --dim 100 --local_epochs 10 --ro 1 --dataset Traffic860 --window 100 --ro_auto 1 --missingVal 60 --mulTS 1 --fac 1 --num_users 10

python3 main.py --algorithm FedPG --learning_rate 0.005 --num_global_iters 5 --dim 100 --local_epochs 10 --ro 1 --dataset Traffic860 --window 100 --ro_auto 1 --missingVal 80 --mulTS 1 --fac 1 --num_users 10


python3 main.py --algorithm FedPG --learning_rate 0.005 --num_global_iters 5 --dim 100 --local_epochs 10 --ro 1 --dataset Traffic860 --window 100 --ro_auto 1 --missingVal 0 --mulTS 1 --fac 1 --num_users 43

python3 main.py --algorithm FedPG --learning_rate 0.005 --num_global_iters 5 --dim 100 --local_epochs 10 --ro 1 --dataset Traffic860 --window 100 --ro_auto 1 --missingVal 20 --mulTS 1 --fac 1 --num_users 43

python3 main.py --algorithm FedPG --learning_rate 0.005 --num_global_iters 5 --dim 100 --local_epochs 10 --ro 1 --dataset Traffic860 --window 100 --ro_auto 1 --missingVal 40 --mulTS 1 --fac 1 --num_users 43

python3 main.py --algorithm FedPG --learning_rate 0.005 --num_global_iters 5 --dim 100 --local_epochs 10 --ro 1 --dataset Traffic860 --window 100 --ro_auto 1 --missingVal 60 --mulTS 1 --fac 1 --num_users 43

python3 main.py --algorithm FedPG --learning_rate 0.005 --num_global_iters 5 --dim 100 --local_epochs 10 --ro 1 --dataset Traffic860 --window 100 --ro_auto 1 --missingVal 80 --mulTS 1 --fac 1 --num_users 43


python3 main.py --algorithm FedPG --learning_rate 0.005 --num_global_iters 5 --dim 100 --local_epochs 10 --ro 1 --dataset Traffic860 --window 100 --ro_auto 1 --missingVal 0 --mulTS 1 --fac 1 --num_users 86

python3 main.py --algorithm FedPG --learning_rate 0.005 --num_global_iters 5 --dim 100 --local_epochs 10 --ro 1 --dataset Traffic860 --window 100 --ro_auto 1 --missingVal 20 --mulTS 1 --fac 1 --num_users 86

python3 main.py --algorithm FedPG --learning_rate 0.005 --num_global_iters 5 --dim 100 --local_epochs 10 --ro 1 --dataset Traffic860 --window 100 --ro_auto 1 --missingVal 40 --mulTS 1 --fac 1 --num_users 86

python3 main.py --algorithm FedPG --learning_rate 0.005 --num_global_iters 5 --dim 100 --local_epochs 10 --ro 1 --dataset Traffic860 --window 100 --ro_auto 1 --missingVal 60 --mulTS 1 --fac 1 --num_users 86

python3 main.py --algorithm FedPG --learning_rate 0.005 --num_global_iters 5 --dim 100 --local_epochs 10 --ro 1 --dataset Traffic860 --window 100 --ro_auto 1 --missingVal 80 --mulTS 1 --fac 1 --num_users 86


python3 main.py --algorithm FedPG --learning_rate 0.005 --num_global_iters 5 --dim 100 --local_epochs 10 --ro 1 --dataset Traffic860 --window 100 --ro_auto 1 --missingVal 0 --mulTS 1 --fac 1 --num_users 172

python3 main.py --algorithm FedPG --learning_rate 0.005 --num_global_iters 5 --dim 100 --local_epochs 10 --ro 1 --dataset Traffic860 --window 100 --ro_auto 1 --missingVal 20 --mulTS 1 --fac 1 --num_users 172

python3 main.py --algorithm FedPG --learning_rate 0.005 --num_global_iters 5 --dim 100 --local_epochs 10 --ro 1 --dataset Traffic860 --window 100 --ro_auto 1 --missingVal 40 --mulTS 1 --fac 1 --num_users 172

python3 main.py --algorithm FedPG --learning_rate 0.005 --num_global_iters 5 --dim 100 --local_epochs 10 --ro 1 --dataset Traffic860 --window 100 --ro_auto 1 --missingVal 60 --mulTS 1 --fac 1 --num_users 172

python3 main.py --algorithm FedPG --learning_rate 0.005 --num_global_iters 5 --dim 100 --local_epochs 10 --ro 1 --dataset Traffic860 --window 100 --ro_auto 1 --missingVal 80 --mulTS 1 --fac 1 --num_users 172


python3 main.py --algorithm FedPG --learning_rate 0.005 --num_global_iters 5 --dim 100 --local_epochs 10 --ro 1 --dataset Traffic860 --window 100 --ro_auto 1 --missingVal 0 --mulTS 1 --fac 1 --num_users 430

python3 main.py --algorithm FedPG --learning_rate 0.005 --num_global_iters 5 --dim 100 --local_epochs 10 --ro 1 --dataset Traffic860 --window 100 --ro_auto 1 --missingVal 20 --mulTS 1 --fac 1 --num_users 430

python3 main.py --algorithm FedPG --learning_rate 0.005 --num_global_iters 5 --dim 100 --local_epochs 10 --ro 1 --dataset Traffic860 --window 100 --ro_auto 1 --missingVal 40 --mulTS 1 --fac 1 --num_users 430

python3 main.py --algorithm FedPG --learning_rate 0.005 --num_global_iters 5 --dim 100 --local_epochs 10 --ro 1 --dataset Traffic860 --window 100 --ro_auto 1 --missingVal 60 --mulTS 1 --fac 1 --num_users 430

python3 main.py --algorithm FedPG --learning_rate 0.005 --num_global_iters 5 --dim 100 --local_epochs 10 --ro 1 --dataset Traffic860 --window 100 --ro_auto 1 --missingVal 80 --mulTS 1 --fac 1 --num_users 430

python3 main.py --algorithm FedPG --learning_rate 0.005 --num_global_iters 5 --dim 100 --local_epochs 10 --ro 1 --dataset Traffic860 --window 100 --ro_auto 1 --missingVal 0 --mulTS 1 --fac 1 --num_users 860

<code></pre>

## Running command for FedLSTM with missing data in multivariate-time series on client with Elec370 dataset
<pre></code>

python3 main.py --dataset Imputed_Elec370 --algorithm FedLSTM --batch_size 64 --learning_rate 0.001 --fac 0.3 --num_global_iters 20 --local_epochs 1 --num_users 10 --mulTS 1 --missingVal 0 --datatype hankel

python3 main.py --dataset Imputed_Elec370 --algorithm FedLSTM --batch_size 64 --learning_rate 0.001 --fac 0.2 --num_global_iters 20 --local_epochs 1 --num_users 10 --mulTS 1 --missingVal 0 --datatype hankel

python3 main.py --dataset Imputed_Elec370 --algorithm FedLSTM --batch_size 64 --learning_rate 0.001 --fac 0.2 --num_global_iters 20 --local_epochs 1 --num_users 10 --mulTS 1 --missingVal 20 --datatype hankel

python3 main.py --dataset Imputed_Elec370 --algorithm FedLSTM --batch_size 64 --learning_rate 0.001 --fac 0.2 --num_global_iters 20 --local_epochs 1 --num_users 10 --mulTS 1 --missingVal 40 --datatype hankel

python3 main.py --dataset Imputed_Elec370 --algorithm FedLSTM --batch_size 64 --learning_rate 0.001 --fac 0.2 --num_global_iters 20 --local_epochs 1 --num_users 10 --mulTS 1 --missingVal 60 --datatype hankel

python3 main.py --dataset Imputed_Elec370 --algorithm FedLSTM --batch_size 64 --learning_rate 0.001 --fac 0.2 --num_global_iters 20 --local_epochs 1 --num_users 10 --mulTS 1 --missingVal 80 --datatype hankel

python3 main.py --dataset Imputed_Elec370 --algorithm FedLSTM --batch_size 64 --learning_rate 0.001 --fac 0.3 --num_global_iters 20 --local_epochs 1 --num_users 18 --mulTS 1 --missingVal 0 --datatype hankel

python3 main.py --dataset Imputed_Elec370 --algorithm FedLSTM --batch_size 64 --learning_rate 0.001 --fac 0.3 --num_global_iters 20 --local_epochs 1 --num_users 37 --mulTS 1 --missingVal 0 --datatype hankel

python3 main.py --dataset Imputed_Elec370 --algorithm FedLSTM --batch_size 64 --learning_rate 0.001 --fac 0.3 --num_global_iters 20 --local_epochs 1 --num_users 37 --mulTS 1 --missingVal 20 --datatype hankel

python3 main.py --dataset Imputed_Elec370 --algorithm FedLSTM --batch_size 64 --learning_rate 0.001 --fac 0.3 --num_global_iters 20 --local_epochs 1 --num_users 37 --mulTS 1 --missingVal 40 --datatype hankel

python3 main.py --dataset Imputed_Elec370 --algorithm FedLSTM --batch_size 64 --learning_rate 0.001 --fac 0.3 --num_global_iters 20 --local_epochs 1 --num_users 37 --mulTS 1 --missingVal 60 --datatype hankel

python3 main.py --dataset Imputed_Elec370 --algorithm FedLSTM --batch_size 64 --learning_rate 0.001 --fac 0.3 --num_global_iters 20 --local_epochs 1 --num_users 37 --mulTS 1 --missingVal 80 --datatype hankel

python3 main.py --dataset Imputed_Elec370 --algorithm FedLSTM --batch_size 64 --learning_rate 0.001 --fac 0.3 --num_global_iters 20 --local_epochs 1 --num_users 74 --mulTS 1 --missingVal 0 --datatype hankel

python3 main.py --dataset Imputed_Elec370 --algorithm FedLSTM --batch_size 64 --learning_rate 0.001 --fac 0.3 --num_global_iters 20 --local_epochs 1 --num_users 74 --mulTS 1 --missingVal 20 --datatype hankel

python3 main.py --dataset Imputed_Elec370 --algorithm FedLSTM --batch_size 64 --learning_rate 0.001 --fac 0.3 --num_global_iters 20 --local_epochs 1 --num_users 74 --mulTS 1 --missingVal 40 --datatype hankel

python3 main.py --dataset Imputed_Elec370 --algorithm FedLSTM --batch_size 64 --learning_rate 0.001 --fac 0.3 --num_global_iters 20 --local_epochs 1 --num_users 74 --mulTS 1 --missingVal 60 --datatype hankel

python3 main.py --dataset Imputed_Elec370 --algorithm FedLSTM --batch_size 64 --learning_rate 0.001 --fac 0.3 --num_global_iters 20 --local_epochs 1 --num_users 74 --mulTS 1 --missingVal 80 --datatype hankel

python3 main.py --dataset Imputed_Elec370 --algorithm FedLSTM --batch_size 64 --learning_rate 0.001 --fac 0.3 --num_global_iters 20 --local_epochs 1 --num_users 148 --mulTS 1 --missingVal 0 --datatype hankel

python3 main.py --dataset Imputed_Elec370 --algorithm FedLSTM --batch_size 64 --learning_rate 0.001 --fac 0.3 --num_global_iters 20 --local_epochs 1 --num_users 148 --mulTS 1 --missingVal 20 --datatype hankel

python3 main.py --dataset Imputed_Elec370 --algorithm FedLSTM --batch_size 64 --learning_rate 0.001 --fac 0.3 --num_global_iters 20 --local_epochs 1 --num_users 148 --mulTS 1 --missingVal 40 --datatype hankel

python3 main.py --dataset Imputed_Elec370 --algorithm FedLSTM --batch_size 64 --learning_rate 0.001 --fac 0.3 --num_global_iters 20 --local_epochs 1 --num_users 148 --mulTS 1 --missingVal 60 --datatype hankel

python3 main.py --dataset Imputed_Elec370 --algorithm FedLSTM --batch_size 64 --learning_rate 0.001 --fac 0.3 --num_global_iters 20 --local_epochs 1 --num_users 148 --mulTS 1 --missingVal 80 --datatype hankel

python3 main.py --dataset Imputed_Elec370 --algorithm FedLSTM --batch_size 64 --learning_rate 0.001 --fac 0.3 --num_global_iters 20 --local_epochs 1 --num_users 370 --mulTS 1 --missingVal 0 --datatype hankel

python3 main.py --dataset Imputed_Elec370 --algorithm FedLSTM --batch_size 64 --learning_rate 0.001 --fac 0.3 --num_global_iters 20 --local_epochs 1 --num_users 370 --mulTS 1 --missingVal 20 --datatype hankel

python3 main.py --dataset Imputed_Elec370 --algorithm FedLSTM --batch_size 64 --learning_rate 0.001 --fac 0.3 --num_global_iters 20 --local_epochs 1 --num_users 370 --mulTS 1 --missingVal 40 --datatype hankel

python3 main.py --dataset Imputed_Elec370 --algorithm FedLSTM --batch_size 64 --learning_rate 0.001 --fac 0.3 --num_global_iters 20 --local_epochs 1 --num_users 370 --mulTS 1 --missingVal 60 --datatype hankel

python3 main.py --dataset Imputed_Elec370 --algorithm FedLSTM --batch_size 64 --learning_rate 0.001 --fac 0.3 --num_global_iters 20 --local_epochs 1 --num_users 370 --mulTS 1 --missingVal 80 --datatype hankel

<code></pre>


## Running command for FedLSTM with missing data in multivariate-time series on client with Traffic860 dataset
<pre></code>

python3 main.py --dataset Imputed_Traff860 --algorithm FedLSTM --batch_size 64 --learning_rate 0.001 --fac 0.3 --num_global_iters 20 --local_epochs 1 --num_users 10 --mulTS 1 --missingVal 0 --datatype hankel

python3 main.py --dataset Imputed_Traff860 --algorithm FedLSTM --batch_size 64 --learning_rate 0.001 --fac 0.3 --num_global_iters 20 --local_epochs 1 --num_users 10 --mulTS 1 --missingVal 20 --datatype hankel

python3 main.py --dataset Imputed_Traff860 --algorithm FedLSTM --batch_size 64 --learning_rate 0.001 --fac 0.3 --num_global_iters 20 --local_epochs 1 --num_users 10 --mulTS 1 --missingVal 40 --datatype hankel

python3 main.py --dataset Imputed_Traff860 --algorithm FedLSTM --batch_size 64 --learning_rate 0.001 --fac 0.3 --num_global_iters 20 --local_epochs 1 --num_users 10 --mulTS 1 --missingVal 60 --datatype hankel

python3 main.py --dataset Imputed_Traff860 --algorithm FedLSTM --batch_size 64 --learning_rate 0.001 --fac 0.3 --num_global_iters 20 --local_epochs 1 --num_users 10 --mulTS 1 --missingVal 80 --datatype hankel


python3 main.py --dataset Imputed_Traff860 --algorithm FedLSTM --batch_size 64 --learning_rate 0.001 --fac 0.3 --num_global_iters 20 --local_epochs 1 --num_users 43 --mulTS 1 --missingVal 0 --datatype hankel

python3 main.py --dataset Imputed_Traff860 --algorithm FedLSTM --batch_size 64 --learning_rate 0.001 --fac 0.3 --num_global_iters 20 --local_epochs 1 --num_users 43 --mulTS 1 --missingVal 20 --datatype hankel

python3 main.py --dataset Imputed_Traff860 --algorithm FedLSTM --batch_size 64 --learning_rate 0.001 --fac 0.3 --num_global_iters 20 --local_epochs 1 --num_users 43 --mulTS 1 --missingVal 40 --datatype hankel

python3 main.py --dataset Imputed_Traff860 --algorithm FedLSTM --batch_size 64 --learning_rate 0.001 --fac 0.3 --num_global_iters 20 --local_epochs 1 --num_users 43 --mulTS 1 --missingVal 60 --datatype hankel

python3 main.py --dataset Imputed_Traff860 --algorithm FedLSTM --batch_size 64 --learning_rate 0.001 --fac 0.3 --num_global_iters 20 --local_epochs 1 --num_users 43 --mulTS 1 --missingVal 80 --datatype hankel


python3 main.py --dataset Imputed_Traff860 --algorithm FedLSTM --batch_size 64 --learning_rate 0.001 --fac 0.3 --num_global_iters 20 --local_epochs 1 --num_users 86 --mulTS 1 --missingVal 0 --datatype hankel

python3 main.py --dataset Imputed_Traff860 --algorithm FedLSTM --batch_size 64 --learning_rate 0.001 --fac 0.3 --num_global_iters 20 --local_epochs 1 --num_users 86 --mulTS 1 --missingVal 20 --datatype hankel

python3 main.py --dataset Imputed_Traff860 --algorithm FedLSTM --batch_size 64 --learning_rate 0.001 --fac 0.3 --num_global_iters 20 --local_epochs 1 --num_users 86 --mulTS 1 --missingVal 40 --datatype hankel

python3 main.py --dataset Imputed_Traff860 --algorithm FedLSTM --batch_size 64 --learning_rate 0.001 --fac 0.3 --num_global_iters 20 --local_epochs 1 --num_users 86 --mulTS 1 --missingVal 60 --datatype hankel

python3 main.py --dataset Imputed_Traff860 --algorithm FedLSTM --batch_size 64 --learning_rate 0.001 --fac 0.3 --num_global_iters 20 --local_epochs 1 --num_users 86 --mulTS 1 --missingVal 80 --datatype hankel


python3 main.py --dataset Imputed_Traff860 --algorithm FedLSTM --batch_size 64 --learning_rate 0.001 --fac 0.3 --num_global_iters 20 --local_epochs 1 --num_users 172 --mulTS 1 --missingVal 0 --datatype hankel

python3 main.py --dataset Imputed_Traff860 --algorithm FedLSTM --batch_size 64 --learning_rate 0.001 --fac 0.3 --num_global_iters 20 --local_epochs 1 --num_users 172 --mulTS 1 --missingVal 20 --datatype hankel

python3 main.py --dataset Imputed_Traff860 --algorithm FedLSTM --batch_size 64 --learning_rate 0.001 --fac 0.3 --num_global_iters 20 --local_epochs 1 --num_users 172 --mulTS 1 --missingVal 40 --datatype hankel

python3 main.py --dataset Imputed_Traff860 --algorithm FedLSTM --batch_size 64 --learning_rate 0.001 --fac 0.3 --num_global_iters 20 --local_epochs 1 --num_users 172 --mulTS 1 --missingVal 60 --datatype hankel

python3 main.py --dataset Imputed_Traff860 --algorithm FedLSTM --batch_size 64 --learning_rate 0.001 --fac 0.3 --num_global_iters 20 --local_epochs 1 --num_users 172 --mulTS 1 --missingVal 80 --datatype hankel


python3 main.py --dataset Imputed_Traff860 --algorithm FedLSTM --batch_size 64 --learning_rate 0.001 --fac 0.3 --num_global_iters 20 --local_epochs 1 --num_users 430 --mulTS 1 --missingVal 0 --datatype hankel

python3 main.py --dataset Imputed_Traff860 --algorithm FedLSTM --batch_size 64 --learning_rate 0.001 --fac 0.3 --num_global_iters 20 --local_epochs 1 --num_users 430 --mulTS 1 --missingVal 20 --datatype hankel

python3 main.py --dataset Imputed_Traff860 --algorithm FedLSTM --batch_size 64 --learning_rate 0.001 --fac 0.3 --num_global_iters 20 --local_epochs 1 --num_users 430 --mulTS 1 --missingVal 40 --datatype hankel

python3 main.py --dataset Imputed_Traff860 --algorithm FedLSTM --batch_size 64 --learning_rate 0.001 --fac 0.3 --num_global_iters 20 --local_epochs 1 --num_users 430 --mulTS 1 --missingVal 60 --datatype hankel

python3 main.py --dataset Imputed_Traff860 --algorithm FedLSTM --batch_size 64 --learning_rate 0.001 --fac 0.3 --num_global_iters 20 --local_epochs 1 --num_users 430 --mulTS 1 --missingVal 80 --datatype hankel

python3 main.py --dataset Imputed_Traff860 --algorithm FedLSTM --batch_size 64 --learning_rate 0.001 --fac 0.3 --num_global_iters 20 --local_epochs 1 --num_users 860 --mulTS 1 --missingVal 0 --datatype hankel



<code></pre>

## Running command for FedLR with no-missing data in multivariate-time series on client
<pre></code>
python3 main.py --dataset Imputed_Elec370 --algorithm FedLR --batch_size 64 --learning_rate 0.001 --fac 0.1 --num_global_iters 20 --local_epochs 1 --num_users 37 --mulTS 1 --missingVal 20 --datatype hankel
<code></pre>
## Dataset:
Electricity dataset
Traffic dataset
