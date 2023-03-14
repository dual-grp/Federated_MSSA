#!/usr/bin/env python
# from comet_ml import Experiment
import h5py
import matplotlib.pyplot as plt
import numpy as np
import argparse
import importlib
import random
import os
from FLAlgorithms.servers.serverADMM import ADMM
from FLAlgorithms.servers.serverLSTM import serverLSTM 
from FLAlgorithms.servers.serverLR import serverLR
from FLAlgorithms.servers.serverSSA2 import ADMM_SSA # Jiayu: add constraint - ZTZ = I
from FLAlgorithms.servers.serverAbnormalDetection import AbnormalDetection
from utils.model_utils import read_data
from FLAlgorithms.trainmodel.models import *
from utils.plot_utils import *
import torch
torch.manual_seed(0)
from utils.options import args_parser
from utils.train_utils import get_lstm, get_lr

# import comet_ml at the top of your file
#                                                                                                                           
# Create an experiment with your api key:
def main(experiment, dataset, algorithm, batch_size, learning_rate, ro, num_glob_iters,
         local_epochs, numusers, fac_users, dim, times, gpu, window, ro_auto, missingVal, mulTS, datatype):
    
    # Get device status: Check GPU or CPU
    device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() and gpu != -1 else "cpu")
    # device = torch.device("cpu")
    # data = read_data(dataset) , dataset
    data = dataset
    if algorithm == "FedLSTM":
        model = get_lstm()
        beta = 0
        L_k = 0
        optimizer = "SGD"
        cutoff = 0
        server = serverLSTM(experiment, device, dataset,algorithm, model, batch_size, learning_rate, beta, L_k, num_glob_iters, local_epochs, optimizer, fac_users, times , cutoff, mulTS, missingVal, numusers, datatype)
    elif algorithm == "FedLR":
        model = get_lr()
        beta = 0
        L_k = 0
        optimizer = "SGD"
        cutoff = 0
        server = serverLR(experiment, device, dataset,algorithm, model, batch_size, learning_rate, beta, L_k, num_glob_iters, local_epochs, optimizer, fac_users, times , cutoff, mulTS, missingVal, numusers, datatype)
    else:
        server = ADMM_SSA(algorithm, experiment, device, data, learning_rate, ro, num_glob_iters, local_epochs, numusers, fac_users, dim, times, window, ro_auto, missingVal, mulTS, imputationORforecast=0)
    print("Initilized server")
    server.train()
    print("Train Done")
    # if server.check_train_exists():
    #     server.train()
    # else:
    #     server.train()
    
    # server_forecast = ADMM_SSA(algorithm, experiment, device, data, learning_rate, ro, num_glob_iters, local_epochs, numusers, dim, times, imputationORforecast=1)
    # server_forecast.train()

if __name__ == "__main__":
    args = args_parser()
    print("=" * 80)
    print("Summary of training process:")
    print("Algorithm: {}".format(args.algorithm))
    print("Batch size: {}".format(args.batch_size))
    print("Learing rate       : {}".format(args.learning_rate))
    print("Average Moving       : {}".format(args.ro))
    print("Auto Average Moving  : {}".format(args.ro_auto))
    print("Number of users      : {}".format(args.num_users))
    print("Number of global rounds       : {}".format(args.num_global_iters))
    print("Number of local rounds       : {}".format(args.local_epochs))
    print("Dataset       : KDD")
    # print("Dataset       : {}".format(args.dataset))
    print("=" * 80)

    if(args.commet):
        # Create an experiment with your api key:
        experiment = Experiment(
            api_key="VtHmmkcG2ngy1isOwjkm5sHhP",
            project_name="multitask-for-test",
            workspace="federated-learning-exp",
        )

        hyper_params = {
            "dataset":args.dataset,
            "algorithm" : args.algorithm,
            "batch_size":args.batch_size,
            "learning_rate":args.learning_rate,
            "ro":args.ro,
            "ro_hong":args.ro_auto,
            "dim" : args.dim,
            "window": args.window,
            "num_glob_iters":args.num_global_iters,
            "local_epochs":args.local_epochs,
            "numusers": args.subusers,
            "times" : args.times,
            "gpu": args.gpu,
            "cut-off": args.cutoff
        }
        
        experiment.log_parameters(hyper_params)
    else:
        experiment = 0

    main(
        experiment= experiment,
        dataset=args.dataset,
        algorithm = args.algorithm,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        ro = args.ro,   
        ro_auto = args.ro_auto,
        num_glob_iters=args.num_global_iters,
        local_epochs=args.local_epochs,
        numusers = args.num_users,
        fac_users = args.fac,
        dim = args.dim,
        window = args.window,
        times = args.times,
        gpu=args.gpu,
        missingVal=args.missingVal,
        mulTS = args.mulTS,
        datatype= args.datatype
        )


