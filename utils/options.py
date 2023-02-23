#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Traffic20") # choices=["debug","Traffic20","Elec5","Elec20", "sine"]
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default = 0.005, help="Local learning rate")
    parser.add_argument("--ro", type=float, default=1.0, help="Regularization term")
    parser.add_argument("--ro_auto", type=int, default=1, help="Regularization term chosen based on Hong2014/Hajinezhad2015")
    parser.add_argument("--num_global_iters", type=int, default=10)
    parser.add_argument("--local_epochs", type=int, default = 1)
    parser.add_argument("--dim", type=int, default = 3)
    parser.add_argument("--window", type=int, default = 20)
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--algorithm", type=str, default="FedPG",choices=["FedPG","FedPE", "FedLSTM"])
    parser.add_argument("--subusers", type = float, default = 1, help="Number of Users per round")
    parser.add_argument("--times", type=int, default=1, help="running time")
    parser.add_argument("--commet", type=int, default=0, help="log data to commet")
    parser.add_argument("--gpu", type=int, default=0, help="Which GPU to run the experiments")
    parser.add_argument("--missingVal", type=int, default=1, help="Train with missing values if True")
    args = parser.parse_args()

    return args
