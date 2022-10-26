import torch
import os
import pandas as pd

from FLAlgorithms.users.userADMM import UserADMM
from FLAlgorithms.users.userADMM2 import UserADMM2
from FLAlgorithms.servers.serverbase2 import Server2
from utils.model_utils import read_data, read_user_data
from sklearn.preprocessing import StandardScaler

import numpy as np
from numpy import pi

# Implementation for FedAvg Server

class ADMM_SSA(Server2):
    def __init__(self, algorithm, experiment, device, dataset, learning_rate, ro, num_glob_iters, local_epochs, num_users, dim, time, imputationORforecast):
        super().__init__(device, dataset, learning_rate, ro, num_glob_iters, local_epochs, num_users, dim, time)

        # Initialize data for all  users
        self.algorithm = algorithm
        self.K = 0
        self.dim = dim
        self.experiment = experiment
        total_users = len(dataset[0][0])
        np.random.seed(1993)
        total_users = 2
        print("total users: ", total_users)
        self.num_users = total_users
        self.imputationORforecast = imputationORforecast

        # self.store_ids = ['2', '3', '4']
        self.house_ids = ['MT_{0:03}'.format(i+1) for i in range(20)]

        for i in range(total_users):            
            # train = self.generate_synthetic_data()
            # store_id = self.store_ids[i]
            # train = self.get_store_sale_data(store_id)
            
            id = i
            train = self.generate_synthetic_data_gaussian(id)
            # house_id = self.house_ids[i]
            # train = self.get_electricity_data(house_id)

            train = torch.Tensor(train)
            # print(train)
            if(i == 0):
                U, S, V = torch.svd(train)
                U = U[:, :dim]
                # self.commonPCAz = V
                # print("type of V", type(U))
                print("shape of U: ", U.shape)
                # print("Init U (svd): \n", U)
                torch.manual_seed(10)
                self.commonPCAz = torch.rand_like(U, dtype=torch.float)
                # print("Init U (randomized): \n", self.commonPCAz)

                check = torch.matmul(U,U.T)

            user = UserADMM2(algorithm, device, id, train, self.commonPCAz, learning_rate, ro, local_epochs, dim)
            self.users.append(user)
            self.total_train_samples += user.train_samples
            
        print("Number of users / total users:",num_users, " / " ,total_users)
        print("Finished creating FedAvg server.")

    def generate_synthetic_data(self):
        N = 200 # The number of time 'moments' in our toy series
        t = np.arange(0,N)
        trend = 0.001 * (t - 100)**2
        p1 = 20
        periodic1 = 2 * np.sin(2*pi*t/p1)
        noise = 2 * (np.random.rand(N) - 0.5)
        F = trend + periodic1 + noise
        L = 20 # The window length
        K = N - L + 1  # number of columns in the trajectory matrix
        X = np.column_stack([F[i:i+L] for i in range(0,K)])
        X.astype(float)
        sX = StandardScaler(copy=True)
        C = sX.fit_transform(X)
        return C

    def generate_synthetic_data_gaussian(self, id):
        np.random.seed(id)
        a = np.random.normal(size=(1000,3))
        X = torch.Tensor(a.T)
        # Xt = X.T
        print("seed id:", id)
        print("Gaussian synthetic data first 3 obs\n", X[:3,:3])
        return X

    def get_store_sale_data(self, store_id):
        DATA_PATH = "data/"
        store_name = f"store{store_id}_salses.csv"
        file_path = DATA_PATH + store_name
        store_sale = pd.read_csv(file_path)
        print(file_path)
        sales = store_sale['sales'].copy()
        F = sales.to_numpy()
        N = F.shape[0]
        L = 20 # The window length
        K = N - L + 1  # number of columns in the trajectory matrix
        X = np.column_stack([F[i:i+L] for i in range(0,K)])
        X.astype(float)
        sX = StandardScaler(copy=True)
        C = sX.fit_transform(X)
        return C
    
    def get_electricity_data(self, mt_id):
        DATA_PATH = "electricity_train/"
        store_name = f"{mt_id}.csv"
        file_path = DATA_PATH + store_name
        house = pd.read_csv(file_path)
        print(file_path)
        colname = mt_id
        elec = house[colname].copy()
        F = elec.to_numpy()
        N = F.shape[0]
        L = 320 # The window length

        # K = N - L + 1  # number of columns in the trajectory matrix
        # X = np.column_stack([F[i:i+L] for i in range(0,K)])

        # Obtain Page matrix instead
        X = F.reshape([L,int(N/L)], order = 'F')
        X.astype(float)

        if self.imputationORforecast: 
            results_folder_path = os.path.join(os.getcwd(), "results/SSA")
            result_filename = f"Grassmann_ADMM_Electricity_{self.num_users}_L20_d{self.dim}_imputation.npy"
            result_path = os.path.join(results_folder_path, result_filename)
            Z = np.load(result_path)
            Xhat = Z.dot(Z.T.dot(X))
            Xhat = Xhat[:-1,:]
            return Xhat
            
        else:
            return X

        # We did scaling when producing csv

        # sX = StandardScaler(copy=True)
        # C = sX.fit_transform(X)
        # return C

        # Obtain X instead of C
        # return X
        
    def train(self):
        self.selected_users = self.select_users(1000,1)
        print("Selected users: ")
        for i, user in enumerate(self.selected_users):
            print("user_id selected for training: ", i)
        for glob_iter in range(self.num_glob_iters):
            if(self.experiment):
                self.experiment.set_epoch( glob_iter + 1)
            print("-------------Round number: ",glob_iter, " -------------")
            #loss_ = 0
            self.send_pca()

            # Evaluate model each interation
            self.evaluate()

            # self.selected_users = self.select_users(glob_iter,self.num_users)
            
            # self.users = self.selected_users 
            #NOTE: this is required for the ``fork`` method to work
            for user in self.selected_users:
                user.train(self.local_epochs)
            # self.users[0].train(self.local_epochs)
            self.aggregate_pca()
        self.Z = self.commonPCAz.detach().numpy().copy()
        directory = os.getcwd()
        results_folder_path = os.path.join(directory, "results/SSA")
        suffix = 'forecast' if self.imputationORforecast else 'imputation'
        result_filename = f"Grassmann_ADMM_Electricity_{self.num_users}_L20_d{self.dim}_{suffix}"
        result_path = os.path.join(results_folder_path, result_filename)
        np.save(result_path, self.Z)
        print("Trained U: \n", self.Z[:3,:3])
        print("Completed training!!!")
        # self.save_results()
        # self.save_model()