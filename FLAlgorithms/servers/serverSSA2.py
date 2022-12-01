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

import h5py

# Implementation for FedAvg Server

class ADMM_SSA(Server2):
    def __init__(self, algorithm, experiment, device, dataset, learning_rate, ro, num_glob_iters, local_epochs, num_users, dim, time, window, ro_auto, imputationORforecast):
        super().__init__(device, dataset, learning_rate, ro, num_glob_iters, local_epochs, num_users, dim, window, ro_auto, time)

        # Initialize data for all  users
        self.algorithm = algorithm
        self.K = 0
        self.dim = dim
        self.experiment = experiment
        np.random.seed(1993)
        if dataset == 'debug': self.debug = True
        else: self.debug = False
        if self.debug:
            total_users = 2
        else:
            if dataset[:4] == 'Elec':
                total_users = int(dataset[4:])
        print("total users: ", total_users)
        self.num_users = total_users
        self.imputationORforecast = imputationORforecast

        # self.store_ids = ['2', '3', '4']
        self.house_ids = ['MT_{0:03}'.format(i+1) for i in range(total_users)]
        
        self.all_train_data = []

        for i in range(total_users):            
            # train = self.generate_synthetic_data()
            # store_id = self.store_ids[i]
            # train = self.get_store_sale_data(store_id)
            
            if self.debug:
                id = i
                train = self.generate_synthetic_data_gaussian(id)
            else:
                id = self.house_ids[i]
                train = self.get_electricity_data(id)

            self.all_train_data.append(train)

            train = torch.tensor(train, dtype=torch.float64)
            
            # Jiayu: init U = svd(X_i)
            U, S, V = torch.svd(train)
            U = U[:, :dim]
            # self.commonPCAz = V
            # print("type of V", type(U))
            # print("shape of U: ", U.shape)
            # print("Init U (svd): \n", U)
            self.commonPCAz = torch.rand_like(U, dtype=torch.float64)
            # self.commonPCAz = U
            # print("Init U (randomized): \n", self.commonPCAz)

            # check = torch.matmul(U.T,U)

            user = UserADMM2(algorithm, device, id, train, self.commonPCAz, learning_rate, ro, local_epochs, dim, ro_auto)
            self.users.append(user)
            self.total_train_samples += user.train_samples
        
        # Jiayu: add constraint - ZTZ = I
        self.localG = torch.matmul(self.commonPCAz.T, self.commonPCAz)

        self.all_train_data = np.hstack(self.all_train_data)
        self.all_train_data = torch.tensor(self.all_train_data, dtype=torch.float64)
            
        print("Number of users / total users:",num_users, " / " ,total_users)
        print("Finished creating FedAvg server.")

    def generate_synthetic_data(self):
        data = pd.read_csv("data/MixtureTS_var.csv", index_col = 'time')
        df = pd.DataFrame(index=data.index)
        df['ts'] = data['ts']
        return df

    def generate_synthetic_data_gaussian(self, id):
        np.random.seed(id)
        a = np.random.normal(size=(1000,3))
        X = torch.tensor(a.T, dtype=torch.float64)
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
        L = self.window # The window length
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
        # print(file_path)
        colname = mt_id
        elec = house[colname].copy()
        F = elec.to_numpy()
        N = F.shape[0] 
        L = self.window # The window length
        M = int(N*self.num_users/L)
        if M%self.num_users != 0:
            M -= M%self.num_users
        M /= self.num_users
        # K = N - L + 1  # number of columns in the trajectory matrix
        # X = np.column_stack([F[i:i+L] for i in range(0,K)])
        # Obtain Page matrix instead
        X = F[:int(L*M)].reshape([int(L),int(M)], order = 'F')
        # print(X.shape)
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
        # self.selected_users = self.select_users(1000,1)
        self.selected_users = self.users

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
            _ = self.evaluate_all_data()

            # self.selected_users = self.select_users(glob_iter,self.num_users)
            
            # self.users = self.selected_users 
            #NOTE: this is required for the ``fork`` method to work
            for user in self.selected_users:
                user.train(self.local_epochs)
            # self.users[0].train(self.local_epochs)

            # self.aggregate_pca()
            # Jiayu: add constraint - ZTZ = I
            self.update_global_pca() 

        loss_all_data = self.evaluate_all_data()
        self.Z = self.commonPCAz.detach().numpy().copy()
        directory = os.getcwd()
        results_folder_path = os.path.join(directory, "results/SSA")
        suffix = 'forecast' if self.imputationORforecast else 'imputation'
        result_filename = f"Grassmann_ADMM_{self.dataset}_N{self.num_users}_L{self.window}_d{self.dim}_rho{self.str_ro}_{suffix}"
        result_path = os.path.join(results_folder_path, result_filename)
        np.save(result_path, self.Z)
        # Jiayu: save Ui for each clients
        with h5py.File(result_path+'.h5', 'w') as hf:
            for i,user in enumerate(self.selected_users):
                hf.create_dataset(str(user.id), data=user.localPCA.detach().numpy().copy())
            hf.close()

        print("Completed training!!!")
        # self.save_results()
        # self.save_model()