import torch
import os
import torch.multiprocessing as mp

from FLAlgorithms.users.userLSTM import UserLSTM
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data
import numpy as np

# Implementation for FedAvg Server

class serverLSTM(Server):
    def __init__(self, experiment, device, dataset,algorithm, model, batch_size, learning_rate, beta, L_k, num_glob_iters, local_epochs, optimizer, subusers, times , cutoff):
        super().__init__(experiment, device, dataset,algorithm, model, batch_size, learning_rate, beta, L_k, num_glob_iters,local_epochs, optimizer, subusers, times)

        # Initialize data for all  users
        self.K = 0
        self.subusers = subusers
        total_users = 20 # pre-define for 20 users
        self.cutoff = cutoff

        if dataset == "sine":
            x_train, x_test, y_train, y_test = self.create_sine_dataset()
            print("--------------------------------------------------")
        elif dataset == "Imputed_Elec370":
            train_data, _ = self.get_imputed_data()

        for i in range(total_users):
            if dataset == "sine":
                id, train , test = self.create_sine_user_data(i, x_train, x_test, y_train, y_test)
            if dataset == "Imputed_Elec370":
                id, train = self.create_elec_user_data(i, train_data=train_data)
                test = train

            # print(f"batch_size: {batch_size}")
            user = UserLSTM(device, id, train, test, model, batch_size, learning_rate,beta,L_k, local_epochs, optimizer)
            print(user.id)
            self.users.append(user)
            
        print("Number of users / total users:",subusers, " / " ,total_users)
        print("Finished creating FedAvg LSTM server.")

    def load_train_data(self):
        results_path = f"results/imputed_data/"
        file_name = f"numuser_370_L_80_dim_78_missingPercentage_20.npy"
        file_path = os.path.join(results_path, file_name)
        return_file = np.load(file_path)
        return return_file
    
    def load_test_data(self):
        results_path = f"results/test_data/"
        file_name = f"test_data_336steps.npy"
        file_path = os.path.join(results_path, file_name)
        return_file = np.load(file_path)
        return return_file 

    def get_imputed_data(self):
        train_data = self.load_train_data()
        test_data = self.load_test_data()
        return train_data, test_data

    def create_elec_user_data(self, id, train_data, window=40):
        # Create train data
        X =  train_data[id]
        length_ts_data = X.shape[0]
        x_train = []
        y_train = []

        for i in range(length_ts_data - window):
            x_train.append(X[i:i+window])
            y_train.append(X[i+window])

        train_x = np.array(x_train)
        train_y = np.array(y_train)
        X_train =  torch.Tensor(train_x)
        y_train = torch.Tensor(train_y)
        # Creating train and test tuples
        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        return id, train_data

    def create_sine_dataset(self):
        #creating the dataset
        x = np.arange(1,721,1)
        y = np.sin(x*np.pi/180)  + np.random.randn(720)*0.05

        # structuring the data 
        X = [] 
        Y = [] 
        for i in range(0,710):
            list1 = []
            for j in range(i,i+10):
                list1.append(y[j])
            X.append(list1)
            Y.append(y[j+1])
        #train test split
        X = np.array(X)
        Y = np.array(Y)
        x_train = X[:360]
        x_test = X[360:]
        y_train = Y[:360]
        y_test = Y[360:]
        return x_train, x_test, y_train, y_test

    def create_sine_user_data(self, id, x_train, x_test, y_train, y_test):
        # Partitioning data for users
        start_idx = id*18
        end_idx = (id + 1)*18
        X_train = torch.Tensor(x_train[start_idx:end_idx])
        y_train = torch.Tensor(y_train[start_idx:end_idx])
        X_test = torch.Tensor(x_test[start_idx:end_idx])
        y_test = torch.Tensor(y_test[start_idx:end_idx])
        # Creating train and test tuples
        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        test_data = [(x, y) for x, y in zip(X_test, y_test)]

        return id, train_data, test_data
        
    def send_grads(self):
        assert (self.users is not None and len(self.users) > 0)
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad)
        for user in self.users:
            user.set_grads(grads)

    def train(self):
        for glob_iter in range(self.num_glob_iters):
            if(self.experiment):
                self.experiment.set_epoch( glob_iter + 1)
            print("-------------Round number: ",glob_iter, " -------------")
            #loss_ = 0
            self.send_parameters()

            # Evaluate model each interation
            # self.evaluate()

            self.selected_users = self.select_users(glob_iter,self.subusers)
            
            #NOTE: this is required for the ``fork`` method to work
            for user in self.selected_users:
                user.train(self.local_epochs)
                print(f"selected user id: {user.id}")

            self.aggregate_parameters()
            
        # self.save_results()
        self.save_model_lstm()