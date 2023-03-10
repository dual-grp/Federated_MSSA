import torch
import os
import torch.multiprocessing as mp

from FLAlgorithms.users.userLSTM import UserLSTM
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data
import numpy as np

# Implementation for FedAvg Server

class serverLSTM(Server):
    def __init__(self, experiment, device, dataset,algorithm, model, batch_size, learning_rate, beta, L_k, num_glob_iters, local_epochs, optimizer, fac_users, times , cutoff, mulTS, missingVal, numusers, datatype):
        super().__init__(experiment, device, dataset,algorithm, model, batch_size, learning_rate, beta, L_k, num_glob_iters,local_epochs, optimizer, fac_users, times)

        # Initialize data for all  users
        self.K = 0
        self.subusers = fac_users
        self.total_users = numusers # pre-define for 20 users
        self.cutoff = cutoff
        self.dataset = dataset
        self.mulTS = mulTS
        self.missingVal = missingVal
        self.datatype = datatype

        if dataset == "sine":
            x_train, x_test, y_train, y_test = self.create_sine_dataset()
            print("--------------------------------------------------")
        elif dataset == "Imputed_Elec370":
            if mulTS == 0:
                train_data, _ = self.get_imputed_data(num_user=20, dim=40, missingPercentage=40)
            else:
                folder_path = f"results/imputed_data/mulTS/electricity_nusers_{self.total_users}_missing_{self.missingVal}"


        elif dataset == "Imputed_Traff20":
            train_data = self.get_traff_imputed_data(num_user=20, dim=80, missingPercentage=0)

        for i in range(self.total_users):
            if dataset == "sine":
                id, train , test = self.create_sine_user_data(i, x_train, x_test, y_train, y_test)

            if dataset == "Imputed_Elec370" or dataset == "Imputed_Traff20":
                if self.mulTS == 0:
                    id, train = self.create_elec_user_data(i, train_data=train_data, window=40)
                    test = train
                else:
                    if self.datatype == "page":
                        id, train = self.get_mulTS_imputed_data(folder_path, i)
                    else:
                        id, train = self.create_mulTS_hankel(folder_path, i)
                    test = train

            # print(f"batch_size: {batch_size}")
            user = UserLSTM(device, id, train, test, model, batch_size, learning_rate,beta,L_k, local_epochs, optimizer)
            print(user.id)
            self.users.append(user)
            
        print("Number of selected users / total users:", int(self.subusers * self.total_users), " / " , self.total_users)
        print("Finished creating FedAvg LSTM server.")

    def load_train_data(self, num_user, dim, missingPercentage):
        results_path = f"results/imputed_data/"
        file_name = f"numuser_{num_user}_L_80_dim_{dim}_missingPercentage_{missingPercentage}.npy"
        file_path = os.path.join(results_path, file_name)
        return_file = np.load(file_path)
        return return_file
    
    def load_test_data(self):
        results_path = f"results/test_data/"
        file_name = f"test_data_336steps.npy"
        file_path = os.path.join(results_path, file_name)
        return_file = np.load(file_path)
        return return_file 

    def load_traff_train_data(self, num_user, dim, missingPercentage):
        results_path = f"results/imputed_data/traffic"
        file_name = f"numuser_{num_user}_L_100_dim_{dim}_missingPercentage_{missingPercentage}.npy"
        file_path = os.path.join(results_path, file_name)
        return_file = np.load(file_path)
        return return_file
    
    def get_imputed_data(self, num_user, dim, missingPercentage):
        train_data = self.load_train_data(num_user, dim, missingPercentage)
        test_data = self.load_test_data()
        return train_data, test_data

    def get_traff_imputed_data(self, num_user, dim, missingPercentage):
        train_data = self.load_traff_train_data(num_user, dim, missingPercentage)
        return train_data
    
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
        Y_train = torch.Tensor(train_y)
        # Creating train and test tuples
        train_data = [(x, y) for x, y in zip(X_train, Y_train)]
        return id, train_data

    def windowing_data(self, train_data, window=40):
        # Create train data
        X =  train_data
        length_ts_data = X.shape[0]
        x_train = []
        y_train = []

        for i in range(length_ts_data - window):
            x_train.append(X[i:i+window])
            y_train.append(X[i+window])

        train_x = np.array(x_train)
        train_y = np.array(y_train)
        return train_x, train_y
    
    def get_mulTS_imputed_data(self, folder_path, i):
        x_file_name = f"x_client_{i}.npy"
        y_file_name = f"y_client_{i}.npy"

        x_file = os.path.join(folder_path, x_file_name)
        y_file = os.path.join(folder_path, y_file_name)

        x_train = np.load(x_file)
        y_train = np.load(y_file)

        X_train = torch.Tensor(x_train)
        Y_train = torch.Tensor(y_train)
        print(X_train.shape)
        print(Y_train.shape)
        train_data = [(x, y) for x, y in zip(X_train, Y_train)]
        return i, train_data

    def create_mulTS_hankel(self, folder_path, i):
        # Get data for client with index i
        file_name = f"all_data_client_{i}.npy"
        file_path = os.path.join(folder_path, file_name)
        data_client_i = np.load(file_path)

        print(data_client_i.shape)
        # Flatten data for client
        num_data = 37
        M_ts = 324
        data_flatten = []
        for n in range(num_data):
            data_i = data_client_i[:, n*M_ts:(n+1)*M_ts]
            data_i_flatten = data_i.flatten('F')
            data_flatten.append(data_i_flatten)
        data_flatten = np.array(data_flatten)

        # Windowing flattened data to create hankel
        x_train = []
        y_train = []
        for n in range(num_data):
            flattened_data_i = data_flatten[n]
            x_train_i, y_train_i = self.windowing_data(flattened_data_i)
            x_train.extend(x_train_i)
            y_train.extend(y_train_i)
        train_x = np.array(x_train)
        print(train_x.shape)
        train_y = np.array(y_train)
        X_train =  torch.Tensor(train_x)
        Y_train = torch.Tensor(train_y)
        # Creating train and test tuples
        train_data = [(x, y) for x, y in zip(X_train, Y_train)]
        id = i
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
        model_name = f"FedLSTM_{self.dataset}_num_user_{self.total_users}_L_80_dim_70_MP_0_W79"
        if self.mulTS == 0:
            self.save_model_lstm(model_name)
        else:
            self.save_model_lstm_mulTS(model_name)
