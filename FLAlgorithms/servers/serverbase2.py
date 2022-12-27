import torch
import os
import numpy as np
import h5py
from utils.model_utils import Metrics
import copy

class Server2:
    def __init__(self, device, dataset, learning_rate, ro, num_glob_iters, local_epochs, num_users, dim, window, ro_auto, times):
        self.device = device
        self.dataset = dataset
        self.num_glob_iters = num_glob_iters
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.total_train_samples = 0
        self.users = []
        self.selected_users = []
        self.num_users = num_users
        self.ro = ro
        self.rs_train_acc, self.rs_train_loss, self.rs_glob_acc = [], [], []
        self.times = times
        self.dim = dim
        self.window = window
        self.ro_auto = ro_auto
        if self.ro_auto:
            self.str_ro = 'auto'
        else: self.str_ro = ro

    def send_pca(self):
        assert (self.users is not None and len(self.users) > 0)
        # print("check Z", (torch.matmul(self.commonPCAz.T,self.commonPCAz)- torch.eye(self.commonPCAz.shape[1])).detach().numpy()[:3,:3] )
        # print("print Z", self.commonPCAz.detach().numpy()[:3,:3])
        # for user in self.users:
        for user in self.selected_users:
            # print("user_id", user.id)
            user.set_commonPCA(self.commonPCAz)
    
    def add_pca(self, user, ratio):
        # ADMM update
        # self.commonPCAz += ratio*(user.localPCA + 1/user.ro * user.localY)
        # simplified ADMM update
        # print("simplified ADMM update")
        self.commonPCAz += ratio*(user.localPCA)

    def aggregate_pca(self):
        assert (self.users is not None and len(self.users) > 0)
        total_train = 0
        #if(self.num_users = self.to)
        for user in self.selected_users:
            total_train += user.train_samples
            # print("user_id", user.id)
        self.commonPCAz = torch.zeros(self.commonPCAz.shape, dtype=torch.float64)
        for user in self.selected_users:
            self.add_pca(user, user.train_samples / total_train)

    def update_global_pca(self):
        if self.algorithm == "FedPE": 
            '''Euclidean space'''
            # for i in range(1):
            for i in range(self.local_epochs):
                self.loss_UZ = torch.zeros(1)
                self.loss = torch.zeros(1)
                self.commonPCAz.requires_grad_(True)
                # if self.commonPCAz.grad is not None:
                #     self.commonPCAz.grad.data.zero_()

                ZTZ = torch.matmul(self.commonPCAz.T, self.commonPCAz) - torch.eye(self.commonPCAz.shape[1])
                hZ = torch.max(torch.zeros(ZTZ.shape),ZTZ)**2
                self.loss_hZ = 0.5 * self.ro * torch.norm(hZ) ** 2 + torch.sum(torch.inner(self.localG, hZ))

                for user in self.users:
                    self.loss_UZ = self.loss_UZ + torch.sum(torch.inner(user.localY, user.localPCA - self.commonPCAz)) + 0.5 * user.ro * torch.norm(user.localPCA - self.commonPCAz)** 2

                self.loss = self.loss_UZ + self.loss_hZ
                self.loss = self.loss / self.total_train_samples
                self.loss.backward(retain_graph=True)

                temp = self.commonPCAz.data.clone()
                # Solve the global problem
                if self.commonPCAz.grad is not None:
                    self.commonPCAz.grad.data.zero_()

                self.loss.backward(retain_graph=True)
                # if i == 0 or i == self.local_epochs:
                    # print("check Z loss", self.loss)
                # Update global pca
                temp = temp - self.learning_rate * self.commonPCAz.grad
                self.commonPCAz = temp.data.clone()
            self.localG = self.localG + self.ro * hZ

        else: 
            # for i in range(1):
            for i in range(self.local_epochs):
                '''Grassmannian manifold'''
                self.loss = torch.zeros(1)
                self.commonPCAz.requires_grad_(True)
                # if self.commonPCAz.grad is not None:
                #     self.commonPCAz.grad.data.zero_()

                for user in self.users:
                    self.loss = self.loss + torch.sum(torch.inner(user.localY, user.localPCA - self.commonPCAz)) + 0.5 * user.ro * torch.norm(user.localPCA - self.commonPCAz)** 2
                self.loss = self.loss / self.total_train_samples
                temp = self.commonPCAz.data.clone()
                # solve global problem
                if self.commonPCAz.grad is not None:
                    self.commonPCAz.grad.data.zero_()
                self.loss.backward(retain_graph=True)
                # if i == 0 or i == self.local_epochs:
                #     print("check Z loss", self.loss)
                '''Moving on Grassmannian manifold'''
                # Projection on tangent space
                projection_matrix = torch.eye(self.commonPCAz.shape[0]) - torch.matmul(self.commonPCAz, self.commonPCAz.T)
                projection_gradient = torch.matmul(projection_matrix, self.commonPCAz.grad)
                temp = temp - self.learning_rate * projection_gradient
                # Exponential mapping to Grassmannian manifold by QR retraction
                q, r = torch.linalg.qr(temp)
                self.commonPCAz = q.data.clone()

    def select_users(self, round, fac_users):
        if(fac_users == 1):
            print("All users are selected")
            # for user in self.users:
            #     print("user_id", user.id)
            return self.users
        num_users = int(fac_users * len(self.users))
        num_users = min(num_users, len(self.users))
        #np.random.seed(round)
        return np.random.choice(self.users, num_users, replace=False) #, p=pk)

    # Save loss, accurancy to h5 fiel
    def train_error_and_loss(self):
        num_samples = []
        losses = []
        for c in self.selected_users:
            cl, ns = c.train_error_and_loss()
            num_samples.append(ns)
            losses.append(cl*1.0)
        
        ids = [c.id for c in self.users]

        return ids, num_samples, losses

    def evaluate(self):
        stats_train = self.train_error_and_loss()
        # print(f"stats_train: {stats_train}")
        # train_loss = sum(stats_train[2])/len(self.users)
        train_loss = torch.sqrt(sum(cl*ns for ns, cl in list(zip(stats_train[1], stats_train[2])))) # Jiayu: replace train_loss by reconstruction error
        self.rs_train_loss.append(train_loss)
        if(self.experiment):
            self.experiment.log_metric("train_loss",train_loss)
        #print("stats_train[1]",stats_train[3][0])
        print("Average Global Trainning Loss: ",train_loss)
        return train_loss
    
    def evaluate_all_data(self):
        # print("check data", self.all_train_data.detach().numpy()[:3,1000:1003]) # verified, data no problems
        # 
        # result_path = os.path.join(os.path.join(os.getcwd(), "results/SSA"), 'commonPCAz')
        # np.save(result_path, self.commonPCAz.detach().numpy())
        residual = torch.matmul((torch.eye(self.commonPCAz.shape[0]) - torch.matmul(self.commonPCAz, self.commonPCAz.T)), self.all_train_data.to(torch.float64))
        # residual verified, same as below
        # residual2 = self.all_train_data.to(torch.float64) - torch.matmul(torch.matmul(self.commonPCAz,self.commonPCAz.T),self.all_train_data.to(torch.float64))
        # loss_train verified, same as below
        loss_train = torch.norm(residual, p="fro")
        # loss_train2 = self.re_error(self.commonPCAz.detach().numpy(), self.all_train_data.to(torch.float64))
        # print("check Z", (torch.matmul(self.commonPCAz.T,self.commonPCAz)- torch.eye(self.commonPCAz.shape[1])).detach().numpy()[:3,:3] )
        # print("print Z", self.commonPCAz.detach().numpy()[:3,:3])
        print("evaluate all data over server", loss_train)
        # print("evaluate all data2", loss_train2)
        return loss_train

    def re_error(self, u, X):
        # comment out codes for array
        # Xhat = u.dot(u.T.dot(X))
        # re_error = np.linalg.norm(X-Xhat)
        Xhat = torch.matmul(u,torch.matmul(u.T, X))
        re_error = torch.norm(X-Xhat)
        return re_error

    def check_train_exists(self):
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

    def save_results(self):
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

    # def save_results(self):
    #     dir_path = "./results"
    #     if not os.path.exists(dir_path):
    #         os.makedirs(dir_path)
    #     alg = self.dataset[1] + "ADMM" + "_" + str(self.learning_rate)  + "_" + str(self.ro) + "_" + str(self.num_users) + "u" + "_" + str(self.batch_size) + "b" + "_" + str(self.local_epochs) 
    #     alg = alg + "_" + str(self.times)
    #     if (len(self.rs_glob_acc) != 0 &  len(self.rs_train_acc) & len(self.rs_train_loss)) :
    #         with h5py.File("./results/"+'{}.h5'.format(alg, self.local_epochs), 'w') as hf:
    #             hf.create_dataset('rs_glob_acc', data=self.rs_glob_acc)
    #             hf.create_dataset('rs_train_acc', data=self.rs_train_acc)
    #             hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
    #             hf.close()

