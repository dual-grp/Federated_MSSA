import torch
import os
import json
from FLAlgorithms.users.userbase import User
import copy
import numpy as np

'''Implementation for FedPCA clients''' 

class UserADMM3():
    def __init__(self, algorithm, device, id, train_data, commonPCA, learning_rate, ro, local_epochs, dim, ro_auto):
        self.localPCA   = copy.deepcopy(commonPCA) # local U
        self.localZ     = copy.deepcopy(commonPCA)
        self.localY     = copy.deepcopy(commonPCA)
        self.localT     = torch.matmul(self.localPCA.T, self.localPCA)
        # Jiayu2: add constraint decorrelated, i.e., UT X XT U = diagonal matrix
        self.localQ = torch.matmul(self.localPCA.T, self.localPCA) # phi(Z)=diag(UT XXT U) - (UT XXT U)I, which is a vector with length dim
        # self.localQ = torch.rand(self.localT.size()) # converge to above results
        # self.localQ = torch.zeros(self.localT.size()) # converge to above results
        # Jiayu3：use vech for constraint2
        # window = self.localPCA.size()[0]
        # self.localQ = torch.zeros(window*(window+1)/2-window)
        # print("localQ = ", self.localQ[:3,:3])

        self.ro = ro
        self.device = device
        self.id = id
        self.train_samples = train_data.shape[1] # input size (m,n)
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.dim = dim
        self.train_data = train_data # This line is used for SSA
        self.algorithm = algorithm
        self.localPCA.requires_grad_(True)
        if ro_auto:
            self.ro = int(4 * torch.norm(torch.matmul(self.train_data,self.train_data.T)) / self.train_samples) * 1.0

    def set_commonPCA(self, commonPCA):
        ''' Update local Y: comment this section if we use simplified version of FedPCA where Y^{i+1}=0 after first iteration'''
        self.localZ = commonPCA.data.clone()
        self.localY = self.localY + self.ro * (self.localPCA - self.localZ)
        ''' Update local Y: because Y^{i+1}=0 after first iteration in FedPCA. This following line can be used for simplified version of FedPCA. Uncomment to use'''
        # self.localY = torch.zeros_like(self.localY)
        # update local T
        temp = torch.matmul(self.localPCA.T, self.localPCA) - torch.eye(self.localPCA.shape[1])
        # print("check U", self.id, temp.detach().numpy()[:3,:3])
        # print("check U", self.id, self.localPCA.detach().numpy())
        hU = torch.max(torch.zeros(temp.shape),temp)
        self.localT = self.localT + self.ro * hU
        # Jiayu2: add constraint decorrelated, i.e., UT X XT U = diagonal matrix
        UTXXTU = torch.matmul(torch.matmul(self.localPCA.T, torch.matmul(self.train_data,self.train_data.T)), self.localPCA)
        temp_phiU = torch.diag(torch.diag(UTXXTU)) - UTXXTU
        phiU = torch.max(torch.zeros(temp_phiU.shape),temp_phiU)
        # Jiayu3: use vech for constraint2, i.e., G vech(UTXXTU) = 0, G.shape=(n*(n+1))/2-n, n*(n+1))/2), vechh.shape(n*(n+1))/2,1), where n is window length of U
        # G_mat = self.create_G_mat(num_vertices)
        # vech_UTXXTU = 1
        # phiU = vech_UTXXTU
        self.localQ = self.localQ + self.ro * phiU

    def create_G_mat(self, n):
        G_mat = np.zeros((n*(n-1)//2, n*(n+1)//2))
        tmp_vec = np.cumsum(np.arange(n, 1, -1))
        tmp2_vec = np.append([0], tmp_vec)
        tmp3_vec = np.delete(np.arange(n*(n+1)//2), tmp2_vec)
        for i in range(G_mat.shape[0]):
            G_mat[i, tmp3_vec[i]] = 1
        return G_mat

    def train_error_and_loss(self):
        residual = torch.matmul((torch.eye(self.localPCA.shape[0]) - torch.matmul(self.localPCA, self.localPCA.T)), self.train_data)
        loss_train = torch.norm(residual, p="fro") ** 2 / self.train_samples
        return loss_train , self.train_samples

    def hMax(self):
        temp = torch.matmul(self.localPCA.T, self.localPCA) - torch.eye(self.localPCA.shape[1])
        return torch.max(torch.zeros(temp.shape),temp)

    def train(self, epochs):
        for i in range(self.local_epochs):
            '''Euclidean space'''
            if self.algorithm == "FedPE": 
                self.localPCA.requires_grad_(True)
                residual = torch.matmul(torch.eye(self.localPCA.shape[0])- torch.matmul(self.localPCA, self.localPCA.T), self.train_data)
                UTU = torch.matmul(self.localPCA.T, self.localPCA) - torch.eye(self.localPCA.shape[1])
                hU = torch.max(torch.zeros(UTU.shape), UTU) # apply max due to non-linear constraint
                # Jiayu2: add constraint decorrelated, i.e., UT X XT U = diagonal matrix
                UTXXTU = torch.matmul(torch.matmul(self.localPCA.T, torch.matmul(self.train_data,self.train_data.T)), self.localPCA)
                # temp_phiU =torch.diag(UTXXTU) - torch.matmul(UTXXTU, torch.ones(self.localPCA.shape[1], dtype=torch.float64))
                temp_phiU = torch.diag(torch.diag(UTXXTU)) - UTXXTU
                phiU = torch.max(torch.zeros(temp_phiU.shape),temp_phiU)

                regularization = 0.5 * self.ro * torch.norm(self.localPCA - self.localZ)** 2 + 0.5 * self.ro * torch.norm(hU) ** 2
                # Jiayu2:
                phiU_regularization = 0.5 * self.ro * torch.norm(phiU)** 2
                regularization += phiU_regularization # step2

                frobenius_inner = torch.sum(torch.inner(self.localY, self.localPCA - self.localZ)) + torch.sum(torch.inner(self.localT, hU))
                # Jiayu2: 
                phiU_inner = torch.sum(torch.inner(self.localQ, phiU))
                # print("frobenius_inner, phiU_inner")
                # print(frobenius_inner.item(),phiU_inner.item())
                frobenius_inner += phiU_inner # step1


                self.loss = 1/self.train_samples * torch.norm(residual, p="fro") ** 2 
                self.lossADMM = self.loss + 1/self.train_samples * (frobenius_inner + regularization)
                temp = self.localPCA.data.clone()
                # Solve the local problem
                if self.localPCA.grad is not None:
                    self.localPCA.grad.data.zero_()

                self.lossADMM.backward(retain_graph=True)
                # print('check local loss', self.lossADMM)
                # Update local pca
                temp  = temp - self.learning_rate * self.localPCA.grad
                self.localPCA = temp.data.clone()
                # Jiayu: assert constraint2
                # proj2 = 1/self.train_samples * torch.matmul(torch.matmul(self.localPCA.T, torch.matmul(self.train_data,self.train_data.T)), self.localPCA)
                # proj2 = proj2.detach().numpy()
                # if i==0 and (self.id=='MT_001' or self.id == 0):
                #     print("assert constraint2, print projTproj check diag")
                #     print(proj2[:4,:4])
                 
            else: 
                '''Grassmannian manifold'''
                # if i==0 and self.id == "MT_001":
                #     print('check local loss', self.lossADMM)
                self.localPCA.requires_grad_(True)
                residual = torch.matmul(torch.eye(self.localPCA.shape[0])- torch.matmul(self.localPCA, self.localPCA.T), self.train_data)
                frobenius_inner = torch.sum(torch.inner(self.localY, self.localPCA - self.localZ)) # Jiayu: perhaps should add max
                # Jiayu2: add constraint decorrelated, i.e., UT X XT U = diagonal matrix
                UTXXTU = torch.matmul(torch.matmul(self.localPCA.T, torch.matmul(self.train_data,self.train_data.T)), self.localPCA)
                # temp_phiU =torch.diag(UTXXTU) - torch.matmul(UTXXTU, torch.ones(self.localPCA.shape[1], dtype=torch.float64))
                temp_phiU = torch.diag(torch.diag(UTXXTU)) - UTXXTU
                phiU = torch.max(torch.zeros(temp_phiU.shape),temp_phiU)
                phiU_inner = torch.sum(torch.inner(self.localQ, phiU))
                # print("frobenius_inner, phiU_inner")
                # print(frobenius_inner.item(),phiU_inner.item())
                frobenius_inner += phiU_inner # step1
                regularization = 0.5 * self.ro * torch.norm(self.localPCA - self.localZ)** 2
                phiU_regularization = 0.5 * self.ro * torch.norm(phiU)** 2
                # print("regularization, phiU_regularization")
                # print(regularization.item(), phiU_regularization.item())
                # print("======")
                regularization += phiU_regularization # step2

                self.loss = 1/self.train_samples * torch.norm(residual, p="fro") ** 2
                self.lossADMM = self.loss + 1/self.train_samples * (frobenius_inner + regularization)
                temp = self.localPCA.data.clone()
                # solve local problem locally
                if self.localPCA.grad is not None:
                    self.localPCA.grad.data.zero_()
                self.lossADMM.backward(retain_graph=True)
                # if i==0 and self.id == "MT_001":
                #     print('check local loss', self.lossADMM)
                '''Moving on Grassmannian manifold'''
                # Projection on tangent space
                projection_matrix = torch.eye(self.localPCA.shape[0]) - torch.matmul(self.localPCA, self.localPCA.T)
                projection_gradient = torch.matmul(projection_matrix, self.localPCA.grad)
                temp = temp - self.learning_rate * projection_gradient
                # Exponential mapping to Grassmannian manifold by QR retraction
                q, r = torch.linalg.qr(temp)
                self.localPCA = q.data.clone()
                # Jiayu: assert constraint2
                # proj2 = 1/self.train_samples * torch.matmul(torch.matmul(self.localPCA.T, torch.matmul(self.train_data,self.train_data.T)), self.localPCA)
                # proj2 = proj2.detach().numpy()
                # if i==0 and (self.id=='MT_001' or self.id == 0):
                #     print("assert constraint2, print projTproj check diag")
                #     print(proj2[:4,:4])
        return  
