import torch
import os
import json
from FLAlgorithms.users.userbase import User
import copy

'''Implementation for FedPCA clients''' 

class UserADMM2():
    def __init__(self, algorithm, device, id, train_data, commonPCA, learning_rate, ro, local_epochs, dim, ro_auto):
        self.localPCA   = copy.deepcopy(commonPCA) # local U
        self.localZ     = copy.deepcopy(commonPCA)
        self.localY     = copy.deepcopy(commonPCA)
        self.localT     = torch.matmul(self.localPCA.T, self.localPCA)
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
            self.ro = int(4 * torch.norm(torch.matmul(self.train_data,self.train_data.T)) / self.train_data.shape[1]) * 1.0

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
        hU = torch.max(torch.zeros(temp.shape),temp)**2
        self.localT = self.localT + self.ro * hU

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
                hU = torch.max(torch.zeros(UTU.shape), UTU)**2
                regularization = 0.5 * self.ro * torch.norm(self.localPCA - self.localZ)** 2 + 0.5 * self.ro * torch.norm(hU) ** 2
                frobenius_inner = torch.sum(torch.inner(self.localY, self.localPCA - self.localZ)) + torch.sum(torch.inner(self.localT, hU))
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
                 
            else: 
                '''Grassmannian manifold'''
                # if i==0 and self.id == "MT_001":
                #     print('check local loss', self.lossADMM)
                self.localPCA.requires_grad_(True)
                residual = torch.matmul(torch.eye(self.localPCA.shape[0])- torch.matmul(self.localPCA, self.localPCA.T), self.train_data)
                frobenius_inner = torch.sum(torch.inner(self.localY, self.localPCA - self.localZ))
                regularization = 0.5 * self.ro * torch.norm(self.localPCA - self.localZ)** 2
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
        return  