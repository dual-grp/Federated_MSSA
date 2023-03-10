import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
from FLAlgorithms.users.userbase import User

# Implementation for FedAvg clients

class UserLSTM(User):
    def __init__(self, device, numeric_id, train_data, test_data, model, batch_size, learning_rate, beta, L_k,
                 local_epochs, optimizer):
        super().__init__(device, numeric_id, train_data, test_data, model, batch_size, learning_rate, beta, L_k,
                         local_epochs)

        self.loss = torch.nn.MSELoss()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def set_grads(self, new_grads):
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.model.parameters(), new_grads):
                model_grad.data = new_grad.data
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.model.parameters()):
                model_grad.data = new_grads[idx]

    def train(self, epochs):
        LOSS = 0
        self.model.train()
        for epoch in range(1, self.local_epochs + 1):
            self.model.train()
            for X,y in self.trainloader:
                X, y = X.to(self.device), y.to(self.device)#self.get_next_train_batch()
                self.optimizer.zero_grad()
                output = self.model(X.view(-1,40,1)).reshape(-1)
                # print(f"output: {output.shape}")
                # print(f"y: {y.shape}")
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()
        if self.id == 1:
            print(f"loss: {loss}")
        self.clone_model_paramenter(self.model.parameters(), self.local_model)
        return LOSS
