import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
from FLAlgorithms.users.userbase import User

# Implementation for FedAvg clients

class UserLR(User):
    def __init__(self, device, numeric_id, train_data, test_data, model, batch_size, learning_rate, beta, L_k,
                 local_epochs, optimizer):
        super().__init__(device, numeric_id, train_data, test_data, model, batch_size, learning_rate, beta, L_k,
                         local_epochs)

        self.loss = torch.nn.MSELoss()

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

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
                y = y.view(-1, 1)
                # print(X.shape)
                # print(y.shape)
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(X)
                # print(output.shape)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()
        if self.id == 1:
            print(f"loss: {loss}")
        self.clone_model_paramenter(self.model.parameters(), self.local_model)
        return LOSS
