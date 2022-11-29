"""Neural network model."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim




# HAVENT DONE YET
# TODO: incorporate sift oh boyyyy



class NeuralNet(torch.nn.Module):

    def __init__(self, lrate,in_size,out_size):
        super(NeuralNet, self).__init__()
        torch.set_num_threads(4)

        self.in_size = in_size
        self.out_size = out_size
        self.h = 128

        # Shallow Net
        self.layers = [
            nn.Linear(in_size,self.h),
            torch.nn.ReLU(),
            nn.Linear(self.h,self.h),
            torch.nn.ReLU(),
            nn.Linear(self.h,out_size),
        ]

        self.net = torch.nn.Sequential(*self.layers)

        self.loss_fn = nn.CrossEntropyLoss()
        self.optim = torch.optim.SGD(self.net.parameters(), lr = lrate,nesterov = True, momentum = 0.9, weight_decay = 0.001)
        # self.optim = torch.optim.Adam(self.net.parameters(), lr = lrate)


    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.net(x)



    def backward(self, x,y):
        self.optim.zero_grad()
        yhat = self(x)
        L = self.loss_fn(yhat,y)
        L.backward()
        self.optim.step()
        return float(L.detach().numpy())





