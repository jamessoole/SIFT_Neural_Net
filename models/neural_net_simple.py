"""Neural network model."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class NeuralNet(torch.nn.Module):

    def __init__(self, lrate,in_size,out_size, epoch=0):
        super(NeuralNet, self).__init__()
        torch.set_num_threads(4)

        self.in_size = in_size
        self.out_size = out_size
        self.h = 128
        self.epoch = epoch

        self.layers = [
            nn.Linear(in_size,self.h),
            torch.nn.ReLU(),
            nn.Linear(self.h,self.h),
            torch.nn.ReLU(),
            nn.Linear(self.h,out_size),
            # softmax need if run more?
        ]
        self.net = torch.nn.Sequential(*self.layers)

#         self.fc1 = nn.Linear(in_size,self.h)
#         self.fc2 = nn.Linear(self.h,self.h)
#         self.fc3 = nn.Linear(self.h,out_size)
        
        self.loss_fn = nn.CrossEntropyLoss()
        self.optim = torch.optim.SGD(self.net.parameters(), lr = lrate,nesterov = True, momentum = 0.9, weight_decay = 0.001)
        # self.optim = torch.optim.Adam(self.net.parameters(), lr = lrate)


    def forward(self, x: np.ndarray) -> np.ndarray:
        x = self.net(x)
        
        # x = self.layers[0](x)
#         x = self.layers[1](x)
#         x = self.layers[2](x)
#         x = self.layers[3](x)

        return x


    def backward(self, x,y):
        self.optim.zero_grad()
        yhat = self(x)
        L = self.loss_fn(yhat,y)
        L.backward()
        self.optim.step()
        return float(L.detach().numpy())





