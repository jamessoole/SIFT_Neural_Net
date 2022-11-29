"""Neural network model."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# 'Tradition CNN Model' for fashionMNIST from
# https://arxiv.org/ftp/arxiv/papers/1904/1904.00197.pdf 


class NeuralNet(torch.nn.Module):

    def __init__(self, lrate,in_size,out_size, epoch=0):
        super(NeuralNet, self).__init__()
        torch.set_num_threads(4)

        self.in_size = in_size
        self.out_size = out_size
        self.h = 128
        self.epoch = epoch

        # self.layers = [
        #     nn.Linear(in_size,self.h),
        #     torch.nn.ReLU(),
        #     nn.Linear(self.h,self.h),
        #     torch.nn.ReLU(),
        #     nn.Linear(self.h,out_size),
        # ]

        # thought - image standardization at beginnning?

        self.layers = [
            nn.Conv2d(1,32, kernel_size=3),
            torch.nn.ReLU(),
            nn.Conv2d(32, 64,kernel_size=3),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Linear(64*12*12,self.out_size),
            torch.nn.ReLU(),
            # dropout in fc, need model.eval()
            nn.Softmax(dim=1)
        ]

        self.net = torch.nn.Sequential(*self.layers)
        
        self.loss_fn = nn.CrossEntropyLoss()
        self.optim = torch.optim.SGD(self.net.parameters(), lr = lrate,nesterov = True, momentum = 0.9, weight_decay = 0.001)
        # self.optim = torch.optim.Adam(self.net.parameters(), lr = lrate)


    def forward(self, x: np.ndarray) -> np.ndarray:
        x = x.view([-1,1,28,28])
        # print('1\n',x)

        x = self.layers[0](x)
        x = self.layers[1](x)
        x = self.layers[2](x)
        x = self.layers[3](x)
        x = x.view(-1, 64*12*12)
        x = self.layers[4](x)
        x = self.layers[5](x)
        # print('2\n',x)
        x = self.layers[6](x)

        return x


    def backward(self, x,y):
        self.optim.zero_grad()
        yhat = self(x)
        # print('yhat',yhat)  # these get really big real quick
        L = self.loss_fn(yhat,y)
        # print('L',L)
        L.backward()
        self.optim.step()
        return float(L.detach().numpy())





