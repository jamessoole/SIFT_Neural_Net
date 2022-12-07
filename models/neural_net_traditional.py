
# 'Tradition CNN Model' for fashionMNIST from
# https://arxiv.org/ftp/arxiv/papers/1904/1904.00197.pdf 


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class NeuralNet(torch.nn.Module):

    def __init__(self, lrate,in_size,out_size):
        super(NeuralNet, self).__init__()
        torch.set_num_threads(4)

        self.in_size = in_size
        self.out_size = out_size

        self.layers = [
            nn.Conv2d(1,32, kernel_size=3),
            torch.nn.ReLU(),
            nn.Conv2d(32, 64,kernel_size=3),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Linear(64*12*12,self.out_size),
            torch.nn.ReLU(),
            nn.Softmax(dim=1)
        ]

        self.net = torch.nn.Sequential(*self.layers)
        
        self.loss_fn = nn.CrossEntropyLoss()
        self.optim = torch.optim.SGD(self.net.parameters(), lr = lrate,nesterov = True, momentum = 0.9, weight_decay = 0.001)


    def forward(self, x: np.ndarray) -> np.ndarray:
        x = x.view([-1,1,28,28])

        x = self.layers[0](x) # conv
        x = self.layers[1](x) # relu
        x = self.layers[2](x) # conv
        x = self.layers[3](x) # pool
        
        x = x.view(-1, 64*12*12)
        x = self.layers[4](x) # fc
        x = self.layers[5](x) # relu
        x = self.layers[6](x) # softmax

        return x


    def backward(self, x,y):
        self.optim.zero_grad()
        yhat = self(x)
        L = self.loss_fn(yhat,y)
        L.backward()
        self.optim.step()
        return float(L.detach().numpy())





