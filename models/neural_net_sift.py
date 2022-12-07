
# 'SIFT Descriptor CNN Model' for fashionMNIST from
# https://arxiv.org/ftp/arxiv/papers/1904/1904.00197.pdf 

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pytorch_sift import SIFTNet
SIFT = SIFTNet(patch_size = 24)
SIFT.eval()


class NeuralNet(torch.nn.Module):

    def __init__(self, lrate,in_size,out_size):
        super(NeuralNet, self).__init__()
        torch.set_num_threads(4)

        self.in_size = in_size
        self.out_size = out_size
        
        
#         # ideal, given computational power
#         self.layers = [
#             nn.Conv2d(1,32, kernel_size=3),
#             torch.nn.ReLU(),
#             nn.Conv2d(32, 64,kernel_size=3),
#             SIFT,
#             nn.Linear(64*128,self.out_size),
#             torch.nn.ReLU(),
#             nn.Softmax(dim=1)
#         ]


        # 3 channels
        self.layers = [
            nn.Conv2d(1,32, kernel_size=3),
            torch.nn.ReLU(),
            nn.Conv2d(32, 3,kernel_size=3),
            SIFT,
            nn.Linear(3*128,self.out_size),
            torch.nn.ReLU(),
            nn.Softmax(dim=1)
        ]
        
#         # 1 channel       
#         self.layers = [
#             nn.Conv2d(1,32, kernel_size=3),
#             torch.nn.ReLU(),
#             nn.Conv2d(32, 1,kernel_size=3), # downgraded number of channels
#             SIFT,
#             nn.Linear(1*128,self.out_size),
#             torch.nn.ReLU(),
#             nn.Softmax(dim=1)
#         ]
        
#         # comparison/traditional
#         self.layers = [
#             nn.Conv2d(1,32, kernel_size=3),
#             torch.nn.ReLU(),
#             nn.Conv2d(32, 1,kernel_size=3), # 1 or 3 channels
#             nn.MaxPool2d(kernel_size=2,stride=2),
#             nn.Linear(1*12*12,self.out_size),
#             torch.nn.ReLU(),
#             nn.Softmax(dim=1)
#         ]

        self.net = torch.nn.Sequential(*self.layers)
        
        self.loss_fn = nn.CrossEntropyLoss()
        self.optim = torch.optim.SGD(self.net.parameters(), lr = lrate,nesterov = True, momentum = 0.9, weight_decay = 0.001)


    def forward(self, x: np.ndarray) -> np.ndarray:
        x = x.view([-1,1,28,28])

              
        # IDEAL or or any number of fewer layers
        x = self.layers[0](x) # conv
        x = self.layers[1](x) # relu
        x = self.layers[2](x) # conv # 200x64x24x24
        sift_outs = torch.zeros(x.shape[0],x.shape[1],128) # 200x64x128
        for channel in range(x.shape[1]):
            sift_outs[:,channel,:] = self.layers[3](x[:,[channel]]) # 200x128
        x = sift_outs
        del sift_outs
        
        x = x.view(x.shape[0],-1) # 200x64*128 = 200x8192
        x = self.layers[4](x) # fc
        x = self.layers[5](x) # relu
        x = self.layers[6](x) # softmax
        
    
    
#         # DOWNGRADED - 1 channel
#         x = self.layers[0](x) # conv
#         x = self.layers[1](x) # relu
#         x = self.layers[2](x) # conv
#         x = self.layers[3](x) # SIFT, returns 1 channel
#         x = self.layers[4](x) # fc       
#         x = self.layers[5](x) # relu
#         x = self.layers[6](x) # softmax
        
        

        
#         # COMPARISON
#         x = self.layers[0](x) # conv
#         x = self.layers[1](x) # relu
#         x = self.layers[2](x) # conv
#         x = self.layers[3](x) # maxpool
#         x = x.view(-1, 1*12*12) # to change: num layers, 1 or 3
#         x = self.layers[4](x) # fc
#         x = self.layers[5](x) # relu
#         x = self.layers[6](x) # softmax

        return x


    def backward(self, x,y):
        self.optim.zero_grad()
        yhat = self(x)
        L = self.loss_fn(yhat,y)
        L.backward()
        self.optim.step()
        return float(L.detach().numpy())



