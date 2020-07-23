import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
from math import isnan, sqrt
import torch.nn as nn
import os
import time
import torch
from torch.autograd import Variable
from torch.optim import Adam
from skimage import io
from sklearn.metrics import mean_squared_error
from helper_functions import *

################################################# network class #################################################
class gamma(nn.Module):
    '''Returns a gamma vector for each layer'''
    def __init__(self, num_kernels):
        super(gamma, self).__init__()
        self.num_kernels = num_kernels
        self.weight = nn.Parameter(torch.ones(self.num_kernels), requires_grad=True)
        self.weight.data.normal_(mean=0, std=sqrt(2./9./64.)).clamp_(-0.025,0.025)
    def forward(self, x):
        return  self.weight.view(1,self.num_kernels,1,1).expand_as(x)



class Net(nn.Module):

    def __init__(self, all_params):
        super(Net, self).__init__()

        self.padding_mid = all_params['padding_mid']
        self.num_mid_kernels = all_params['num_mid_kernels']
        self.kernel_size_mid = all_params['kernel_size_mid']
        self.num_mid_layers = all_params['num_mid_layers']

        self.conv_layers = nn.ModuleList([])
        self.gammas = nn.ModuleList([])
        self.running_sd = []

        self.conv_layers.append(nn.Conv2d(1,self.num_mid_kernels, self.kernel_size_mid, padding=self.padding_mid , bias=False))

        for l in range(1,self.num_mid_layers-1):
            self.conv_layers.append(nn.Conv2d(self.num_mid_kernels ,self.num_mid_kernels, self.kernel_size_mid, padding=self.padding_mid , bias=False))
            self.gammas.append(gamma(self.num_mid_kernels))
            self.running_sd.append(0)

        self.conv_layers.append(nn.Conv2d(self.num_mid_kernels,1, self.kernel_size_mid, padding=self.padding_mid , bias=False))


    def forward(self, x):

        ## pass through the denoising function/network

        relu = nn.ReLU(inplace=True)
        x = relu(self.conv_layers[0](x))
        for l in range(1,self.num_mid_layers-1):
            x = self.conv_layers[l](x)
            
            sd_x = torch.sqrt(x.permute(1,0,2,3).contiguous().view(self.num_mid_kernels,-1).var(unbiased = False, dim = 1) + 1e-05)

            if self.conv_layers[l].training:
                x = x / (sd_x.view(1,self.num_mid_kernels,1,1).expand_as(x))
                x = x *  self.gammas[l-1](x)
                self.running_sd[l-1] = (1-.1) * self.running_sd[l-1] + .1 * sd_x.data
            else:
                x = x / Variable(self.running_sd[l-1].view(1,self.num_mid_kernels,1,1).expand_as(x), requires_grad=False)
                x = x *  self.gammas[l-1](x)

            x = relu(x)

        x = self.conv_layers[-1](x)

        return x

