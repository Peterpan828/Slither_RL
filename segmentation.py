import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import os
import numpy as np
import matplotlib.pyplot as plt
import time

class UNET(nn.Module):
    def __init__(self, i_ch, o_ch):
        super().__init__()
        
        self.conv1 = self.contraction_block(i_ch, 32, 7, 3)
        self.conv2 = self.contraction_block(32, 64, 3, 1)
        self.conv3 = self.contraction_block(64, 128 ,3 ,1)
        
        self.upconv3 = self.expansive_block(128, 64, 3, 1)
        self.upconv2 = self.expansive_block(64*2, 32, 3, 1)
        self.upconv1 = self.expansive_block(32*2, o_ch, 3, 1)
        
    def __call__(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        
        upconv3 = self.upconv3(conv3)
        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))
        
        return upconv1
    
    def expansive_block(self, i_ch, o_ch, filter_size, padding):
        block = nn.Sequential(torch.nn.Conv2d(i_ch, o_ch, kernel_size = filter_size, stride = 1, padding = padding ),
                             torch.nn.BatchNorm2d(o_ch),
                             torch.nn.ReLU(),
                             torch.nn.Conv2d(o_ch, o_ch, kernel_size = filter_size, stride = 1, padding = padding ),
                             torch.nn.BatchNorm2d(o_ch),
                             torch.nn.ReLU(),
                             torch.nn.ConvTranspose2d(o_ch, o_ch, kernel_size = 3, stride = 2, padding = 1, output_padding = 1))
        return block

    def contraction_block(self, i_ch, o_ch, filter_size, padding):
        block = nn.Sequential(torch.nn.Conv2d(i_ch, o_ch, kernel_size = filter_size, stride = 1, padding = padding ),
                             torch.nn.BatchNorm2d(o_ch),
                             torch.nn.ReLU(),
                             torch.nn.Conv2d(o_ch, o_ch, kernel_size = filter_size, stride = 1, padding = padding ),
                             torch.nn.BatchNorm2d(o_ch),
                             torch.nn.ReLU(),
                             torch.nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1))
        return block




