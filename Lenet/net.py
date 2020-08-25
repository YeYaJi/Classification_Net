import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

import os
from tqdm import tqdm, trange
import time


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # F.MaxPool2d有点问题不能这么用
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, input):
        layer_1_out = self.maxpool1(F.relu(self.conv1(input)))
        layer_2_out = self.maxpool1(F.relu(self.conv2(layer_1_out)))
        layer_1d = layer_2_out.view(-1, 32 * 5 * 5)
        layer_af1_out = F.relu(self.fc1(layer_1d))
        layer_af2_out = F.relu(self.fc2(layer_af1_out))
        layer_af3_out = self.fc3(layer_af2_out)
        return layer_af3_out
