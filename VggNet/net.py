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


class Vgg(nn.Module):

    def __init__(self, num_classes, features):
        super().__init__()
        self.backbone = features
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512 * 7 * 7, 2048),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Linear(2048, num_classes))
    def forward(self,inputs):
        features = self.backbone(inputs)
        features = torch.flatten(features, start_dim=1)  # 从第一维度C开始展开（N,C,H,W)
        # a=features.size()#看一下展开之后是多少
        classification = self.classifier(features)
        return classification


class Feature_mode():
    def __init__(self, net_name):
        super().__init__()

        self.cfgs = {
            'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512,
                      'M'],
        }
        self.vgg(net_name)

    def vgg(self, net_name):
        try:
            self.cfg = self.cfgs[net_name]
        except:
            print("网络输入错误，请选择: 'vgg11'，'vgg13'，'vgg16'，'vgg19' ")
            exit(-1)

    def make_feature(self):
        layers = []
        in_channels = 3
        for v in self.cfg:
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels=in_channels, out_channels=v, kernel_size=3, stride=1, padding=1)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels=v
        return nn.Sequential(*layers)


if __name__ == "__main__":
    feature_net = Feature_mode(net_name="vgg16")
    feature = feature_net.make_feature()
    net = Vgg(num_classes=5, features=feature)
    print(net)
