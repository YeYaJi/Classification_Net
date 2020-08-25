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




class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()  # 卷积后图像的ＨＷ会向下取整
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=(2, 2)),
            nn.ReLU(inplace=True),  # inplace是可以降低消耗的内存
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(6*6*256, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 5))

    def forward(self, inputs):
        features = self.backbone(inputs)
        features = torch.flatten(features, start_dim=1)  # 从第一维度C开始展开（N,C,H,W)
        # a=features.size()#看一下展开之后是多少
        classification = self.classifier(features)
        return classification

    # 这里pytorch默认会用如下的方法进行初始化要学习的参数
    def _initialize_weights(self):
        for m in self.modules():  # Returns an iterator over all modules in the network,返回网络中所有的层
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)  # 如果权重存在的话，用0初始化
            elif isinstance(m, nn.Linear):
                m.init.normal_(m.weight, 0, 0.01)  # 正态分布赋予初值
                nn.init.constant_(m.bias, 0)

if __name__=="__main__":
    model = AlexNet()
    print(model)