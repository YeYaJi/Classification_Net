import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import net
import os
from tqdm import tqdm, trange
import time
import PIL
import cv2

image = "/home/gsh/Desktop/mmexport1501259902293.jpg"

transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),  # 注意注意resize必须在Totensor前面，阿西吧什么鬼规定
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

device = torch.device("cuda:0")
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
model = net.LeNet()
model.load_state_dict(torch.load("./weights.pth"))
model.to(device)
model.eval()
img = PIL.Image.open(image)  # h,w,c
# img.show()
# img=img.resize((32,32))
# img.show()
# print(img.size)
# print(img.mode)
img = transform(img)
img = torch.unsqueeze(img, dim=0)
img = img.to(device)
output = model(img)
_, max_index = output.max(1)
print(classes[max_index])
