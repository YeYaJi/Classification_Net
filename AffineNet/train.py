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

train_batch_size = 500
test_batch_size = 500
learning_rate = 0.01
epoch = 200
lr = 0.01
momentum = 0.5

device = torch.device("cuda:0")
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
train_dataset = torchvision.datasets.MNIST("./data", train=True, transform=transform, download=False)
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, drop_last=True, num_workers=12)

model = net.All_affine_Net(28 * 28, 300, 100, 10)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

# 开始train
model.train()

epoch_correct_rate = []
epoch_loss_rate = []

pbar = tqdm(range(1, epoch + 1))
for n_epoch in pbar:
    # 参数
    train_loss = 0
    epoch_num_correct = 0
    if int(n_epoch) % 5 == 0:
        optimizer.param_groups[0]["lr"] *= 0.1  # 优化器中lr的位置
    for img, label in train_loader:

        img = img.to(device)
        label = label.to(device)
        img = img.view(img.size(0), -1)
        output = model(img)
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        # 计算准确率
        _, max_index = output.max(1)
        for i in range(train_batch_size):
            if max_index[i] == label[i]:
                epoch_num_correct += 1  # 一个batch中正确的个数

    epoch_correct_rate.append(epoch_num_correct / (len(train_loader) * train_batch_size))
    epoch_loss_rate.append(train_loss / len(train_loader))
    # 进度条左侧显示右侧显示
    pbar.set_description("epoch=%d" % n_epoch)
    pbar.set_postfix(loss=train_loss / len(train_loader), correct=epoch_num_correct / (len(train_loader) * train_batch_size),
                     lr=optimizer.param_groups[0]["lr"])

# 数据保存
torch.save(model.state_dict(), "./weights.pth")

if "train_epoch_correct_rate" in os.listdir("./"):
    os.remove("./train_epoch_correct_rate")

with open("./train_epoch_correct_rate", "a") as f:
    data_generator = (str(i) + "\n" for i in epoch_correct_rate)
    for data in data_generator:
        f.write(data)

if "train_epoch_loss_rate" in os.listdir("./"):
    os.remove("./train_epoch_loss_rate")

with open("./train_epoch_loss_rate", "a") as f:
    data_generator = (str(i) + "\n" for i in epoch_loss_rate)
    for data in data_generator:
        f.write(data)
