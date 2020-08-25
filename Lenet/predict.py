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

test_batch_size = 100
device = torch.device("cuda:0")
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
test_dataset = torchvision.datasets.CIFAR10(root="./data1", train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, shuffle=False, batch_size=test_batch_size,
                                          num_workers=12,
                                          drop_last=True)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

model = net.LeNet()

model.load_state_dict(torch.load("./weights.pth"))
model.to(device)
model.eval()
criterion = nn.CrossEntropyLoss()
eval_loss = 0
eval_correct_rate = 0
eval_num_correct = 0

with tqdm(total=len(test_loader)) as pbar:
    for img, label in test_loader:
        img = img.to(device)
        label = label.to(device)
        output = model(img)
        loss = criterion(output, label)
        eval_loss += loss.item()
        _, max_index = output.max(1)
        for i in range(test_batch_size):
            if max_index[i] == label[i]:
                eval_num_correct += 1
        pbar.update(1)
eval_correct_rate = (eval_num_correct / (len(test_loader) * test_batch_size))
eval_loss_ave = (eval_loss / len(test_loader))

print("eval_correct_rate=", eval_correct_rate)
print("eval_loss_ave=", eval_loss_ave)
