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

test_batch_size = 500
device = torch.device("cuda:0")
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

test_dataset = torchvision.datasets.MNIST("./data", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=12, drop_last=True)

model = net.all_affine_Net(28 * 28, 300, 100, 10)

model.load_state_dict(torch.load("./weights.pth"))
model.to(device)

criterion = nn.CrossEntropyLoss()
eval_loss = 0
eval_correct_rate = 0
eval_num_correct = 0

model.eval()
with tqdm(total=len(test_loader)) as pbar:
    for img, label in test_loader:
        img = img.to(device)
        label = label.to(device)
        img = img.view(img.size(0), -1)
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
