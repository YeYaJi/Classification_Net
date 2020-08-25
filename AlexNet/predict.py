import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import json
import os
from tqdm import tqdm, trange
import time
import net

test_batch_size = 10  # test_batch_size必须<=len(test_dataset)
device = torch.device("cuda:0")
transform = transforms.Compose(
    [transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
test_dataset = torchvision.datasets.ImageFolder(root="./flower_data/val", transform=transform)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, shuffle=False, batch_size=test_batch_size,
                                          num_workers=12,
                                          drop_last=True)
# print((len(test_dataset)))
# print(len(test_loader))
# len(test_dataset)==len(test_loader)*test_batch_size
model = net.AlexNet()
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

eval_correct_rate = (eval_num_correct / len(test_dataset))
eval_loss_ave = (eval_loss / len(test_loader))

print("eval_correct_rate=", eval_correct_rate)
print("eval_loss_ave=", eval_loss_ave)
