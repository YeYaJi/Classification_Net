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

train_batch_size = 50

epoch = 200
lr = 0.001

device = torch.device("cuda:0")
transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset = torchvision.datasets.ImageFolder(root="./flower_data/train", transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, shuffle=True, batch_size=train_batch_size,
                                           num_workers=12,
                                           drop_last=True)

train_image_num = len(train_dataset)  # 训练图片总数
print("训练图片总数=", train_image_num)

classes = train_dataset.class_to_idx  # 这里会输出{'daisy': 0, 'dandelion': 1, 'roses': 2, 'sunflowers': 3, 'tulips': 4}
classes_r = {}  # 翻转后变成{0: 'daisy', 1: 'dandelion', 2: 'roses', 3: 'sunflowers', 4: 'tulips'}
for key, val in classes.items():
    classes_r[val] = key

# 保存classes为json格式
json_str = json.dumps(classes_r, indent=4)
with open("classes.json", "w") as classes_file:
    classes_file.write(json_str)

model = net.AlexNet()
# print(list(model.parameters()))#这里可以看一下默认参数的样子

model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
# 开始训练
model.train()

epoch_correct_rate = []
epoch_loss_rate = []
pbar = tqdm(range(1, epoch + 1))
step = 0
for n_epoch in pbar:
    # 参数
    train_loss = 0
    epoch_num_correct = 0
    # if int(n_epoch) % 5 == 0:
    #     optimizer.param_groups[0]["lr"] *= 0.1  # 优化器中lr的位置
    for img, label in train_loader:
        step += 1
        img = img.to(device)
        label = label.to(device)

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
    pbar.set_description("step=%d" % step, "epoch=%d" % n_epoch)
    pbar.set_postfix(loss=train_loss / len(train_loader),
                     correct=epoch_num_correct / (len(train_loader) * train_batch_size),
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




# 查看网络中总参数量
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


parameter_number = get_parameter_number(model)
print("总参数量=", parameter_number)
