from torch.utils.tensorboard import SummaryWriter
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
import net
import json
import torchvision.models.mobilenet

train_batch_size = 32
test_batch_size = 32
epoch = 200
lr = 0.001
num_classes = 5
device = torch.device("cuda:0")
# writer = SummaryWriter('runs/resnet_logs')
print(device)

data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),  #
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),  # 这个标准化是根据官网例子弄得
    "test": transforms.Compose([transforms.Resize(256),  # 输入的是int,长宽比固定不变 最小边缩放到256
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

train_dataset = torchvision.datasets.ImageFolder(root="./flower_data/train", transform=data_transform["train"])
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, shuffle=True, batch_size=train_batch_size,
                                           num_workers=12,
                                           drop_last=True)
test_dataset = torchvision.datasets.ImageFolder(root="./flower_data/val", transform=data_transform["test"])
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, shuffle=False, batch_size=test_batch_size,
                                          num_workers=12,
                                          drop_last=True)

train_image_num = len(train_dataset)  # 训练图片总数
test_image_num = len(test_dataset)  # 训练图片总数
print("训练图片总数=", train_image_num)
print("测试图片总数=", test_image_num)

# 翻转后变成{0: 'daisy', 1: 'dandelion', 2: 'roses', 3: 'sunflowers', 4: 'tulips'}
classes = train_dataset.class_to_idx  # 这里会输出{'daisy': 0, 'dandelion': 1, 'roses': 2, 'sunflowers': 3, 'tulips': 4}
classes_r = {}
for key, val in classes.items():
    classes_r[val] = key

# 保存classes为json格式
json_str = json.dumps(classes_r, indent=4)
with open("classes.json", "w") as classes_file:
    classes_file.write(json_str)

model = net.MobileNet_V2(num_classes, alpha=1.0, round_nearest=8)
# print(list(model.parameters()))#这里可以看一下默认参数的样子


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
# 迁移学习
model_weight_path = "mobilenet_v2-b0353104.pth"
pre_weight = torch.load(model_weight_path)  # pre_dict是个字典格式
# 去除最后面的classifier weights
pre_dict = {k: v for k, v in pre_weight.items() if "classifier" not in k}
missing_keys, unexpected_keys = model.load_state_dict(pre_dict, strict=False)

# for param in model.backbone.parameters():  # 冻结backbon里面的参数
#     param.requires_grad = False
# inchannel = model.fc.in_features  # 这里拿出来网络中fc层的输入层数
# model.fc = nn.Linear(inchannel, 5)  # 把网络里面的fc层改掉
model.to(device)
# 开始训练


train_epoch_correct_rate = []
train_epoch_loss_ave = []
eval_epoch_correct_rate = []
eval_epoch_loss_ave = []
best_eval_correct_rate = 0

pbar = tqdm(range(1, epoch + 1))
step = 0
for n_epoch in pbar:
    # 参数
    train_loss = 0
    epoch_num_correct = 0
    # if int(n_epoch) % 5 == 0:
    #     optimizer.param_groups[0]["lr"] *= 0.1  # 优化器中lr的位置
    for img, label in train_loader:
        # img_grid = torchvision.utils.make_grid(img)  #
        # writer.add_image('flower_images', img_grid)  #
        # img = img.to(device)
        # writer.add_graph(model, img)#
        # writer.close()#

        model.train()
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
    optim.lr_scheduler.StepLR(optimizer, 10, 0.5)  # 每10个epoch  lr减半
    train_epoch_correct_rate.append(epoch_num_correct / (len(train_loader) * train_batch_size))
    train_epoch_loss_ave.append(train_loss / len(train_loader))
    # 进度条左侧显示右侧显示
    pbar.set_description("step=%d" % step, "epoch=%d" % n_epoch)
    pbar.set_postfix(loss=train_loss / len(train_loader),
                     correct=epoch_num_correct / (len(train_loader) * train_batch_size),
                     lr=optimizer.param_groups[0]["lr"])

    # predict验证集，保存验证集准确率最高的权重
    model.eval()
    eval_loss = 0
    eval_num_correct = 0
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

    eval_correct_rate = eval_num_correct / len(test_dataset)
    eval_epoch_correct_rate.append(eval_correct_rate)
    eval_epoch_loss_ave.append(eval_loss / len(test_loader))
    # 挑测试集准确率最高的保存权重
    if eval_correct_rate > best_eval_correct_rate:
        best_eval_correct_rate = eval_correct_rate
        torch.save(model.state_dict(), "./weights.pth")

# 数据保存
if "train_epoch_correct_rate" in os.listdir("./"):
    os.remove("./train_epoch_correct_rate")

with open("./train_epoch_correct_rate", "a") as f:
    data_generator = (str(i) + "\n" for i in train_epoch_correct_rate)
    for data in data_generator:
        f.write(data)

if "train_epoch_loss_ave" in os.listdir("./"):
    os.remove("./train_epoch_loss_ave")

with open("./train_epoch_loss_ave", "a") as f:
    data_generator = (str(i) + "\n" for i in train_epoch_loss_ave)
    for data in data_generator:
        f.write(data)

if "test_epoch_correct_rate" in os.listdir("./"):
    os.remove("./test_epoch_correct_rate")

with open("./test_epoch_correct_rate", "a") as f:
    data_generator = (str(i) + "\n" for i in eval_epoch_correct_rate)
    for data in data_generator:
        f.write(data)

if "test_epoch_loss_ave" in os.listdir("./"):
    os.remove("./test_epoch_loss_ave")

with open("./test_epoch_loss_ave", "a") as f:
    data_generator = (str(i) + "\n" for i in eval_epoch_loss_ave)
    for data in data_generator:
        f.write(data)


# 查看网络中总参数量
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


parameter_number = get_parameter_number(model)
print("总参数量=", parameter_number)
