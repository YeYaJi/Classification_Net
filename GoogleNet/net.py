import torch
import torch.nn as nn
import torch.nn.functional as F


class GoogleNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=2)
        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)  # 这里ceil_mode=True是向上取整，相当于padding=1的向下取整
        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

        # 辅助分类器
        self.aux1 = InceptionAux(512, num_classes)
        self.aux2 = InceptionAux(528, num_classes)

    def forward(self, input):
        aux1 = 0
        aux2 = 0
        # N x 3 x 224 x 224
        input = F.relu(self.conv1(input), inplace=True)  # N x 64 x 112 x 112
        # print(input.shape)
        input = self.maxpool1(input)  # N x 64 x 56 x 56
        # print(input.shape)
        input = F.relu(self.conv2(input), inplace=True)  # N x 64 x 56 x 56
        # print(input.shape)
        input = F.relu(self.conv3(input), inplace=True)  # N x 192 x 56 x 56
        # print(input.shape)
        input = self.maxpool1(input)  # N x 192 x 28 x 28
        # print("pool1=", input.shape)
        input = self.inception3a(input)  # N x 256 x 28 x 28
        # print("3a=", input.shape)
        input = self.inception3b(input)  # N x 480 x 28 x 28
        # print("3b=", input.shape)
        input = self.maxpool3(input)  # N x 480 x 14 x 14
        # print(input.shape)
        input = self.inception4a(input)  # N x 512 x 14 x 14
        # print(input.shape)
        # 辅助分类器
        if self.training:
            aux1 = self.aux1(input)

        input = self.inception4b(input)  # N x 512 x 14 x 14
        input = self.inception4c(input)  # N x 512 x 14 x 14
        input = self.inception4d(input)  # N x 528 x 14 x 14

        # 辅助分类器
        if self.training:
            aux2 = self.aux2(input)

        input = self.inception4e(input)  # N x 832 x 14 x 14
        input = self.maxpool4(input)  # N x 832 x 7 x 7
        input = self.inception5a(input)  # N x 832 x 7 x 7
        input = self.inception5b(input)  # N x 1024 x 7 x 7
        input = self.avgpool(input)  # N x 1024 x 1 x 1
        input = torch.flatten(input, 1)  # N x 1024
        output = self.fc(input)  # N x 1000 (num_classes)
        if self.training:
            return output, aux1, aux2
        return output


# 建立Inception结构
class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3_r, ch3x3, ch5x5_r, ch5x5, ch_pool):
        super(Inception, self).__init__()
        self.batch1 = nn.Conv2d(in_channels=in_channels, out_channels=ch1x1, kernel_size=1, stride=1)
        self.batch2 = nn.Sequential(nn.Conv2d(in_channels, ch3x3_r, kernel_size=1, stride=1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(ch3x3_r, ch3x3, kernel_size=3, stride=1,
                                              padding=1))  # 这里的padding是为了保持输入输出的H，W不变
        self.batch3 = nn.Sequential(nn.Conv2d(in_channels, ch5x5_r, kernel_size=1, stride=1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(ch5x5_r, ch5x5, kernel_size=5, stride=1,
                                              padding=2))  # 这里的padding是为了保持输入输出的H，W不变
        self.batch4 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                                    nn.Conv2d(in_channels, ch_pool, kernel_size=1, stride=1))

    def forward(self, inputs):
        output1 = F.relu(self.batch1(inputs))
        output2 = F.relu(self.batch2(inputs))
        output3 = F.relu(self.batch3(inputs))
        output4 = F.relu(self.batch4(inputs))
        outputs = torch.cat((output1, output2, output3, output4), 1)  # N,C,H,W 拼第C维
        return outputs


# 建立辅助分类器网络
class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.averagePool = nn.AvgPool2d(kernel_size=5, stride=3)  # [14,14]->[4,4]   [H,W]
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=1, stride=1)
        self.line_1 = nn.Linear(128 * 4 * 4, 1024)
        self.line_2 = nn.Linear(1024, num_classes)

    def forward(self, input):
        input = self.averagePool(input)
        input = F.relu(self.conv(input), inplace=True)
        input = torch.flatten(input, start_dim=1)
        input = F.dropout(input, p=0.7,
                          training=self.training)  # self.training是nn.Modeld中定义的，在model.train()是true，在model.eval()是False
        input = F.relu(self.line_1(input))
        input = F.dropout(input, p=0.7, training=self.training)
        output = self.line_2(input)
        return output


if __name__ == "__main__":
    model = GoogleNet(num_classes=5)
    print(model)
