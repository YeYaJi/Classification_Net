import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size) // 2  # kernel_size=3,padding就等于1，kernel_size=1,padding=0
        super(ConvBNReLU, self).__init__(  # 用父类初始化的方法初始化ConvBNReLU 的nn.Conv2d，nn.BatchNorm2d，nn.ReLU6
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                      bias=False),
            # groups=1是普通卷积，groups=in_channel就是dw卷积
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):  # 倒残差结构
    def __init__(self, in_channels, out_channels, stride, expand_ratio):

        super(InvertedResidual, self).__init__()
        hidden_channel = in_channels * expand_ratio  # 第一层卷积核的个数,也就是k*t
        self.use_shortcut = (stride == 1) and in_channels == out_channels  # 是否使用捷近分支

        layer = []
        if expand_ratio != 1:  # 当扩展因子=1时，倒残差结构不需要第一个1x1的卷积层了
            layer.append(ConvBNReLU(in_channels, hidden_channel, kernel_size=1))  # 第一层 (输入->tk)
        layer.extend([
            ConvBNReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),  # 第二层 (输入=输出)
            nn.Conv2d(hidden_channel, out_channels, kernel_size=1, bias=False),  # 第三层，后面跟着线性激活函数y=x，所以就不写激活函数了
            nn.BatchNorm2d(out_channels)
        ])
        self.conv = nn.Sequential(*layer)

    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)


def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


class MobileNet_V2(nn.Module):
    def __init__(self, num_classes=1000, alpha=1.0, round_nearest=8):  # 超参数alpha，卷基层使用卷积核个数的倍率

        super(MobileNet_V2, self).__init__()
        block = InvertedResidual
        input_channel = _make_divisible(32 * alpha, round_nearest)  # backbone开始的卷积，把通道变成32 * alpha，在变成round_nearest的整数倍
        last_channel = _make_divisible(1280 * alpha, round_nearest)  # backbone结束的卷积

        inverted_residual_setting = [
            # t:输出深度倍数,c：输出层数,n：重复几次倒残差,s：第一层的倍数
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        features = []
        features.append(ConvBNReLU(in_channels=3, out_channels=input_channel, stride=2))
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * alpha, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride=stride, expand_ratio=t))
                input_channel = output_channel
        features.append(ConvBNReLU(input_channel, last_channel, kernel_size=1))

        self.backbone = nn.Sequential(*features)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(nn.Dropout2d(0.5), nn.Linear(last_channel, num_classes))

        # 初始化参数
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
