# GoogleNet
创新点在于
   1. Inception结构（并行结构）
   2. 1X1卷积核进行降维（显著减少参数）
   3. 2个辅助分类器

网络结构如下
```python
GoogleNet(
  (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
  (maxpool1): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  (conv2): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
  (conv3): Conv2d(64, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (maxpool2): MaxPool2d(kernel_size=3, stride=2, padding=2, dilation=1, ceil_mode=False)
  (inception3a): Inception(
    (batch1): Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1))
    (batch2): Sequential(
      (0): Conv2d(192, 96, kernel_size=(1, 1), stride=(1, 1))
      (1): ReLU(inplace=True)
      (2): Conv2d(96, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
    (batch3): Sequential(
      (0): Conv2d(192, 16, kernel_size=(1, 1), stride=(1, 1))
      (1): ReLU(inplace=True)
      (2): Conv2d(16, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    )
    (batch4): Sequential(
      (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
      (1): Conv2d(192, 32, kernel_size=(1, 1), stride=(1, 1))
    )
  )
  (inception3b): Inception(
    (batch1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
    (batch2): Sequential(
      (0): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
      (1): ReLU(inplace=True)
      (2): Conv2d(128, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
    (batch3): Sequential(
      (0): Conv2d(256, 32, kernel_size=(1, 1), stride=(1, 1))
      (1): ReLU(inplace=True)
      (2): Conv2d(32, 96, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    )
    (batch4): Sequential(
      (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
      (1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
    )
  )
  (maxpool3): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
  (inception4a): Inception(
    (batch1): Conv2d(480, 192, kernel_size=(1, 1), stride=(1, 1))
    (batch2): Sequential(
      (0): Conv2d(480, 96, kernel_size=(1, 1), stride=(1, 1))
      (1): ReLU(inplace=True)
      (2): Conv2d(96, 208, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
    (batch3): Sequential(
      (0): Conv2d(480, 16, kernel_size=(1, 1), stride=(1, 1))
      (1): ReLU(inplace=True)
      (2): Conv2d(16, 48, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    )
    (batch4): Sequential(
      (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
      (1): Conv2d(480, 64, kernel_size=(1, 1), stride=(1, 1))
    )
  )
  (inception4b): Inception(
    (batch1): Conv2d(512, 160, kernel_size=(1, 1), stride=(1, 1))
    (batch2): Sequential(
      (0): Conv2d(512, 112, kernel_size=(1, 1), stride=(1, 1))
      (1): ReLU(inplace=True)
      (2): Conv2d(112, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
    (batch3): Sequential(
      (0): Conv2d(512, 24, kernel_size=(1, 1), stride=(1, 1))
      (1): ReLU(inplace=True)
      (2): Conv2d(24, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    )
    (batch4): Sequential(
      (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
      (1): Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1))
    )
  )
  (inception4c): Inception(
    (batch1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))
    (batch2): Sequential(
      (0): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))
      (1): ReLU(inplace=True)
      (2): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
    (batch3): Sequential(
      (0): Conv2d(512, 24, kernel_size=(1, 1), stride=(1, 1))
      (1): ReLU(inplace=True)
      (2): Conv2d(24, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    )
    (batch4): Sequential(
      (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
      (1): Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1))
    )
  )
  (inception4d): Inception(
    (batch1): Conv2d(512, 112, kernel_size=(1, 1), stride=(1, 1))
    (batch2): Sequential(
      (0): Conv2d(512, 144, kernel_size=(1, 1), stride=(1, 1))
      (1): ReLU(inplace=True)
      (2): Conv2d(144, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
    (batch3): Sequential(
      (0): Conv2d(512, 32, kernel_size=(1, 1), stride=(1, 1))
      (1): ReLU(inplace=True)
      (2): Conv2d(32, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    )
    (batch4): Sequential(
      (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
      (1): Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1))
    )
  )
  (inception4e): Inception(
    (batch1): Conv2d(528, 256, kernel_size=(1, 1), stride=(1, 1))
    (batch2): Sequential(
      (0): Conv2d(528, 160, kernel_size=(1, 1), stride=(1, 1))
      (1): ReLU(inplace=True)
      (2): Conv2d(160, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
    (batch3): Sequential(
      (0): Conv2d(528, 32, kernel_size=(1, 1), stride=(1, 1))
      (1): ReLU(inplace=True)
      (2): Conv2d(32, 128, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    )
    (batch4): Sequential(
      (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
      (1): Conv2d(528, 128, kernel_size=(1, 1), stride=(1, 1))
    )
  )
  (maxpool4): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
  (inception5a): Inception(
    (batch1): Conv2d(832, 256, kernel_size=(1, 1), stride=(1, 1))
    (batch2): Sequential(
      (0): Conv2d(832, 160, kernel_size=(1, 1), stride=(1, 1))
      (1): ReLU(inplace=True)
      (2): Conv2d(160, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
    (batch3): Sequential(
      (0): Conv2d(832, 32, kernel_size=(1, 1), stride=(1, 1))
      (1): ReLU(inplace=True)
      (2): Conv2d(32, 128, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    )
    (batch4): Sequential(
      (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
      (1): Conv2d(832, 128, kernel_size=(1, 1), stride=(1, 1))
    )
  )
  (inception5b): Inception(
    (batch1): Conv2d(832, 384, kernel_size=(1, 1), stride=(1, 1))
    (batch2): Sequential(
      (0): Conv2d(832, 192, kernel_size=(1, 1), stride=(1, 1))
      (1): ReLU(inplace=True)
      (2): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
    (batch3): Sequential(
      (0): Conv2d(832, 48, kernel_size=(1, 1), stride=(1, 1))
      (1): ReLU(inplace=True)
      (2): Conv2d(48, 128, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    )
    (batch4): Sequential(
      (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
      (1): Conv2d(832, 128, kernel_size=(1, 1), stride=(1, 1))
    )
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
  (dropout): Dropout(p=0.4, inplace=False)
  (fc): Linear(in_features=1024, out_features=5, bias=True)
  (aux1): InceptionAux(
    (averagePool): AvgPool2d(kernel_size=5, stride=3, padding=0)
    (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))
    (line_1): Linear(in_features=2048, out_features=1024, bias=True)
    (line_2): Linear(in_features=1024, out_features=5, bias=True)
  )
  (aux2): InceptionAux(
    (averagePool): AvgPool2d(kernel_size=5, stride=3, padding=0)
    (conv): Conv2d(528, 128, kernel_size=(1, 1), stride=(1, 1))
    (line_1): Linear(in_features=2048, out_features=1024, bias=True)
    (line_2): Linear(in_features=1024, out_features=5, bias=True)
  )
)
```
![image](https://github.com/704494891/Classification_Net/blob/master/GoogleNet/ReadMe_images/net.png)


