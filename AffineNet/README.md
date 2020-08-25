# 复现算法

BP网络
全部由全连接层组成
自己加了俩BN层
数据集就是用烂了的MINIST
网络结构如下：
```python
All_affine_Net(
  (layers1): Sequential(
    (0): Linear(in_features=784, out_features=300, bias=True)
    (1): BatchNorm1d(300, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (layers2): Sequential(
    (0): Linear(in_features=300, out_features=100, bias=True)
    (1): BatchNorm1d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (layers3): Sequential(
    (0): Linear(in_features=100, out_features=10, bias=True)
  )
)
```
![image](https://github.com/704494891/Classification_Net/blob/master/AffineNet/ReadMe_images/net.png)

