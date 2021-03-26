# MobelNet
### MobelNet—V1牛逼在哪了？

  引入了Depthwise Separable Convolution
（Depthwise Separable Convolution，DSC）=Depthwise Convolution（DWC）+ Pointwise convolution（PWC）
,减少了大量的学习参数,
1. DWC 一个输入就对应一个卷积核，就是一层输入对应一层卷积核，导致:输入多少层=输出多少层

![image](https://github.com/704494891/Classification_Net/blob/master/MobelNet_V2/ReadMe_images/DW.png)

2. PWC 就跟简单了，就是1X1卷积核啦，通过卷积核的数量控制输出的层数，
我的理解哈，１Ｘ１的卷积就是全连接层，只不过可以用控制输出层数，增加维度

![image](https://github.com/704494891/Classification_Net/blob/master/MobelNet_V2/ReadMe_images/PW.png)
![image](https://github.com/704494891/Classification_Net/blob/master/MobelNet_V2/ReadMe_images/DW+PW.png)
3. 其中第一个卷积层为传统的卷积；前面的卷积层均有bn和relu，最后一个全连接层只有BN，无ReLU。
![image](https://github.com/704494891/Classification_Net/blob/master/MobelNet_V2/ReadMe_images/jiegou.png)

4. 引入了参数 α、β

    α 控制卷基层的输出层数比例
    β 控制图像分辨率比例
    

### MobelNet-V2咋好了?
1. 相比V1引入了残差结构(ResNet牛逼),而且是创新性的引入了 Inverted Residuals (倒残差结构)
![image](https://github.com/704494891/Classification_Net/blob/master/MobelNet_V2/ReadMe_images/Inverted-Residuals.png)
2. 激活函数用的ReLU6
3. 
![image](https://github.com/704494891/Classification_Net/blob/master/MobelNet_V2/ReadMe_images/V2-cancha.png)
4. 
![image](https://github.com/704494891/Classification_Net/blob/master/MobelNet_V2/ReadMe_images/all-jiegou-v2.png)






[参考文献1](https://ww111w.cnblogs.com/darkknightzh/p/9410540.html)
