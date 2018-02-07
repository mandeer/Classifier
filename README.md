# Classifier

使用Pytorch实现了经典的分类算法
1. LeNet
2. AlexNet

## Prerequisites:
* anaconda
* pytorch
* torchvision
* visdom

## 1. LeNet [paper](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)
卷积神经网络的开山之作，麻雀虽小但五脏俱全。
![LeNet-5](./imgs/LeNet-5.png)
### 主要贡献
1. 局部感受野(local receptive fields)
2. 权值共享(shared weights)
3. 下采样(sub-sampling)

简化了网络参数并使网络具有一定程度的位移、尺度、形状不变性

### 本工程实现的LeNet与LeNet-5略有区别
1. 使用了ReLU而不是sigmoid函数
2. S2中的每个特征图连接到了每个C3的特征图

### 训练结果
1. 在mnist进行了10次迭代，最终在测试集上的识别结果为 0.983
2. 在cifar10进行了40次迭代，最终测试集上的识别结果为 0.636