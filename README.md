# Classifier

使用Pytorch实现了经典的分类算法：  
* [**LeNet**](#lenet)
* [**AlexNet**](#alexnet)
* [VGG](#vgg)
* GoogLeNet  
* ResNet
* DenseNet
* SqueezeNet
* MobileNet

------
## Prerequisites:
* anaconda
* pytorch-0.3.0
* torchvision
* visdom

------
## LeNet
[LeNet](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)
是卷积神经网络的开山之作，麻雀虽小但五脏俱全。
![LeNet-5](./imgs/LeNet-5.png)
### 主要贡献
* 局部感受野(local receptive fields)
* 权值共享(shared weights)
* 下采样(sub-sampling)

简化了网络参数并使网络具有一定程度的位移、尺度、形状不变性

### 本工程实现的LeNet与原始的LeNet-5略有区别
* 使用了ReLU而不是sigmoid函数
* S2中的每个特征图连接到了每个C3的特征图

### 训练结果
* 在mnist训练集进行了10次迭代，最终在测试集上的识别结果为 0.983
* 在cifar10训练集进行了40次迭代，最终测试集上的识别结果为 0.636

------
## AlexNet
[AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
在2012年的ImageNet图像分类竞赛中，top-5错误率比上一年的冠军下降了十个百分点，
且远远超过当年的第二名。将沉寂多年的神经网络重新引入大众的视野。
![AlexNet](./imgs/AlexNet.png)
### 主要贡献
* 非线性激活函数: ReLU
* 防止过拟合的方法: Dropout, Data augmentation
* 大数据训练: imageNet
* 高性能计算平台: GPU
* 重叠Pooling: kernel_size=3, stride=2
* 局部相应归一化(LRN, Local Response Normalization), 
[VGG](https://arxiv.org/abs/1409.1556)说这个没什么用

### 本工程实现的AlexNet与原始的AlexNet略有区别
* 没有使用LRN
* 没有使用group

### 训练结果
* 在cifar10数据集上没有达到论文中给出的结果, 50次epochs才达到0.706
* cifarNet在5个epochs后达到了0.509，最终在20次epochs后达到了0.750  
![AlexNet_cifar10](./imgs/AlexNet_cifar10.png)

------
## VGG
[VGG](https://arxiv.org/abs/1409.1556)


[返回顶部](#classifier)
