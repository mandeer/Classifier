# Classifier
使用PyTorch实现了经典的深度学习分类算法：  

* [**LeNet**](#lenet)
* [**AlexNet**](#alexnet)
    * [ReLU](#relu)
* [ZFNet](#zfnet)
* [**VGG**](#vgg)
* [**NIN**](#nin)
* [**GoogLeNet**](#googlenet)
    * [BatchNorm](#batchnorm)
* [**ResNet**](#resnet)
* [DenseNet](#densenet)
* [DiracNets](#diracnets)

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
### 主要创新点
* 局部感受野(local receptive fields):  
卷积层, 用于提取特征
* 权值共享(shared weights):  
因为目标可能出现在图像的任何位置，所以同一特征图中不同的节点需要在图像的不同位置执行相同的操作。
即，同一层的不同节点拥有相同的权重。该操作使提取的特征拥有了位移不变性，同时大大降低了参数的数量。
* 下采样(sub-sampling):  
pooling层，下采样可以有效的降低输出对尺度和形变的敏感性。
特征图的个数通常随着空间分辨率的降低而增加

#### 卷积
![conv](./imgs/conv.gif)
#### max pooling
![pooling](./imgs/pooling.gif)

### 本工程实现的LeNet与原始的LeNet-5略有区别
* 使用了ReLU而不是sigmoid函数
* S2中的每个特征图连接到了每个C3的特征图

### 训练结果
* 在mnist训练集进行了10次迭代，最终在测试集上的识别结果为 0.983
* 在cifar10训练集进行了40次迭代，最终测试集上的识别结果为 0.636
* sigmoid版LeNet在mnist训练集上迭代30次后，识别率达到了0.975

[返回顶部](#classifier)

------
## AlexNet
[AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
在2012年的ImageNet图像分类竞赛中，top-5错误率比上一年的冠军下降了十个百分点，
且远远超过当年的第二名。将沉寂多年的神经网络重新引入了大众的视野。
![AlexNet](./imgs/AlexNet.png)
### 主要创新点
* 非线性激活函数: [**ReLU**](#relu)
* 防止过拟合的方法: Dropout, Data augmentation
* 大数据训练: imageNet
* 高性能计算平台: GPU
* 重叠Pooling: kernel_size=3, stride=2
* 局部响应归一化(LRN, Local Response Normalization), 
[VGG](#vgg)说这个没什么用，可以使用更强大的[BatchNorm](#batchnorm)代替。

### 本工程实现的AlexNet与原始的AlexNet略有区别
* 没有使用LRN
* 没有使用group

### 训练结果
* 在cifar10数据集上没有达到论文中给出的结果, 50次epochs才达到0.706
* cifarNet在5个epochs后达到了0.632，最终在20次epochs后达到了0.784
![AlexNet_cifar10](./imgs/AlexNet_cifar10.png)
* 使用预训练好的[AlexNet](https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth)，
在imageNet2012验证集上测试结果为：top1 = 0.565， top5 = 0.791

[返回顶部](#classifier)

------
### ReLU
修正线性单元([ReLU](http://proceedings.mlr.press/v15/glorot11a/glorot11a.pdf), Rectified linear unit)
能够有效缓解梯度消失的问题，从而直接以监督的方式训练深度神经网络，无需依赖无监督的逐层预训练。  
![AvtFunc](./imgs/ActFunc.png)

#### 优点
* 收敛速度快:  
sigmoid和tanh的梯度在饱和区域非常平缓，接近于0，很容易造成梯度消失的问题，减缓收敛速度。
而ReLu激活函数的梯度为1，而且只有在x的右半轴才饱和，这样使梯度可以很好的反向传播中传递，
不但提高了网络的训练速度也避免了因为梯度消失导致的网络退化。  
另外，如果涉及到概率问题，如DBN, RNN, LSTM中的一些gate，就不能使用ReLU了，需要使用sigmoid, 
不然，概率表达就错了。
* 稀疏激活性:  
大脑同时被激活的神经元只有4%左右，激活稀疏性匹配了生物学的概念。
早期Bengio教授认为稀疏激活性是网络性能提升的原因之一，
但后来的研究发现稀疏性并非是性能提升的必要条件。
如[PReLU](https://arxiv.org/abs/1502.01852), [ELU](https://arxiv.org/abs/1511.07289)等。

#### 缺点
* 神经元死亡: 随着训练的推进，部分输入会落入硬饱和区，导致对应权重无法更新。
(这个问题可以用[Leaky ReLu](http://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf)解决)
* 输出偏移: 即输出均值恒大于零。(可以使用[BatchNorm](#batchnorm)进行改善)  
偏移现象和神经元死亡会共同影响网络的收敛性。

[返回顶部](#classifier)

------
## ZFNet
[ZFNet](https://arxiv.org/abs/1311.2901v3)
是2013年ILSVRC的冠军。其网络结构是在[AlexNet](#alexnet)上进行了微调：
![ZFNet](./imgs/ZFNet.png)
### 主要创新点
* 卷积网络的可视化技术: 反卷积(Deconvolution), 也被称作转置卷积(Transpose convolution)
* 依据可视化的结果，优化了[AlexNet](#alexnet):
    * 第一层卷积的kernel从11改成7; stride从4改称2
    * 去掉了[AlexNet](#alexnet)中的group

#### 卷积与转置卷积
![Conv2D](./imgs/Conv2D.gif) ![ConvTrans2D](./imgs/ConvTrans2D.gif)  
图片来自[这里](https://github.com/vdumoulin/conv_arithmetic)

[返回顶部](#classifier)

------
## VGG
[VGG](https://arxiv.org/abs/1409.1556)
在2014年ILSVRC挑战中获得了定位问题的第一和分类问题的第二(第一是[GoogLeNet](#googlenet))。
该模型可以很好的推广到其他数据集上，是最常用的base网络之一。
本工程实现了ABDE 4个网络及其添加了[BatchNorm](#batchnorm)的ABDE网络。

![VGG](./imgs/VGG.png)

### 主要创新点
* 具有小过滤器的深度网络优于具有较大过滤器的浅层网络
* 使用多种尺度的图像进行训练和测试
* deep, very deep, very very deep

### 模型测试
pytorch中给出的VGG模型在imageNet2012验证集上的测试结果

|VGG|use_BN|no_BN|
|---|---|---|
|VGG11|0.704(0.898)|0.690(0.886)|
|VGG13|0.716(0.904)|0.699(0.892)|
|VGG16|0.734(0.915)|0.716(0.904)|
|VGG19|0.742(0.918)|0.724(0.909)|

[返回顶部](#classifier)

------
## NIN
[NIN](https://arxiv.org/abs/1312.4400)
对cnn的结构进行了改进。其提出的1*1卷积和全局均值池化已经成为了后来网络设计的标准结构。

![NIN](./imgs/Mlpconv.png)
![NIN](./imgs/NIN.png)

### 主要创新点
* 使用Mlpconv替代卷积：
    * cnn的高层特征其实是低层特征通过某种运算的组合，作者换了个思路，在局部感受野中进行更加复杂的运算。
    * Mlpconv等价于1*1的卷积层。
    * 1*1卷积核可以起到一个跨通道聚合的作用。进一步可以起到降维（或者升维）的作用，起到减少参数的目的
     ([GoogLeNet](#googlenet))。
    * Mlpconv能够提取感受野中的非线性特征，增强了局部模型的表达能力。这样就可以使用均值池化来进行分类。
    
* 使用全局均值池化替代全链接层：
    * 大大降低了参数的数量：原来的cnn参数主要集中在全链接层，特别是与卷积层相连的第一个全链接层。
    * 强化了最后一层特征图与类别的关系：最后一层输出的特征图的空间平均值可以解释为相应类别的置信度。
    * 降低了overfitting：因为均值池化本身就是一种结构性的规则项，且没有参数需要优化。
    使用全链接层的cnn容易过拟合，且严重依赖dropout进行规则化。

### 本工程实现的NIN与原始的NIN略有区别
* 添加了[BatchNorm](#batchnorm)层， 若不添加则很难收敛。
* 去掉了dropout层

### 训练结果
* 在cifar10数据集上，迭代30次后达到了0.897
* 在cifar100数据集上，迭代30次后达到了0.665

[返回顶部](#classifier)

------
## GoogLeNet
GoogLeNet包括V1-V4共四个版本，本工程实现了V3版本。

------
* [Inception V1](https://arxiv.org/abs/1409.4842) 
: Going Deeper with Convolutions [2014.9] [top5: 6.67%]
![Inception module](./imgs/Inception_module.png)
![GoogLeNet](./imgs/GoogLeNet.png)
![architecture](./imgs/GoogLeNet_architecture.png)
### 主要创新点
* 提出了Inception， 在利用密集矩阵的高计算性能的基础上，保持了网络结构的稀疏性。
22层网络，参数却只有AlexNet的约1/12。
* 使用不同大小的卷积核提取不同大小感受野的特征，然后对不同尺度的特征进行拼接融合。
* 使用1x1卷积核来进行降维。
* 训练时为了避免梯度消失，增加了辅助分类器用于向前传导梯度（测试时可以忽略，V3中有了新的解释）。

[返回顶部](#classifier)

------
* [Inception V2](https://arxiv.org/abs/1502.03167)
: Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
 [2015.2] [top5: 4.8%]
 ![GoogLeNetV2](./imgs/GoogLeNetV2.png)
 ### 主要创新点
 * 提出了[**BatchNorm**](#batchnorm)
    * 提高SGD中的初始学习率： 加快了学习速度，而不会发生梯度弥散
    * 去掉了Dropout层： BN也可以降低过拟合，去掉dropout可以加快学习速度
    * 减小了L2正则化项： L2正则项也是为了降低过拟合，减小L2正则项可以提高准确率
    * 加快了学习率的衰退速度： 因为BN大大加快了学习速度
    * 去掉了LRN层： 都是Normalization
    * 更彻底地打乱训练样本： why??
    * 减少图像扭曲的使用： epoch数减少，需要多学习真实的数据
 ![BatchNorm](./imgs/BatchNorm.png)
 
 [返回顶部](#classifier)

------
* [Inception V3](https://arxiv.org/abs/1512.00567)
: Rethinking the Inception Architecture for Computer Vision [2015.12] [top5: 3.5%]  
![Inception-V3](./imgs/Inception-V3.png)

### 主要创新点
* 网络设计的通用原则
    * 避免表示瓶颈，尤其是在前面的网络：pooling后特征图变小了，会造成信息丢失。
    * 高维的特征更容易处理，在高维特征上训练更快，更容易收敛。
    * 空间聚合可以通过较低维度嵌入上完成，而不会在表示能力上造成多少损失：
    相邻的神经单元之间具有很强的相关性，信息有冗余。
    * 平衡好网络的宽度与深度
* 将大的卷积拆分成若干个小的卷积：降低计算量的同时增加了空间的多样性。
在每个卷积层后面添加激活函数会比不添加更好。
* 非对称卷积：n\*n的卷积核可以分解成1\*n和n\*1非的卷积核。
在中等大小的feature map中效果比较好。  
![Mini-network](./imgs/Mini-network.png)
* 优化辅助分类器：辅助分类器起到了正则化(??)的作用，而不是V1中提到的作用。
* 混合poolong：避免了表示瓶颈(representational bottleneck)  
![new-pooling](./imgs/new-pooling.png)
* 标签平滑(Label Smoothing): 对网络输出进行正则化。
* 低分辨率图像的识别；在相同计算量的前提下，
低分辨率的网络需要更长的时间去训练，但最终的结果与高分辨率网络的差别不大。

### 模型测试
pytorch中给出的Inception-V3模型在imageNet2012验证集上的测试结果为：
 
|top1|top5|
|---|---|
|77.560|93.694|

[返回顶部](#classifier)

------
* [Inception V4](https://arxiv.org/abs/1602.07261)
: Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning
 [2016.2] [top5: 3.08%]

[返回顶部](#classifier)

------
### BatchNorm
批规范化(Batch Normalization, Inception-V2)通过使深度神经网络训练过程中的每一层神经网络的
输入保持相同分布，解决了反向传播过程中梯度消失(0.9^100 = 2.66 * 10e-5)和
梯度爆炸(1.1^100 = 1.38*10e5)的问题。大大加快了训练的速度，并缓解了过拟合问题。

#### 优点
* 加快了训练速度
    * 解决了梯度消失和梯度爆炸问题
    * 可以在开始训练的时候使用较大的学习率
    * 可以弃用或少用Dropout
* 缓解了过拟合问题
    * 可以弃用或少用Dropout
    * 可以使用比较小的L2正则化项
    * 可以弃用LRN
* 降低了网络对初始化权重的敏感性
* 减小了优化算法参数对结果的影响
* 提高了模型的容纳能力

#### 缺点
* 计算代价：有文章称使用Batch Norm会带来30%的额外计算开销。
* 对浅层网络效果不理想
* RNN、LSTM上效果不理想
* GANs上效果不理想
* 不适合在线学习：mini-batch=1

#### 原因
* 正则化：使用SGD进行训练时，均值和偏差都是在一个Mini-batch上进行计算，
而不是整个数据样集。因此均值和偏差会有一些比较小的噪声。
* 归一化：将输入的特征进行归一化，改变了Cost function的形状，
使得每一次梯度下降都可以更快的接近函数的最小值点，从而加速了模型训练。
* “独立同分布”：Batch Norm限制了前层网络的参数更新对后面网络数值分布程度的影响，
从而使得输入后层的数值变得更加稳定。
* 权重伸缩不变性：避免了反向传播时因为权重过大或过小导致的梯度消失或梯度爆炸问题，
从而加速了神经网络的训练。
* 数据伸缩不变性：数据的伸缩变化不会影响到对该层的权重参数更新，
使得训练过程更加鲁棒，简化了对学习率的选择。

#### 其他规范化方式
参考了[详解深度学习中的Normalization，不只是BN](https://zhuanlan.zhihu.com/p/33173246)
* [Layer Normalization](https://arxiv.org/abs/1607.06450)
综合考虑一层所有维度的输入，计算该层的平均输入值和输入方差，
然后用同一个规范化操作来转换各个维度的输入。
    * LN针对单个训练样本进行，不依赖于其他数据，因此可以避免BN中受mini-batch数据分布影响的问题，
    可以用于小mini-batch场景、动态网络场景和RNN，特别是自然语言处理领域。
    * LN不需要保存mini-batch的均值和方差，节省了额外的存储空间。
* [Weight Normalization](https://arxiv.org/abs/1602.07868)
通过重写网络权重W的方式来进行正则化。
    * 不依赖mini-batch，适用于RNN、LSTM
    * 不基于mini-batch，引入的噪声少于BN
    * WN也不需要保存mini-batch的均值和方差，节省了额外的存储空间
    * WN没有规范化输入，因此需要特别注意网络参数的初始化
* [Cosine Normalization](https://arxiv.org/abs/1702.05870)
通过用余弦计算代替内积计算实现了规范化。
    * 使用余弦代替原来的点积，将数据规范化到[-1,1]
    * CN丢失了scale信息，可能导致表达能力的下降
* [Instance Normalization](https://arxiv.org/abs/1607.08022)
直接的对单幅图像进行的归一化操作，且没有scale和shift。
    * 在图片视频分类等特征提取网络中大多数情况下BN效果优于IN
    * 在超分辨率、生成式类任务中的网络IN优于BN，因为BN破坏了图像原本的对比度信息

[返回顶部](#classifier)

------
## ResNet
[ResNet](https://arxiv.org/abs/1512.03385)
解决了深层网络训练困难的问题，并在2015年ImageNet的classification、detection、
localization以及COCO的detection和segmentation上均斩获了第一名的成绩，
且获得了CVPR2016的best paper。
ResNet有152层，之后的[改进版](https://arxiv.org/abs/1603.05027)
甚至达到了1001层之多。  
![ResNet](./imgs/ResNet.png)
![ResBlock](./imgs/Res-block.png)

### 主要创新点
* 解决了网络退化问题：随着网络深度的增加，误差趋向于饱和，然后会随之上升。
* 残差网络，恒等映射：如果新添加的层是恒等映射，那么更深层的网络不应该比相应的浅层网络
具有更高的训练误差。
* 加快了训练速度：如果最优的方案等于(近似于)恒等映射，那么将一个残差逼近零
比使用一组堆叠的非线性层来拟合恒等映射要容易的多。

### 集成(ensemble)
![Res-ensemble](./imgs/Res-ensemble.png)  
[残差网络单元可以分解成右图的形式](https://arxiv.org/abs/1605.06431)。
从图中可以看出，残差网络其实是由多种路径组合的一个网络，
换句话说，残差网络其实是很多并行子网络的组合，残差网络其实相当于一个多人投票系统。
删除一个基本的残差单元，对最后的分类结果影响很小。

### 模型测试
pytorch中给出的ResNet模型在imageNet2012验证集上的测试结果为：

|ResNet|top1|top5|
|---|---|---|
|ResNet18|00|00|
|ResNet34|00|00|
|ResNet50|00|00|
|ResNet101|00|00|
|ResNet152|00|00|


[返回顶部](#classifier)

------
## DenseNet
[DenseNet](https://arxiv.org/abs/1608.06993)

[返回顶部](#classifier)

------
## DiracNets
[DiracNets](https://arxiv.org/abs/1706.00388)

[返回顶部](#classifier)
