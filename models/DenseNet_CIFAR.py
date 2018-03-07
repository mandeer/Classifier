import math
import torch
import torch.nn as nn
from.BasicModule import BasicModule

class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1    = nn.BatchNorm2d(in_planes)
        self.conv1  = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2    = nn.BatchNorm2d(4*growth_rate)
        self.conv2  = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        self.relu   = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = torch.cat([out,x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn      = nn.BatchNorm2d(in_planes)
        self.relu    = nn.ReLU(inplace=True)
        self.conv    = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.bn(x)
        out = self.relu(out)
        out = self.conv(out)
        out = self.avgpool(out)
        return out


class DenseNet_CIFAR(BasicModule):
    def __init__(self, block=Bottleneck, nblocks=[6,12,24,16], growth_rate=12, reduction=0.5, num_classes=10):
        super(DenseNet_CIFAR, self).__init__()
        self.model_name = 'densenet_cifar'
        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = self._make_trans_layers(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = self._make_trans_layers(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = self._make_trans_layers(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3]*growth_rate

        self.bn     = nn.BatchNorm2d(num_planes)
        self.relu   = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(4, stride=1)
        self.linear = nn.Linear(num_planes, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)
    def _make_trans_layers(self, in_planes, out_planes):
        layers = []
        layers.append(Transition(in_planes, out_planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)

        out = self.bn(out)
        out = self.relu(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def DenseNet121():
    return DenseNet_CIFAR(Bottleneck, [6,12,24,16], growth_rate=32)

def DenseNet169():
    return DenseNet_CIFAR(Bottleneck, [6,12,32,32], growth_rate=32)

def DenseNet201():
    return DenseNet_CIFAR(Bottleneck, [6,12,48,32], growth_rate=32)

def DenseNet161():
    return DenseNet_CIFAR(Bottleneck, [6,12,36,24], growth_rate=48)

def densenet_cifar():
    return DenseNet_CIFAR(Bottleneck, [6,12,24,16], growth_rate=12)

