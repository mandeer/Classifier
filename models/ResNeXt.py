import torch.nn as nn
import math
from.BasicModule import BasicModule


class Block(nn.Module):
    expansion = 2
    def __init__(self, in_planes, cardinality=32, bottleneck_width=4, stride=1):
        super(Block, self).__init__()
        group_width = cardinality * bottleneck_width
        self.conv1  = nn.Conv2d(in_planes, group_width, kernel_size=1, bias=False)
        self.bn1    = nn.BatchNorm2d(group_width)
        self.conv2  = nn.Conv2d(group_width, group_width, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2    = nn.BatchNorm2d(group_width)
        self.conv3  = nn.Conv2d(group_width, self.expansion*group_width, kernel_size=1, bias=False)
        self.bn3    = nn.BatchNorm2d(self.expansion*group_width)
        self.relu   = nn.ReLU(inplace=True)

        self.shortcut = None
        if stride != 1 or in_planes != self.expansion*group_width:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*group_width, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*group_width)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.shortcut == None:
            residual = x
        else:
            residual = self.shortcut(x)

        out += residual
        out = self.relu(out)
        return out


class ResNeXt(BasicModule):
    def __init__(self, num_blocks=None, cardinality=32, bottleneck_width=4, num_classes=10):
        super(ResNeXt, self).__init__()
        self.model_name = 'resnext'
        if num_blocks == None:
            num_blocks = [3, 3, 3]
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        self.in_planes = 64

        self.conv1  = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1    = nn.BatchNorm2d(64)
        self.relu   = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(num_blocks[0], 1)
        self.layer2 = self._make_layer(num_blocks[1], 2)
        self.layer3 = self._make_layer(num_blocks[2], 2)
        self.apool  = nn.AvgPool2d(8, stride=1)
        self.linear = nn.Linear(cardinality*bottleneck_width*8, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(Block(self.in_planes, self.cardinality, self.bottleneck_width, stride))
            self.in_planes = Block.expansion * self.cardinality * self.bottleneck_width
        self.bottleneck_width *= 2
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.apool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


