import torch
import torch.nn as nn
from.BasicModule import BasicModule
import math


class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N,C,H,W = x.size()
        g = self.groups
        return x.view(N,g,C/g,H,W).permute(0,2,1,3,4).contiguous().view(N,C,H,W)


class Bottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, stride, groups=1):
        super(Bottleneck, self).__init__()
        self.stride = stride

        mid_planes = out_planes // 4
        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=1, groups=groups, bias=False)
        self.bn1   = nn.BatchNorm2d(mid_planes)
        self.shuffle = ShuffleBlock(groups=groups)
        self.conv2 = nn.Conv2d(mid_planes, mid_planes, kernel_size=3, stride=stride, padding=1, groups=mid_planes, bias=False)
        self.bn2   = nn.BatchNorm2d(mid_planes)
        self.conv3 = nn.Conv2d(mid_planes, out_planes, kernel_size=1, groups=groups, bias=False)
        self.bn3   = nn.BatchNorm2d(out_planes)
        self.relu  = nn.ReLU(inplace=True)

        self.shortcut = None
        if stride == 2:
            self.shortcut = nn.Sequential(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(x)
        out = self.relu(x)
        out = self.shuffle(out)

        out = self.conv2(out)
        out = self.bn2(x)
        out = self.relu(x)

        out = self.conv3(out)
        out = self.bn3(x)

        if self.shortcut == None:
            out += x
            out = self.relu(x)
        else:
            res = self.shortcut(x)
            out = torch.cat([out,res], 1)
            out = self.relu(x)
        return out


class ShuffleNet(BasicModule):
    def __init__(self, num_classes=10):
        super(ShuffleNet, self).__init__()
        self.model_name = 'shufflenet'
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            Bottleneck( 32,  64, stride=2, groups=4),
            Bottleneck( 64,  64, stride=1, groups=4),
            Bottleneck( 64, 128, stride=2, groups=4),
            Bottleneck(128, 128, stride=1, groups=4),
            Bottleneck(128, 256, stride=2, groups=4),
            Bottleneck(256, 256, stride=1, groups=4),
            Bottleneck(256, 256, stride=1, groups=4),
            nn.AvgPool2d(kernel_size=4, stride=1),
        )
        self.fc = nn.Linear(256, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

