import torch.nn as nn
import math
from.BasicModule import BasicModule

class Conv_BN(nn.Module):
    def __init__(self, in_planes, opt_planes, kernel_size=1, stride=1, padding=0, bias=False):
        super(Conv_BN, self).__init__()
        self.conv = nn.Conv2d(in_planes, opt_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn   = nn.BatchNorm2d(opt_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class Conv_DW(nn.Module):
    def __init__(self, in_planes, opt_planes, stride=1):
        super(Conv_DW, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1   = nn.BatchNorm2d(in_planes)
        self.relu  = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_planes, opt_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2   = nn.BatchNorm2d(opt_planes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out


class MobileNet(BasicModule):
    def __init__(self, num_classes=10):
        super(MobileNet, self).__init__()
        self.model_name = 'mobilenet'

        self.model = nn.Sequential(
            Conv_BN(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            Conv_DW(32, 64, 1),
            Conv_DW(64, 128, 2),
            Conv_DW(128, 128, 1),
            Conv_DW(128, 256, 2),
            Conv_DW(256, 256, 1),
            Conv_DW(256, 512, 2),
            Conv_DW(512, 512, 1),
            Conv_DW(512, 512, 1),
            Conv_DW(512, 512, 1),
            nn.AvgPool2d(kernel_size=4, stride=1),
        )
        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.model(x)
        out = out.view(out.size(0), -1)
        x = self.fc(out)
        return x