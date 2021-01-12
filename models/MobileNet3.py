import torch
import torch.nn as nn
import torch.nn.functional as F
from .BasicModule import BasicModule


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        out = F.relu6(x + 3., self.inplace) / 6.
        return out * x


ACT_FNS = {
    'RE': nn.ReLU6(inplace=True),
    'HS': h_swish(),
    'HG': h_sigmoid()
}


class Conv_BN(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, groups=1, act='RE'):
        super(Conv_BN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
        )
        if act in ACT_FNS:
            self.conv.add_module('ACT', ACT_FNS[act])

    def forward(self, x):
        return self.conv(x)


class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv_BN(in_planes=in_size, out_planes=in_size//reduction, kernel_size=1, stride=1, padding=0, act='RE'),
            Conv_BN(in_planes=in_size//reduction, out_planes=in_size, kernel_size=1, stride=1, padding=0, act='HG'),
        )

    def forward(self, x):
        return x * self.se(x)


class MobileBlock(nn.Module):
    def __init__(self, in_size, expand_size, out_size, kernel_size, stride, act, use_SE):
        super(MobileBlock, self).__init__()
        self.use_connect = stride == 1 and in_size == out_size
        self.use_SE = use_SE
        if self.use_SE:
            self.se = SeModule(expand_size)
        self.conv1 = Conv_BN(in_planes=in_size, out_planes=expand_size, kernel_size=1, stride=1, padding=0, act=act)
        self.conv2 = Conv_BN(in_planes=expand_size, out_planes=expand_size, kernel_size=kernel_size, stride=stride,
                             padding=kernel_size//2, groups=expand_size, act=act)
        self.conv3 = Conv_BN(in_planes=expand_size, out_planes=out_size, kernel_size=1, stride=1, padding=0, act='none')

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.use_SE:
            out = self.se(out)
        out = self.conv3(out)

        if self.use_connect:
            return x + out
        else:
            return out


class MobileNetV3(BasicModule):
    def __init__(self, num_classes=1000):
        super(MobileNetV3, self).__init__()
        self.init_conv = Conv_BN(in_planes=3, out_planes=16, kernel_size=3, stride=2, padding=1, act='HS')
        self.bneck = nn.Sequential(
            MobileBlock(16, 16, 16, kernel_size=3, stride=2, act='RE', use_SE=True),
            MobileBlock(16, 72, 24, kernel_size=3, stride=2, act='RE', use_SE=False),
            MobileBlock(24, 88, 24, kernel_size=3, stride=1, act='RE', use_SE=False),
            MobileBlock(24, 96, 40, kernel_size=5, stride=2, act='HS', use_SE=True),
            MobileBlock(40, 240, 40, kernel_size=5, stride=1, act='HS', use_SE=True),
            MobileBlock(40, 240, 40, kernel_size=5, stride=1, act='HS', use_SE=True),
            MobileBlock(40, 120, 48, kernel_size=5, stride=1, act='HS', use_SE=True),
            MobileBlock(48, 144, 48, kernel_size=5, stride=1, act='HS', use_SE=True),
            MobileBlock(48, 288, 96, kernel_size=5, stride=2, act='HS', use_SE=True),
            MobileBlock(96, 576, 96, kernel_size=5, stride=1, act='HS', use_SE=True),
            MobileBlock(96, 576, 96, kernel_size=5, stride=1, act='HS', use_SE=True),
        )
        self.head = nn.Sequential(
            Conv_BN(in_planes=96, out_planes=576, kernel_size=1, stride=1, padding=1, act='HS'),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Conv2d(576, 1280, 1, 1, 0, bias=True),
            h_swish(),
            nn.Dropout(),
            nn.Conv2d(1280, num_classes, 1, 1, 0, bias=True),
        )

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.init_conv(x)
        out = self.bneck(out)
        out = self.head(out)

        out = out.view(out.size(0), -1)
        return out
