import torch
import torch.nn as nn
from.BasicModule import BasicModule


class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,G,C/G,H,W] -> [N,C/G,G,H,W] -> [N,C,H,W]'''
        N,C,H,W = x.size()
        G = self.groups
        return x.view(N, G, C//G, H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)


class Conv_BN(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, groups=1, use_RELU=False):
        super(Conv_BN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
        )
        if use_RELU:
            self.conv.add_module('ReLU', nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


class BasicUnit(nn.Module):
    def __init__(self, in_size):
        super(BasicUnit, self).__init__()
        planes = in_size // 2
        self.banch2 = nn.Sequential(
            Conv_BN(planes, planes, kernel_size=1, stride=1, padding=0, use_RELU=True),
            Conv_BN(planes, planes, kernel_size=3, stride=1, padding=1, groups=planes, use_RELU=False),
            Conv_BN(planes, planes, kernel_size=1, stride=1, padding=0, use_RELU=True),
        )
        self.shuffle = ShuffleBlock(groups=planes)

    def forward(self, x):
        x1 = x[:, :(x.shape[1] // 2), :, :]
        x2 = x[:, (x.shape[1] // 2):, :, :]
        out = torch.cat((x1, self.banch2(x2)), 1)
        out = self.shuffle(out)
        return out


class DownSampleUnit(nn.Module):
    def __init__(self, in_size, out_size):
        super(DownSampleUnit, self).__init__()
        out_planes = in_size // 2
        self.banch1 = nn.Sequential(
            Conv_BN(in_size, in_size, kernel_size=3, stride=2, padding=1, groups=in_size, use_RELU=False),
            Conv_BN(in_size, out_planes, kernel_size=1, stride=1, padding=0, use_RELU=True),
        )

        self.banch2 = nn.Sequential(
            Conv_BN(in_size, out_planes, kernel_size=1, stride=1, padding=0, use_RELU=True),
            Conv_BN(out_planes, out_planes, kernel_size=3, stride=2, padding=1, groups=out_planes, use_RELU=False),
            Conv_BN(out_planes, out_planes, kernel_size=1, stride=1, padding=0, use_RELU=True),
        )
        self.shuffle = ShuffleBlock(groups=out_size)

    def forward(self, x):
        out = torch.cat((self.banch1(x), self.banch2(x)), 1)
        out = self.shuffle(out)
        return out


class ShuffleNetV2(nn.Module):
    def __init__(self, num_classes=1000):
        super(ShuffleNetV2, self).__init__()
        self.init_conv = nn.Sequential(
            Conv_BN(in_planes=3, out_planes=24, kernel_size=3, stride=2, padding=1, use_RELU=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.bneck = nn.Sequential(
            DownSampleUnit(24, 48),     # stage 2
            BasicUnit(48),
            BasicUnit(48),
            BasicUnit(48),
            DownSampleUnit(48, 96),     # stage 3
            BasicUnit(96),
            BasicUnit(96),
            BasicUnit(96),
            BasicUnit(96),
            BasicUnit(96),
            BasicUnit(96),
            BasicUnit(96),
            DownSampleUnit(96, 192),     # stage 4
            BasicUnit(192),
            BasicUnit(192),
            BasicUnit(192),
        )
        self.head = nn.Sequential(
            Conv_BN(in_planes=192, out_planes=1024, kernel_size=1, stride=1, padding=0, use_RELU=True),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Conv2d(1024, num_classes, 1, 1, 0, bias=True),
        )

    def forward(self, x):
        out = self.init_conv(x)
        out = self.bneck(out)
        out = self.head(out)

        out = out.view(out.size(0), -1)
        return out