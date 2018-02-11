# -*- coding: utf-8 -*-

import math
import torch.nn as nn
from .BasicModule import BasicModule


def makeMlpConv(in_channel, hid_channel, out_channel, kernel_size=3, stride=1, padding=1):
    assert isinstance(hid_channel, int) or len(hid_channel) == 2
    if isinstance(hid_channel, int):
        hid1 = hid_channel
        hid2 = hid_channel
    else:
        hid1 = hid_channel[0]
        hid2 = hid_channel[1]

    return nn.Sequential(
        nn.Conv2d(in_channel, hid1, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(hid1),
        nn.ReLU(inplace=True),
        nn.Conv2d(hid1, hid2, kernel_size=1, stride=1, padding=0),
        nn.BatchNorm2d(hid2),
        nn.ReLU(inplace=True),
        nn.Conv2d(hid2, out_channel, kernel_size=1, stride=1, padding=0),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True),
    )


class NIN(BasicModule):
    def __init__(self, n_class, init_weights=True):
        super(NIN, self).__init__()
        self.model_name = 'nin'
        self.n_class = n_class
        self.MlpConv1 = makeMlpConv(3, (192, 160), 96, kernel_size=5, stride=1, padding=2)
        self.MlpConv2 = makeMlpConv(96, 192, 192, kernel_size=5, stride=1, padding=2)
        self.MlpConv3 = makeMlpConv(192, 192, self.n_class, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.pool2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.pool3 = nn.AvgPool2d(kernel_size=8, stride=1, padding=0)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.MlpConv1(x)
        x = self.pool1(x)
        x = self.MlpConv2(x)
        x = self.pool2(x)
        x = self.MlpConv3(x)
        x = self.pool3(x)
        x = x.view(x.size(0), self.n_class)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
