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


class InvertedResidual(nn.Module):
    def __init__(self, in_planes, opt_planes, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.use_res_connect = self.stride == 1 and in_planes == opt_planes
        self.hidden_planes = in_planes * expand_ratio
        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(in_planes, self.hidden_planes, 1, 1, 0, bias=False),
            nn.BatchNorm2d(in_planes * expand_ratio),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(self.hidden_planes, self.hidden_planes, 3, stride, 1, groups=self.hidden_planes, bias=False),
            nn.BatchNorm2d(self.hidden_planes),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(self.hidden_planes, opt_planes, 1, 1, 0, bias=False),
            nn.BatchNorm2d(opt_planes),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(BasicModule):
    def __init__(self, num_classes=10):
        super(MobileNetV2, self).__init__()
        self.model_name = 'mobilenet2'

        self.features = nn.Sequential(
            Conv_BN(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            InvertedResidual( 32,  16, 1, 2),
            InvertedResidual( 16,  32, 2, 6),
            InvertedResidual( 32,  32, 1, 6),
            InvertedResidual( 32,  64, 2, 6),
            InvertedResidual( 64,  64, 1, 6),
            InvertedResidual( 64,  64, 1, 6),
            InvertedResidual( 64, 128, 2, 6),
            InvertedResidual(128, 128, 1, 6),
            InvertedResidual(128, 128, 1, 6),
            InvertedResidual(128, 128, 1, 6),
            Conv_BN(128, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(kernel_size=4, stride=1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256, num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

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
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
