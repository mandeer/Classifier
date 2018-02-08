from torch import nn
from.BasicModule import BasicModule

class LeNet(BasicModule):
    def __init__(self, inChannels, n_class, use_ReLU=True):
        super(LeNet, self).__init__()
        self.model_name = 'lenet'
        if use_ReLU:
            self.features = nn.Sequential(
                nn.Conv2d(inChannels, 6, 5),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(6, 16, 5),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)
            )
        else:
            self.features = nn.Sequential(
                nn.Conv2d(inChannels, 6, 5),
                nn.Sigmoid(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(6, 16, 5),
                nn.Sigmoid(),
                nn.MaxPool2d(2, 2)
            )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, n_class)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 16 * 5 * 5)
        x = self.classifier(x)
        return x

