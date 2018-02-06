# -*- coding: utf-8 -*-

import torch
from torchvision import datasets
from torchvision import transforms


def getDataLoader(config):
    transform = transforms.Compose([
        transforms.Resize(config.imageSize),
        transforms.ToTensor(),      # This makes it into [0,1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    if config.dataset == 'MNIST':
        dataTrain = datasets.MNIST(root=config.dataPath, train=True, download=True, transform=transform)
        dataTest = datasets.MNIST(root=config.dataPath, train=False, download=True, transform=transform)
    elif config.dataset == 'CIFAR10':
        dataTrain = datasets.CIFAR10(root=config.dataPath, train=True, download=True, transform=transform)
        dataTest = datasets.CIFAR10(root=config.dataPath, train=False, download=True, transform=transform)

    trainLoader = torch.utils.data.DataLoader(dataset=dataTrain,
                                              batch_size=config.batchSize,
                                              shuffle=True,
                                              num_workers=config.n_workers,
                                              drop_last=True)
    testLoader = torch.utils.data.DataLoader(dataset=dataTest,
                                             batch_size=config.batchSize,
                                             shuffle=False,
                                             num_workers=config.n_workers,
                                             drop_last=True)
    return trainLoader, testLoader
