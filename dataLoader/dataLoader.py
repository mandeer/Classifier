# -*- coding: utf-8 -*-

import torch
from torchvision import datasets
from torchvision import transforms


def getDataLoader(config):
    if config.dataset == 'CIFAR10' or config.dataset == 'CIFAR100':
        if config.mode == 'train':
            transform = transforms.Compose([
                transforms.Resize(config.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),      # This makes it into [0,1]
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(config.image_size),
                transforms.ToTensor(),  # This makes it into [0,1]
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
    else:
        transform = transforms.Compose([
            transforms.Resize(config.image_size),
            transforms.ToTensor(),  # This makes it into [0,1]
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    if config.dataset == 'MNIST':
        dataTrain = datasets.MNIST(root=config.data_path, train=True, download=False, transform=transform)
        dataTest = datasets.MNIST(root=config.data_path, train=False, download=False, transform=transform)
    elif config.dataset == 'CIFAR10':
        dataTrain = datasets.CIFAR10(root=config.data_path, train=True, download=False, transform=transform)
        dataTest = datasets.CIFAR10(root=config.data_path, train=False, download=False, transform=transform)
    elif config.dataset == 'CIFAR100':
        dataTrain = datasets.CIFAR100(root=config.data_path, train=True, download=False, transform=transform)
        dataTest = datasets.CIFAR100(root=config.data_path, train=False, download=False, transform=transform)

    trainLoader = torch.utils.data.DataLoader(dataset=dataTrain,
                                              batch_size=config.batch_size,
                                              shuffle=True,
                                              num_workers=config.n_workers,
                                              drop_last=True)
    testLoader = torch.utils.data.DataLoader(dataset=dataTest,
                                             batch_size=config.batch_size,
                                             shuffle=False,
                                             num_workers=config.n_workers,
                                             drop_last=True)
    return trainLoader, testLoader
