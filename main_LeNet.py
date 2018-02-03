# -*- coding: utf-8 -*-

import argparse
import os
import torch
from torch.autograd import Variable

from utils.tools import str2bool
from dataLoader.dataLoader import getDataLoader
import models

class Solver(object):
    def __init__(self, config, trainLoader, testLoader):
        self.trainLoader = trainLoader
        self.testLoader = testLoader
        self.LeNet = getattr(models, 'LeNet')()
        self.optimizer = torch.optim.SGD(self.LeNet.parameters(), lr=config.lr, weight_decay=config.weightDecay)
        self.n_epochs = config.n_epochs
        self.logStep = config.logStep

    def test(self):
        model = self.LeNet
        testLoader = self.testLoader
        model.eval()  # 验证模式
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        accuracy = list(0. for i in range(10 + 1))
        for ii, (datas, labels) in enumerate(testLoader):
            val_inputs = Variable(datas, volatile=True)
            # print(labels)
            outputs = model(val_inputs)
            _, predicted = torch.max(outputs.data, 1)
            c = (predicted == labels).squeeze()
            for jj in range(labels.size()[0]):
                label = labels[jj]
                class_correct[label] += c[jj]
                class_total[label] += 1

        correct = 0
        total = 0
        for ii in range(10):
            if class_total[ii] == 0:
                accuracy[ii] = 0
            else:
                correct = correct + class_correct[ii]
                total = total + class_total[ii]
                accuracy[ii] = class_correct[ii] / class_total[ii]
        accuracy[10] = correct / total

        model.train()  # 训练模式
        return accuracy

    def train(self):
        LeNet = self.LeNet
        criterion = torch.nn.CrossEntropyLoss()
        for epoch in range(self.n_epochs):
            for ii, (data, label) in enumerate(self.trainLoader):
                input = Variable(data)
                target = Variable(label)
                self.optimizer.zero_grad()

                score = LeNet(input)
                loss = criterion(score, target)
                loss.backward()
                self.optimizer.step()

                if (ii + 1) % self.logStep == 0:
                    print('epoch: ', epoch, 'train_num: ', ii + 1, loss.data.numpy())

            val_accuracy = self.test()
            # for jj in range(10):
            #     print('accuracy_', jj, ': ', val_accuracy[jj])
            print('accuracy total: ', val_accuracy[10])


def main(config):
    if config.cuda:
        from torch.backends import cudnn
        cudnn.benchmark = True
    elif torch.cuda.is_available():
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # create directories if not exist
    if not os.path.exists(config.outPath):
        os.makedirs(config.outPath)

    trainLoader, testLoader = getDataLoader(config)
    print('train samples num: ', len(trainLoader), '  test samples num: ', len(testLoader))

    solver = Solver(config, trainLoader, testLoader)
    print(solver.LeNet)

    solver.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model hyper-parameters
    parser.add_argument('--imageSize', type=int, default=32)

    # training hyper-parameters
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--batchSize', type=int, default=64)
    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weightDecay', type=float, default=1e-4, help='')

    # other
    parser.add_argument('--outPath', type=str, default='./output')
    parser.add_argument('--mnistPath', type=str, default='./data/mnist')
    parser.add_argument('--logStep', type=int, default=100)
    parser.add_argument('--cuda', type=str2bool, default=True, help='enables cuda')

    config = parser.parse_args()
    if config.cuda and not torch.cuda.is_available():
        config.cuda = False
        print("WARNING: You have no CUDA device")

    args = vars(config)
    print('------------ Options -------------')
    for key, value in sorted(args.items()):
        print('%16.16s: %16.16s' % (str(key), str(value)))
    print('-------------- End ----------------')

    main(config)
    print('End!!')
