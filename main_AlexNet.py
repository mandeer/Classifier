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
        self.n_class = config.n_class
        self.use_cuda = config.use_cuda
        self.AlexNet = getattr(models, 'CIFAR10')(self.n_class)
        if config.model_name != '':
            print('use pretrained model: ', config.model_name)
            self.AlexNet.load(config.model_name)
        self.lr = config.lr
        self.optimizer = torch.optim.SGD(self.AlexNet.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.0005)
        self.criterion = torch.nn.CrossEntropyLoss()
        if self.use_cuda:
            self.AlexNet = self.AlexNet.cuda()
            self.criterion = self.criterion.cuda()

        self.n_epochs = config.n_epochs
        self.log_step = config.log_step
        self.out_path = config.out_path

    def val(self):
        AlexNet = self.AlexNet
        testLoader = self.testLoader
        AlexNet.eval()  # 验证模式
        class_correct = list(0. for i in range(self.n_class))
        class_total = list(0. for i in range(self.n_class))
        accuracy = list(0. for i in range(self.n_class + 1))
        loss = 0.0
        for ii, (datas, labels) in enumerate(testLoader):
            val_inputs = Variable(datas, volatile=True)
            target = Variable(labels)
            if self.use_cuda:
                val_inputs = val_inputs.cuda()
                target = target.cuda()
            # print(labels)
            score = AlexNet(val_inputs)
            loss += self.criterion(score, target)
            _, predicted = torch.max(score.data, 1)
            c = (predicted.cpu() == labels).squeeze()
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

        AlexNet.train()  # 训练模式
        return accuracy, loss.cpu().data.numpy()

    def train(self):
        val_accuracy, val_loss = self.val()
        print('begin with accuracy: ', val_accuracy[10])

        AlexNet = self.AlexNet
        for epoch in range(self.n_epochs):
            for ii, (data, label) in enumerate(self.trainLoader):
                input = Variable(data)
                target = Variable(label)
                if self.use_cuda:
                    input = input.cuda()
                    target = target.cuda()
                self.optimizer.zero_grad()
                score = AlexNet(input)
                loss = self.criterion(score, target)
                loss.backward()
                self.optimizer.step()

                if (ii + 1) % self.log_step == 0:
                    print('epoch: ', epoch, 'train_num: ', ii + 1, loss.cpu().data.numpy()[0])

            per_val_loss = val_loss
            val_accuracy, val_loss = self.val()
            print('val accuracy: ', val_accuracy[10])
            print('val loss:     ', val_loss[0])

            if per_val_loss <= val_loss:
                self.lr *= 0.1
                self.optimizer = torch.optim.SGD(self.AlexNet.parameters(), lr=self.lr, momentum=0.9,
                                                 weight_decay=0.0005)
                print('now lr is: ', self.lr)

        AlexNet.save(root=self.out_path, name='CIFAR10.pth')
        return

    def test(self):
        accuracy, loss = self.val()

        for jj in range(10):
            print('accuracy_', jj, ': ', accuracy[jj])
        print('accuracy total: ', accuracy[10])
        return


def main(config):
    if config.use_cuda:
        from torch.backends import cudnn
        cudnn.benchmark = True
    elif torch.cuda.is_available():
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # create directories if not exist
    if not os.path.exists(config.out_path):
        os.makedirs(config.out_path)

    trainLoader, testLoader = getDataLoader(config)
    print('train samples num: ', len(trainLoader), '  test samples num: ', len(testLoader))

    solver = Solver(config, trainLoader, testLoader)
    print(solver.AlexNet)

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()
    else:
        print('error mode!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--image-size', type=int, default=32)
    parser.add_argument('--n-epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--n-workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--out-path', type=str, default='./output')

    parser.add_argument('--log-step', type=int, default=100)
    parser.add_argument('--use-cuda', type=str2bool, default=True, help='enables cuda')

    parser.add_argument('--data-path', type=str, default='./data/cifar10')
    parser.add_argument('--n-class', type=int, default=10, help='10, 100, or 1000')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='CIFAR10 or CIFAR100')
    parser.add_argument('--mode', type=str, default='train', help='train, test')
    parser.add_argument('--model-name', type=str, default='', help='model for test or retrain')

    config = parser.parse_args()
    if config.use_cuda and not torch.cuda.is_available():
        config.use_cuda = False
        print("WARNING: You have no CUDA device")

    args = vars(config)
    print('------------ Options -------------')
    for key, value in sorted(args.items()):
        print('%16.16s: %16.16s' % (str(key), str(value)))
    print('-------------- End ----------------')

    main(config)
    print('End!!')
