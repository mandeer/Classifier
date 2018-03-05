# -*- coding: utf-8 -*-

import argparse
import os
import random
import torch
from torch.autograd import Variable
from utils.tools import str2bool
from dataLoader.dataLoader import getDataLoader
import models

class Solver(object):
    def __init__(self, config, model, trainLoader, testLoader):
        self.trainLoader = trainLoader
        self.testLoader  = testLoader
        self.n_class     = config.n_class
        self.use_cuda    = config.use_cuda
        self.model_name  = config.model
        self.model       = model
        self.lr          = config.lr
        self.optimizer   = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion   = torch.nn.CrossEntropyLoss()
        if self.use_cuda:
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()

        self.n_epochs = config.n_epochs
        self.log_step = config.log_step
        self.out_path = config.out_path

    def val(self):
        model = self.model
        testLoader = self.testLoader
        model.eval()  # 验证模式
        class_correct = list(0. for i in range(self.n_class))
        class_total   = list(0. for i in range(self.n_class))
        accuracy      = list(0. for i in range(self.n_class + 1))
        loss = 0.0
        for ii, (datas, labels) in enumerate(testLoader):
            val_inputs = Variable(datas, volatile=True)
            target = Variable(labels)
            if self.use_cuda:
                val_inputs = val_inputs.cuda()
                target = target.cuda()
            # print(labels)
            score = model(val_inputs)
            loss += self.criterion(score, target)
            _, predicted = torch.max(score.data, 1)
            c = (predicted.cpu() == labels).squeeze()
            for jj in range(labels.size()[0]):
                label = labels[jj]
                class_correct[label] += c[jj]
                class_total[label] += 1

        correct = 0
        total = 0
        for ii in range(self.n_class):
            if class_total[ii] == 0:
                accuracy[ii] = 0
            else:
                correct = correct + class_correct[ii]
                total = total + class_total[ii]
                accuracy[ii] = class_correct[ii] / class_total[ii]
        accuracy[self.n_class] = correct / total

        model.train()  # 训练模式
        return accuracy, loss.cpu().data.numpy()

    def train(self):
        val_accuracy, val_loss = self.val()
        print('begin with accuracy: ', val_accuracy[self.n_class])

        model = self.model
        for epoch in range(self.n_epochs):
            for ii, (data, label) in enumerate(self.trainLoader):
                input  = Variable(data)
                target = Variable(label)
                if self.use_cuda:
                    input = input.cuda()
                    target = target.cuda()
                self.optimizer.zero_grad()
                score = model(input)
                loss = self.criterion(score, target)
                loss.backward()
                self.optimizer.step()

                if (ii + 1) % self.log_step == 0:
                    print('epoch: ', epoch + 1, 'train_num: ', ii + 1, loss.cpu().data.numpy()[0])

            val_accuracy, val_loss = self.val()
            print('val accuracy: ', val_accuracy[self.n_class])
            print('val loss:     ', val_loss[0])

            if (epoch + 1) % 10 == 0:
                model.save(root=self.out_path,
                           name=self.model_name + '_cifar' + str(self.n_class) + '_' + str(epoch+1) + '.pth')
        return

    def test(self):
        accuracy, loss = self.val()

        for jj in range(self.n_class):
            print('accuracy_', jj, ': ', accuracy[jj])
        print('accuracy total: ', accuracy[self.n_class])
        return


def main(config):
    # cuda
    if config.use_cuda:
        from torch.backends import cudnn
        cudnn.benchmark = True
    elif torch.cuda.is_available():
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # seed
    if config.seed == 0:
        config.seed = random.randint(1, 10000)  # fix seed
    print("Random Seed: ", config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if config.use_cuda:
        torch.cuda.manual_seed_all(config.seed)

    # create directories if not exist
    if not os.path.exists(config.out_path):
        os.makedirs(config.out_path)

    trainLoader, testLoader = getDataLoader(config)
    print('train samples num: ', len(trainLoader), '  test samples num: ', len(testLoader))

    model = getattr(models, config.model)(num_classes=config.n_class)
    if config.model_preTrained != '':
        print('use pretrained model: ', config.model_preTrained)
        model.load(config.model_preTrained)
    print(model)
    solver = Solver(config, model, trainLoader, testLoader)

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()
    else:
        print('error mode!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--image-size', type=int,      default=32)
    parser.add_argument('--n-epochs',   type=int,      default=50)
    parser.add_argument('--batch-size', type=int,      default=128)
    parser.add_argument('--n-workers',  type=int,      default=4)
    parser.add_argument('--lr',         type=float,    default=0.001)
    parser.add_argument('--out-path',   type=str,      default='./output')
    parser.add_argument('--seed',       type=int,      default=0,           help='random seed for all')
    parser.add_argument('--log-step',   type=int,      default=100)
    parser.add_argument('--use-cuda',   type=str2bool, default=True,        help='enables cuda')

    parser.add_argument('--data-path',  type=str,      default='./data/cifar100')
    parser.add_argument('--n-class',    type=int,      default=100, help='10, 100')
    parser.add_argument('--dataset',    type=str,      default='CIFAR100', help='CIFAR10 or CIFAR100')
    parser.add_argument('--mode',       type=str,      default='train', help='train, test')
    parser.add_argument('--model',      type=str,      default='VGG_CIFAR', help='model')
    parser.add_argument('--model-preTrained', type=str, default='', help='model for test or retrain')

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
