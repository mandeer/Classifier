# -*- coding: utf-8 -*-
import sys
sys.path.append('./')

import argparse
import os
import time
import random
import torch
from torch.optim import lr_scheduler
from utils.tools import str2bool
from dataLoader import getDataLoader
import models


class Solver(object):
    def __init__(self, config, model, trainLoader, testLoader):
        self.config      = config
        self.trainLoader = trainLoader
        self.testLoader  = testLoader
        self.model       = model.to(self.config.device)
        if self.config.device == 'cuda':
            self.model = torch.nn.DataParallel(self.model)
        self.optimizer   = torch.optim.Adam(self.model.parameters(), weight_decay=1e-4)
        self.criterion   = torch.nn.CrossEntropyLoss()

    def test_epoch(self, epoch):
        self.model.eval()  # 验证模式
        class_correct = list(0. for i in range(self.config.num_classes))
        class_total   = list(0. for i in range(self.config.num_classes))
        accuracy      = list(0. for i in range(self.config.num_classes + 1))
        test_loss = 0.0
        with torch.no_grad():
            for ii, (datas, labels) in enumerate(self.testLoader):
                datas, labels = datas.to(self.config.device), labels.to(self.config.device)
                score = self.model(datas)
                test_loss += self.criterion(score, labels)
                _, predicted = torch.max(score.data, 1)
                c = (predicted == labels)
                for jj in range(labels.size()[0]):
                    label = labels[jj]
                    class_correct[label] += int(c[jj])
                    class_total[label] += 1

            if (epoch + 1) % 10 == 0 or epoch + 1 == self.config.n_epochs:
                correct = 0
                total = 0
                for kk in range(self.config.num_classes):
                    if class_total[kk] == 0:
                        accuracy[kk] = 0
                    else:
                        correct = correct + int(class_correct[kk])
                        total = total + class_total[kk]
                        accuracy[kk] = int(class_correct[kk]) / class_total[kk]
                accuracy[self.config.num_classes] = correct / total

            if (ii + 1) == len(self.testLoader):
                print('epoch=%d, test_loss: %.6f [%d/%d]'
                      % (epoch + 1, test_loss / (ii + 1), ii + 1, len(self.testLoader)), end=' | ')
                print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
                if epoch + 1 == self.config.n_epochs:
                    for jj in range(self.config.num_classes):
                        print('label=%d, correct=%d, total=%d, accuracy=%.3f' %
                              (jj, class_correct[jj], class_total[jj], accuracy[jj]))
                if (epoch + 1) % 10 == 0 or epoch + 1 == self.config.n_epochs:
                    print('correct=%d, total=%d, accuracy=%.3f' %
                          (correct, total, accuracy[self.config.num_classes]))
        return

    def train_epoch(self, epoch):
        self.model.train()  # 训练模式
        train_loss = 0.0
        for ii, (datas, labels) in enumerate(self.trainLoader):
            datas, labels = datas.to(self.config.device), labels.to(self.config.device)
            self.optimizer.zero_grad()
            score = self.model(datas)
            loss = self.criterion(score, labels)
            loss.backward()
            self.optimizer.step()
            train_loss += float(loss)

            if (ii + 1) % self.config.log_step == 0 or (ii + 1) == len(self.trainLoader):
                current_lr = self.optimizer.param_groups[0]['lr']
                print('epoch=%d, [%d/%d], loss=%.6f, lr=%.6f' % (epoch + 1, ii + 1, len(self.trainLoader),
                                                                 train_loss / (ii + 1), current_lr), end=' | ')
                print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    def train(self):
        self.test_epoch(epoch=-1)
        for epoch in range(self.config.n_epochs):
            self.train_epoch(epoch=epoch)
            self.test_epoch(epoch=epoch)
            if (epoch + 1) % self.config.n_epochs == 0:
                self.model.save(root=self.config.out_path,
                                name=self.config.model_name + '_cifar' + str(self.config.num_classes) + '.pth',
                                device=self.config.device)
        return

    def test(self):
        self.test_epoch(epoch=self.config.n_epochs)
        return


def main(config):
    # cuda
    if config.use_cuda:
        from torch.backends import cudnn
        cudnn.benchmark = True
    elif torch.cuda.is_available():
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    config.device = 'cuda' if config.use_cuda else 'cpu'

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

    # data
    if config.dataset == 'CIFAR10':
        config.data_path = './data/cifar10'
        config.num_classes = 10
    elif config.dataset == 'CIFAR100':
        config.data_path = './data/cifar100'
        config.num_classes = 100
    else:
        print('Only support CIFAR10 and CIFAR100!!')
        return
    trainLoader, testLoader = getDataLoader(config)
    print('train samples num: ', len(trainLoader), '  test samples num: ', len(testLoader))

    model = getattr(models, config.model)(num_classes=config.num_classes)
    # model = models.SENet_CIFAR(num_classes=config.num_classes)
    if config.pretrained != '':
        print('use pretrained model: ', config.pretrained)
        model.load(config.pretrained)
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
    parser.add_argument('--batch-size', type=int,      default=32)
    parser.add_argument('--n-workers',  type=int,      default=4)
    parser.add_argument('--out-path',   type=str,      default='./output')
    parser.add_argument('--seed',       type=int,      default=0,           help='random seed for all')
    parser.add_argument('--log-step',   type=int,      default=100)
    parser.add_argument('--use-cuda',   type=str2bool, default=True,        help='enables cuda')
    parser.add_argument('--dataset',    type=str,      default='CIFAR10',  help='CIFAR10 or CIFAR100')
    parser.add_argument('--mode',       type=str,      default='train',     help='train, test')
    parser.add_argument('--model',      type=str,      default='MobileNetV2', help='model')
    parser.add_argument('--pretrained', type=str,      default='')

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
