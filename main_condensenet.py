# -*- coding: utf-8 -*-

import argparse
import os
import random
import torch
from torch.autograd import Variable
from utils.tools import str2bool
from dataLoader import getDataLoader
import models



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
    if config.pretrained != '':
        print('use pretrained model: ', config.pretrained)
        model.load(config.model_preTrained)
    print(model)
    # solver = Solver(config, model, trainLoader, testLoader)

    # if config.mode == 'train':
    #     solver.train()
    # elif config.mode == 'test':
    #     solver.test()
    # else:
    #     print('error mode!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--image-size', type=int,      default=32)
    parser.add_argument('--n-epochs',   type=int,      default=50)
    parser.add_argument('--batch-size', type=int,      default=128)
    parser.add_argument('--n-workers',  type=int,      default=4)

    parser.add_argument('--out-path',   type=str,      default='./output')
    parser.add_argument('--seed',       type=int,      default=0,           help='random seed for all')
    parser.add_argument('--log-step',   type=int,      default=100)
    parser.add_argument('--use-cuda',   type=str2bool, default=True,        help='enables cuda')

    parser.add_argument('--lr',         type=float,     default=0.1)
    parser.add_argument('--momentum',   type=float,     default=0.9)
    parser.add_argument('--w-decay',    type=float,     default=1e-4)

    parser.add_argument('--data-path',  type=str,       default='./data/cifar100')
    parser.add_argument('--n-class',    type=int,       default=100,        help='10, 100')
    parser.add_argument('--dataset',    type=str,       default='CIFAR100', help='CIFAR10 or CIFAR100')
    parser.add_argument('--mode',       type=str,       default='train',    help='train, test')
    parser.add_argument('--model',      type=str,       default='ShuffleNet', help='model')
    parser.add_argument('--pretrained', type=str,       default='',         help='model for test or retrain')

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