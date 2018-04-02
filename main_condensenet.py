import argparse
import os
import random
import time
import math
import torch
from torch.autograd import Variable
from utils.tools import str2bool
from dataLoader import getDataLoader
import models
from utils import convert_model, measure_model


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, epochs, ori_lr, batch=None, nBatch=None):
    T_total = epochs * nBatch
    T_cur = (epoch % epochs) * nBatch + batch
    lr = 0.5 * ori_lr * (1 + math.cos(math.pi * T_cur / T_total))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class Solver(object):
    def __init__(self, config, model, trainLoader, testLoader):
        self.trainLoader = trainLoader
        self.testLoader  = testLoader
        self.num_classes = config.num_classes
        self.use_cuda    = config.use_cuda
        self.model_name  = config.model
        self.model       = model
        self.n_epochs    = config.n_epochs
        self.progress    = 0.0
        self.ori_lr      = config.lr
        self.group_lasso_lambda = config.group_lasso_lambda
        self.criterion   = torch.nn.CrossEntropyLoss()
        self.optimizer   = torch.optim.SGD(model.parameters(), config.lr, momentum=config.momentum,
                                           weight_decay=config.weight_decay, nesterov=True)
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
        class_correct = list(0. for i in range(self.num_classes))
        class_total   = list(0. for i in range(self.num_classes))
        accuracy      = list(0. for i in range(self.num_classes + 1))
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
        for ii in range(self.num_classes):
            if class_total[ii] == 0:
                accuracy[ii] = 0
            else:
                correct = correct + class_correct[ii]
                total = total + class_total[ii]
                accuracy[ii] = class_correct[ii] / class_total[ii]
        accuracy[self.num_classes] = correct / total

        model.train()  # 训练模式
        return accuracy, loss.cpu().data.numpy()

    def train_epoch(self, epoch):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        learned_module_list = []

        ### Switch to train mode
        self.model.train()
        ### Find all learned convs to prepare for group lasso loss
        for m in self.model.modules():
            if m.__str__().startswith('LearnedGroupConv'):
                learned_module_list.append(m)
        running_lr = None

        end = time.time()
        for i, (data, label) in enumerate(self.trainLoader):
            progress = float(epoch * len(self.trainLoader) + i) / (self.n_epochs * len(self.trainLoader))
            self.progress = progress
            ### Adjust learning rate
            lr = adjust_learning_rate(self.optimizer, epoch, self.n_epochs, self.ori_lr,
                                      batch=i, nBatch=len(self.trainLoader))
            if running_lr is None:
                running_lr = lr

            ### Measure data loading time
            data_time.update(time.time() - end)

            input = Variable(data)
            target = Variable(label)
            if self.use_cuda:
                input = input.cuda()
                target = target.cuda()

            ### Compute output
            output = self.model(input, progress)
            loss = self.criterion(output, target)

            ### Add group lasso loss
            if self.group_lasso_lambda > 0:
                lasso_loss = 0
                for m in learned_module_list:
                    lasso_loss = lasso_loss + m.lasso_loss
                loss = loss + self.group_lasso_lambda * lasso_loss

            ### Measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.data[0], input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            ### Compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            ### Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f}\t'  # ({batch_time.avg:.3f}) '
                      'Data {data_time.val:.3f}\t'  # ({data_time.avg:.3f}) '
                      'Loss {loss.val:.4f}\t'  # ({loss.avg:.4f}) '
                      'Prec@1 {top1.val:.3f}\t'  # ({top1.avg:.3f}) '
                      'Prec@5 {top5.val:.3f}\t'  # ({top5.avg:.3f})'
                      'lr {lr: .4f}'.format(
                    epoch, i, len(self.trainLoader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5, lr=lr))
        return 100. - top1.avg, 100. - top5.avg, losses.avg, running_lr

    def train(self):
        val_accuracy, val_loss = self.val()
        print('begin with accuracy: ', val_accuracy[self.num_classes])

        for epoch in range(self.n_epochs):
            tr_prec1, tr_prec5, loss, lr = train_epoch(epoch)

            print('epoch: ', epoch + 1, 'tr_prec1: ', tr_prec1, tr_prec1, lr)
            if (epoch + 1) % 10 == 0:
                self.model.save(root=self.out_path,
                           name=self.model_name + '_cifar' + str(self.num_classes) + '_' + str(epoch + 1) + '.pth')
        return

    def test(self):
        accuracy, loss = self.val()

        for jj in range(self.num_classes):
            print('accuracy_', jj, ': ', accuracy[jj])
        print('accuracy total: ', accuracy[self.num_classes])
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
    config.stages = [14, 14, 14]
    config.growth = [8, 16, 32]

    trainLoader, testLoader = getDataLoader(config)
    print('train samples num: ', len(trainLoader), '  test samples num: ', len(testLoader))

    model = getattr(models, config.model)(config)
    ### Calculate FLOPs & Param
    n_flops, n_params = measure_model(model, config.image_size, config.image_size)
    print('FLOPs: %.2fM, Params: %.2fM' % (n_flops / 1e6, n_params / 1e6))
    del (model)
    model = getattr(models, config.model)(config)

    if config.pretrained != '':
        print('use pretrained model: ', config.pretrained)
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

    parser.add_argument('--image-size', type=int, default=32)
    parser.add_argument('--model', type=str, default='condensenet', help='model')
    parser.add_argument('--n-workers', type=int, default=4)
    parser.add_argument('--n-epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--lr-type', type=str, default='cosine')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', type=int, default=10, help='print frequency (default: 10)')
    parser.add_argument('--seed', type=int, default=0, help='random seed for all')
    parser.add_argument('--use-cuda', type=str2bool, default=True, help='enables cuda')
    parser.add_argument('--out-path', type=str, default='./output')
    parser.add_argument('--stages', type=str, metavar='STAGE DEPTH', help='per layer depth')
    parser.add_argument('--bottleneck', default=4, type=int, metavar='B', help='bottleneck (default: 4)')
    parser.add_argument('--group-1x1', type=int, metavar='G', default=4, help='1x1 group convolution (default: 4)')
    parser.add_argument('--group-3x3', type=int, metavar='G', default=4, help='3x3 group convolution (default: 4)')
    parser.add_argument('--condense-factor', type=int, metavar='C', default=4, help='condense factor (default: 4)')
    parser.add_argument('--growth', type=str, metavar='GROWTH RATE', help='per layer growth')
    parser.add_argument('--reduction', default=0.5, type=float, metavar='R', help='transition reduction (default: 0.5)')
    parser.add_argument('--dropout-rate', default=0, type=float, help='drop out (default: 0)')
    parser.add_argument('--group-lasso-lambda', default=0., type=float, metavar='LASSO', help='group lasso loss weight (default: 0)')
    parser.add_argument('--log-step', type=int, default=100)

    parser.add_argument('--pretrained', type=str, default='', help='model for test or retrain')
    parser.add_argument('--mode', type=str, default='train', help='train, test')
    parser.add_argument('--data', type=str, default='CIFAR100', help='CIFAR10 or CIFAR100')
    parser.add_argument('--dataset', type=str, default='CIFAR100', help='CIFAR10 or CIFAR100')

    config = parser.parse_args()
    if config.condense_factor is None:
        config.condense_factor = config.group_1x1
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