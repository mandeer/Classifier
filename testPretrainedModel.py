# -*- coding: utf-8 -*-

import argparse
import time
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from utils.tools import str2bool
import models


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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def validate(val_loader, model, criterion, use_cuda):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)
        if use_cuda:
            input_var = input_var.cuda()
            target_var = target_var.cuda()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.cpu().data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg


def main(config):
    # cuda
    if config.use_cuda:
        from torch.backends import cudnn
        cudnn.benchmark = True
    elif torch.cuda.is_available():
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # model
    model = getattr(models, config.model)(depth=152)
    print(model)
    model.load(config.model_name)
    criterion = torch.nn.CrossEntropyLoss().cuda()
    if config.use_cuda:
        model.cuda()
        criterion = criterion.cuda()

    # data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    test_data = datasets.ImageFolder(config.data_path, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(config.image_size),
        transforms.ToTensor(),
        normalize]))
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=config.batch_size,
                                              shuffle=False, num_workers=config.n_workers, pin_memory=True)

    # test
    validate(test_loader, model, criterion, config.use_cuda)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--n-workers', type=int, default=4)
    parser.add_argument('--out-path', type=str, default='./output')

    parser.add_argument('--use-cuda', type=str2bool, default=True, help='enables cuda')

    parser.add_argument('--data-path', type=str, default='../data/imageNet2012/val')
    parser.add_argument('--dataset', type=str, default='ImageNet2012', help='ImageNet2012')
    parser.add_argument('--model', type=str, default='ResNet', help='model')
    parser.add_argument('--model-name', type=str,
                        default='./pretrained_models/models_pretrained/resnet152-b121ed2d.pth', help='model for test')

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

