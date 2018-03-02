"""
Training file for training SkipNets for supervised pre-training stage
"""

from __future__ import print_function

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import os
import shutil
import argparse
import time
import logging

import models
from data import *


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith('__')
                     and callable(models.__dict__[name])
                     )


def parse_args():
    # hyper-parameters are from ResNet paper
    parser = argparse.ArgumentParser(
        description='PyTorch CIFAR10 training with gating')
    parser.add_argument('cmd', choices=['train', 'test'])
    parser.add_argument('arch', metavar='ARCH',
                        default='cifar10_feedforward_38',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: cifar10_feedforward_38)')
    parser.add_argument('--gate-type', type=str, default='ff',
                        choices=['ff', 'rnn'], help='gate type')
    parser.add_argument('--dataset', '-d', default='cifar10', type=str,
                        choices=['cifar10', 'cifar100'],
                        help='dataset type')
    parser.add_argument('--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4 )')
    parser.add_argument('--iters', default=64000, type=int,
                        help='number of total iterations (default: 64,000)')
    parser.add_argument('--start-iter', default=0, type=int,
                        help='manual iter number (useful on restarts)')
    parser.add_argument('--batch-size', default=128, type=int,
                        help='mini-batch size (default: 128)')
    parser.add_argument('--lr', default=0.1, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', default=10, type=int,
                        help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str,
                        help='path to  latest checkpoint (default: None)')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pretrained model')
    parser.add_argument('--step-ratio', default=0.1, type=float,
                        help='ratio for learning rate deduction')
    parser.add_argument('--warm-up', action='store_true',
                        help='for n = 18, the model needs to warm up for 400 '
                             'iterations')
    parser.add_argument('--save-folder', default='save_checkpoints',
                        type=str,
                        help='folder to save the checkpoints')
    parser.add_argument('--eval-every', default=1000, type=int,
                        help='evaluate model every (default: 1000) iterations')
    parser.add_argument('--verbose', action="store_true",
                        help='print layer skipping ratio at training')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    save_path = args.save_path = os.path.join(args.save_folder, args.arch)
    os.makedirs(save_path, exist_ok=True)

    # config logger file
    args.logger_file = os.path.join(save_path, 'log_{}.txt'.format(args.cmd))
    handlers = [logging.FileHandler(args.logger_file, mode='w'),
                logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO,
                        datefmt='%m-%d-%y %H:%M',
                        format='%(asctime)s:%(message)s',
                        handlers=handlers)

    if args.cmd == 'train':
        logging.info('start training {}'.format(args.arch))
        run_training(args)

    elif args.cmd == 'test':
        logging.info('start evaluating {} with checkpoints from {}'.format(
            args.arch, args.resume))
        test_model(args)


def run_training(args):
    # create model
    model = models.__dict__[args.arch](args.pretrained)
    model = torch.nn.DataParallel(model).cuda()

    best_prec1 = 0

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logging.info('=> loading checkpoint `{}`'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_iter = checkpoint['iter']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            logging.info('=> loaded checkpoint `{}` (iter: {})'.format(
                args.resume, checkpoint['iter']
            ))
        else:
            logging.info('=> no checkpoint found at `{}`'.format(args.resume))

    cudnn.benchmark = True

    train_loader = prepare_train_data(dataset=args.dataset,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=args.workers)
    test_loader = prepare_test_data(dataset=args.dataset,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=args.workers)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,
                                       model.parameters()),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    skip_ratios = ListAverageMeter()

    end = time.time()
    for i in range(args.start_iter, args.iters):
        model.train()
        adjust_learning_rate(args, optimizer, i)

        input, target = next(iter(train_loader))
        # measuring data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=False)
        input_var = Variable(input).cuda()
        target_var = Variable(target).cuda()

        # compute output
        output, masks, logprobs = model(input_var)

        # collect skip ratio of each layer
        skips = [mask.data.le(0.5).float().mean() for mask in masks]
        if skip_ratios.len != len(skips):
            skip_ratios.set_len(len(skips))

        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, = accuracy(output.data, target, topk=(1,))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        skip_ratios.update(skips, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # repackage hidden units for RNN Gate
        if args.gate_type == 'rnn':
            model.module.control.repackage_hidden()

        batch_time.update(time.time() - end)
        end = time.time()

        # print log
        if i % args.print_freq == 0 or i == (args.iters - 1):
            logging.info("Iter: [{0}/{1}]\t"
                         "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                         "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                         "Loss {loss.val:.3f} ({loss.avg:.3f})\t"
                         "Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t".format(
                            i,
                            args.iters,
                            batch_time=batch_time,
                            data_time=data_time,
                            loss=losses,
                            top1=top1)
            )
            for idx in range(skip_ratios.len):
                logging.info(
                    "{} layer skipping = {:.3f}({:.3f})".format(
                        idx,
                        skip_ratios.val[idx],
                        skip_ratios.avg[idx],
                    )
                )

        # evaluate every 1000 steps
        if (i % args.eval_every == 0 and i > 0) or (i == (args.iters-1)):
            prec1 = validate(args, test_loader, model, criterion)
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            checkpoint_path = os.path.join(args.save_path,
                                           'checkpoint_{:05d}.pth.tar'.format(
                                               i))
            save_checkpoint({
                'iter': i,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            },
                is_best, filename=checkpoint_path)
            shutil.copyfile(checkpoint_path, os.path.join(args.save_path,
                                                          'checkpoint_latest'
                                                          '.pth.tar'))


def validate(args, test_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    skip_ratios = ListAverageMeter()

    # switch to evaluation mode
    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(test_loader):
        target = target.cuda(async=True)
        input_var = Variable(input, volatile=True).cuda()
        target_var = Variable(target, volatile=True).cuda()
        # compute output
        output, masks, _ = model(input_var)
        skips = [mask.data.le(0.5).float().mean() for mask in masks]
        if skip_ratios.len != len(skips):
            skip_ratios.set_len(len(skips))
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, = accuracy(output.data, target, topk=(1,))
        top1.update(prec1[0], input.size(0))
        skip_ratios.update(skips, input.size(0))
        losses.update(loss.data[0], input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 or (i == (len(test_loader) - 1)):
            logging.info(
                'Test: [{}/{}]\t'
                'Time: {batch_time.val:.4f}({batch_time.avg:.4f})\t'
                'Loss: {loss.val:.3f}({loss.avg:.3f})\t'
                'Prec@1: {top1.val:.3f}({top1.avg:.3f})\t'.format(
                    i, len(test_loader), batch_time=batch_time,
                    loss=losses,
                    top1=top1
                )
            )
    logging.info(' * Prec@1 {top1.avg:.3f}, Loss {loss.avg:.3f}'.format(
        top1=top1, loss=losses))

    skip_summaries = []
    for idx in range(skip_ratios.len):
        # logging.info(
        #     "{} layer skipping = {:.3f}".format(
        #         idx,
        #         skip_ratios.avg[idx],
        #     )
        # )
        skip_summaries.append(1-skip_ratios.avg[idx])
    # compute `computational percentage`
    cp = ((sum(skip_summaries) + 1) / (len(skip_summaries) + 1)) * 100
    logging.info('*** Computation Percentage: {:.3f} %'.format(cp))

    return top1.avg


def test_model(args):
    # create model
    model = models.__dict__[args.arch](args.pretrained)
    model = torch.nn.DataParallel(model).cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            logging.info('=> loading checkpoint `{}`'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_iter = checkpoint['iter']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            logging.info('=> loaded checkpoint `{}` (iter: {})'.format(
                args.resume, checkpoint['iter']
            ))
        else:
            logging.info('=> no checkpoint found at `{}`'.format(args.resume))

    cudnn.benchmark = False
    test_loader = prepare_test_data(dataset=args.dataset,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=args.workers)
    criterion = nn.CrossEntropyLoss().cuda()

    validate(args, test_loader, model, criterion)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        save_path = os.path.dirname(filename)
        shutil.copyfile(filename, os.path.join(save_path,
                                               'model_best.pth.tar'))


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


class ListAverageMeter(object):
    """Computes and stores the average and current values of a list"""
    def __init__(self):
        self.len = 10000  # set up the maximum length
        self.reset()

    def reset(self):
        self.val = [0] * self.len
        self.avg = [0] * self.len
        self.sum = [0] * self.len
        self.count = 0

    def set_len(self, n):
        self.len = n
        self.reset()

    def update(self, vals, n=1):
        assert len(vals) == self.len, 'length of vals not equal to self.len'
        self.val = vals
        for i in range(self.len):
            self.sum[i] += self.val[i] * n
        self.count += n
        for i in range(self.len):
            self.avg[i] = self.sum[i] / self.count


def adjust_learning_rate(args, optimizer, _iter):
    """ divide lr by 10 at 32k and 48k """
    if args.warm_up and (_iter < 400):
        lr = 0.01
    elif 32000 <= _iter < 48000:
        lr = args.lr * (args.step_ratio ** 1)
    elif _iter >= 48000:
        lr = args.lr * (args.step_ratio ** 2)
    else:
        lr = args.lr

    if _iter % args.eval_every == 0:
        logging.info('Iter [{}] learning rate = {}'.format(_iter, lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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


if __name__ == '__main__':
    main()
