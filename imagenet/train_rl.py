"""
Training file for HRL stage. Support Pytorch 3.0 and multiple GPUs.
"""

from __future__ import print_function

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F

import os
import shutil
import argparse
import time
import logging
import json
import itertools
import models
import sys
import pdb

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith('__')
                     and callable(models.__dict__[name]))


def parse_args():
    parser = argparse.ArgumentParser(
        description='PyTorch ImageNet training with gating')
    parser.add_argument('cmd', choices=['train', 'test'])
    parser.add_argument('arch', metavar='ARCH',
                        default='imagenet_rnn_gate_rl_34',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: imagenet_rnn_gate_rl_34)')
    parser.add_argument('--gate-type', default='rnn',
                        choices=['rnn'], help='gate type,only support RNN Gate')
    parser.add_argument('--data', '-d', default='/home/ubuntu/imagenet/',
                        type=str, help='path to the imagenet data')
    parser.add_argument('--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--epochs', default=40, type=int,
                        help='number of total epochs (default: 120)')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=256, type=int,
                        help='mini-batch size (default: 256)')
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum used in SGD')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', default=10, type=int,
                        help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str,
                        help='path to  latest checkpoint (default: None)')
    parser.add_argument('--pretrained', dest='pretrained',
                        action='store_true', help='use pretrained model')
    parser.add_argument('--save-folder', default='save_checkpoints',
                        type=str, help='folder to save the checkpoints')
    parser.add_argument('--crop-size', default=224, type=int,
                        help='cropping size of the input')
    parser.add_argument('--scale-size', default=256, type=int,
                        help='scaling size of the input')
    parser.add_argument('--step-ratio', default=0.1, type=float,
                        help='ratio for learning rate deduction')
    parser.add_argument('--alpha', default=0.01, type=float,
                        help='tuning hyper-parameter in the hybrid loss')
    parser.add_argument('--rl-weight', default=0.01, type=float,
                        help='scaling weight for rewards')
    parser.add_argument('--gamma', default=1, type=float,
                        help='discount factor')
    parser.add_argument('--restart', action='store_true', help='restart ckpt')
    parser.add_argument('--temp', type=float, default=0.05,
                        help='temperature for gate parameter initialization')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    more_config(args)

    logging.info('CMD: '+' '.join(sys.argv))
    if args.cmd == 'train':
        logging.info('start training {}'.format(args.arch))
        run_training(args)
    elif args.cmd == 'test':
        test_model(args)


def run_training(args):
    # create model
    model = models.__dict__[args.arch](args.pretrained)
    model = torch.nn.DataParallel(model).cuda()
    best_prec1 = 0

    if args.gate_type == 'rnn':
        logging.info('initialize rnn parameters')
        for name, param in model.named_parameters():
            if 'control' in name:
                if 'weight' in name:
                    nn.init.xavier_normal(param)
                elif 'bias' in name:
                    nn.init.constant(param, 0.0)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logging.info('=> loading checkpoint `{}`'.format(args.resume))
            checkpoint = torch.load(args.resume)
            if args.restart:
                args.start_epoch = checkpoint['epoch']
                best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            logging.info('=> loaded checkpoint `{}` (epoch: {})'.format(
                args.resume, checkpoint['epoch']))
        else:
            logging.info('=> no checkpoint found at `{}`'.format(args.resume))

    cudnn.benchmark = True

    # hotter
    if not args.restart:
        logging.info('temperature = {}'.format(args.temp))
        model.module.control.hotter(args.temp)

    # Data Loading Code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomSizedCrop(args.crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
        drop_last=True if args.gate_type == 'rnn' else False)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Scale(args.scale_size),
            transforms.CenterCrop(args.crop_size),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function
    criterion = nn.CrossEntropyLoss().cuda()
    # calculate loss for each input
    batch_criterion = BatchCrossEntropy().cuda()

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,
                                       model.parameters()), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # start training
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(args, optimizer, epoch)

        train(args, train_loader, model, criterion,
              batch_criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(args, val_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        checkpoint_path = os.path.join(args.save_path,
                                       'checkpoint_{:03d}.pth.tar'.format(
                                           epoch))
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=checkpoint_path)
        shutil.copyfile(checkpoint_path, os.path.join(
            args.save_path, 'checkpoint_latest.pth.tar'))


def repackage_hidden(h):
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def train(args, train_loader, model, criterion,
          batch_criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    skip_ratios = ListAverageMeter()
    total_rewards = AverageMeter()  # for RL only

    model.train()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda()
        target = target.cuda(async=True)
        input_var = Variable(input)
        target_var = Variable(target)

        output, masks, probs, hidden = model(input_var, target_var)

        # collect gate actions, inputs and targets on different GPUs.
        # a walk around to Pytorch's gather function
        actions = model.module.saved_actions
        rewards = {k: [] for k in range(len(actions))}
        dists = model.module.saved_dists
        inputs = model.module.saved_outputs
        targets = model.module.saved_targets

        skips = [mask.data.le(0.5).float().mean() for mask in masks]
        if skip_ratios.len != len(skips):
            skip_ratios.set_len(len(skips))

        # collect prediction loss for the last action.
        pred_losses = {}
        for idx in range(len(inputs)):
            # gather output and targets for each device
            pred_losses[idx] = batch_criterion(inputs[idx], targets[idx])

        # still use the whole batch to calculate classification loss
        loss = criterion(output, target_var)

        # collect rewards for each node
        normalized_skip_weight = args.alpha / (len(actions[0]))
        cum_rewards = {k: [] for k in range(len(actions))}
        for idx in range(len(actions)):
            for act in actions[idx]:
                rewards[idx].append((1 - act.squeeze().float())
                                    * normalized_skip_weight)
            R = - pred_losses[idx].squeeze() * normalized_skip_weight
            for r in rewards[idx][::-1]:
                R = r + args.gamma * R
                cum_rewards[idx].insert(0, R.view(-1, 1))

        # calculate losses
        rl_losses = []
        for idx in range(len(actions)):
            for idy in range(len(actions[0])):
                # actions[idx][idy].reinforce(args.rl_weight *
                #                             cum_rewards[idx][idy].data)
                _loss = (- dists[idx][idy].log_prob(actions[idx][idy])\
                        * (cum_rewards[idx][idy] * args.rl_weight))
                rl_losses.append(_loss.mean())

        # for idx in range(len(actions)):
        #     rl_losses += actions[idx]

        # back-propagate the hybrid loss
        optimizer.zero_grad()
        torch.autograd.backward(rl_losses + [loss])
        optimizer.step()

        if args.gate_type == 'rnn':
            # for memory efficiency
            hidden = repackage_hidden(hidden)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        total_rewards.update(torch.cat(list(itertools.chain.from_iterable(
            cum_rewards.values()))).mean().data[0],
                             input.size(0))
        total_gate_rewards = torch.cat(list(itertools.chain.from_iterable(
            rewards.values()))).sum().data[0]
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))
        skip_ratios.update(skips, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logging.info("Epoch: [{0}][{1}/{2}]\t"
                         "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                         "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                         "Loss {loss.val:.3f} ({loss.avg:.3f})\t"
                         "Total rewards {rewards.val:.3f} ({rewards.avg:.3f})\t"
                         "Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                         "Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t"
                .format(
                epoch, i, len(train_loader),
                batch_time=batch_time,
                data_time=data_time,
                loss=losses,
                rewards=total_rewards,
                top1=top1,
                top5=top5))
            logging.info('total gate rewards = {:.3f}'.format(
                total_gate_rewards))

            skip_summaries = []
            for idx in range(skip_ratios.len):
                # logging.info(
                #     "block {:03d} skipping = {:.3f}({:.3f})".format(
                #         idx,
                #         skip_ratios.val[idx],
                #         skip_ratios.avg[idx]))
                skip_summaries.append(1 - skip_ratios.avg[idx])
            cp = ((sum(skip_summaries) + 1) / (len(skip_summaries) + 1)) * 100
            logging.info('*** Computation Percentage: {:.3f} %'.format(cp))


def test_model(args):
    # create model
    model = models.__dict__[args.arch](args.pretrained)
    model = torch.nn.DataParallel(model).cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            logging.info('=> loading checkpoint `{}`'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            logging.info('=> loaded checkpoint `{}` (epoch: {})'.format(
                args.resume, checkpoint['epoch']
            ))
        else:
            logging.info('=> no checkpoint found at `{}`'.format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    t = transforms.Compose([
        transforms.Scale(args.scale_size),
        transforms.CenterCrop(args.crop_size),
        transforms.ToTensor(),
        normalize])
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, t),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    criterion = nn.CrossEntropyLoss().cuda()
    validate(args, val_loader, model, criterion, args.start_epoch)


def validate(args, val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    skip_ratios = ListAverageMeter()

    # switch to evaluation mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = Variable(input, volatile=True)
        target_var = Variable(target, volatile=True)

        # compute output
        output, masks, logprobs, hidden = model(input_var, target_var)

        skips = [mask.data.le(0.5).float().mean() for mask in masks]
        if skip_ratios.len != len(skips):
            skip_ratios.set_len(len(skips))
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))
        skip_ratios.update(skips, input.size(0))
        losses.update(loss.data[0], input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if args.gate_type == 'rnn':
            hidden = repackage_hidden(hidden)

        if i % args.print_freq == 0 or (i == (len(val_loader) - 1)):
            logging.info(
                'Test: Epoch[{0}][{1}/{2}]\t'
                'Time: {batch_time.val:.4f}({batch_time.avg:.4f})\t'
                'Loss: {loss.val:.3f}({loss.avg:.3f})\t'
                'Prec@1: {top1.val:.3f}({top1.avg:.3f})\t'
                'Prec@5: {top5.val:.3f}({top5.avg:.3f})\t'
                    .format(epoch, i, len(val_loader),
                            batch_time=batch_time,
                            loss=losses,
                            top1=top1,
                            top5=top5,
                )
            )

            skip_summaries = []
            for idx in range(skip_ratios.len):
                # logging.info(
                #     "block {:03d}  skipping = {:.3f}({:.3f})".format(
                #         idx,
                #         skip_ratios.val[idx],
                #         skip_ratios.avg[idx]))
                skip_summaries.append(1 - skip_ratios.avg[idx])
            cp = ((sum(skip_summaries) + 1) / (len(skip_summaries) + 1)) * 100
            logging.info('*** Computation Percentage: {:.3f} %'.format(cp))

    logging.info(' * Prec@1 {top1.avg:.3f}\tPrec@5 {top5.avg:.3f}\t'
                 'Loss {loss.avg:.3f}'
                 .format(top1=top1, top5=top5, loss=losses))

    skip_summaries = []
    for idx in range(skip_ratios.len):
        # logging.info(
        #     "block {:03d} skipping = {:.3f}".format(
        #         idx,
        #         skip_ratios.avg[idx]))
        skip_summaries.append(1 - skip_ratios.avg[idx])

    # always keep the first block
    cp = ((sum(skip_summaries) + 1) / (len(skip_summaries) + 1)) * 100
    logging.info('* Total Computation Percentage: {:.3f} %'.format(cp))

    return top1.avg


def more_config(args):
    """config save path and logging"""
    args.save_path = os.path.join(args.save_folder, args.arch)
    os.makedirs(args.save_path, exist_ok=True)

    args.logger_file = os.path.join(args.save_path,
                                    'log_{}.txt'.format(args.cmd))
    handlers = [logging.FileHandler(args.logger_file, mode='w'),
                logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO,
                        datefmt='%m-%d-%y %H:%M',
                        format='%(asctime)s:%(message)s',
                        handlers=handlers)

    # save training parameters to the folder
    save_args(args)


def save_args(args):
    """save training hyper-parameters"""
    args.args_file = args_file = os.path.join(args.save_path, 'train_args.json')
    with open(args_file, 'w') as f:
        args_dict = {
            k: v for k, v in args._get_kwargs()}
        json.dump(args_dict, f)


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


def adjust_learning_rate(args, optimizer, epoch):
    """Sets the learning rate to the initial lr
    decayed by 10 every 30 epochs"""
    lr = args.lr * (args.step_ratio ** (epoch // 30))
    logging.info('Epoch [{}] Learning rate: {}'.format(epoch, lr))
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


class BatchCrossEntropy(nn.Module):
    def __init__(self):
        super(BatchCrossEntropy, self).__init__()

    def forward(self, x, target):
        logp = F.log_softmax(x, dim=1)
        target = target.view(-1, 1)
        output = - logp.gather(1, target)
        return output


if __name__ == '__main__':
    main()
