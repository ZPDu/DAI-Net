# -*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import random
import time
import torch
import argparse
import torch.optim as optim
import torch.utils.data as data
import numpy as np
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure as ssim

from data.config import cfg
from layers.modules import MultiBoxLoss, EnhanceLoss
from data.widerface import WIDERDetection, detection_collate
from models.factory import build_net, basenet_factory
from models.enhancer import RetinexNet
from utils.DarkISP import Low_Illumination_Degrading
from PIL import Image

parser = argparse.ArgumentParser(
    description='DSFD face Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--batch_size',
                    default=4, type=int,
                    help='Batch size for training')
parser.add_argument('--model',
                    default='dark', type=str,
                    choices=['dark', 'vgg', 'resnet50', 'resnet101', 'resnet152'],
                    help='model for training')
parser.add_argument('--resume',
                    default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--num_workers',
                    default=0, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda',
                    default=True, type=bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate',
                    default=5e-4, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum',
                    default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay',
                    default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma',
                    default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--multigpu',
                    default=True, type=bool,
                    help='Use mutil Gpu training')
parser.add_argument('--save_folder',
                    default='weights/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--local_rank',
                    type=int,
                    help='local rank for dist')

args = parser.parse_args()
global local_rank
local_rank = args.local_rank

if 'LOCAL_RANK' not in os.environ:
    os.environ['LOCAL_RANK'] = str(args.local_rank)

if torch.cuda.is_available():
    if args.cuda:
        # torch.set_default_tensor_type('torch.cuda.FloatTensor')
        import torch.distributed as dist

        gpu_num = torch.cuda.device_count()
        if local_rank == 0:
            print('Using {} gpus'.format(gpu_num))
        rank = int(os.environ['RANK'])
        torch.cuda.set_device(rank % gpu_num)
        dist.init_process_group('nccl')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

save_folder = os.path.join(args.save_folder, args.model)
if not os.path.exists(save_folder):
    os.mkdir(save_folder)

train_dataset = WIDERDetection(cfg.FACE.TRAIN_FILE, mode='train')

val_dataset = WIDERDetection(cfg.FACE.VAL_FILE, mode='val')
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
train_loader = data.DataLoader(train_dataset, args.batch_size,
                               num_workers=args.num_workers,
                               collate_fn=detection_collate,
                               sampler=train_sampler,
                               pin_memory=True)
val_batchsize = args.batch_size
val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=True)
val_loader = data.DataLoader(val_dataset, val_batchsize,
                             num_workers=0,
                             collate_fn=detection_collate,
                             sampler=val_sampler,
                             pin_memory=True)


min_loss = np.inf


def train():
    per_epoch_size = len(train_dataset) // (args.batch_size * torch.cuda.device_count())
    start_epoch = 0
    iteration = 0
    step_index = 0

    basenet = basenet_factory(args.model)
    dsfd_net = build_net('train', cfg.NUM_CLASSES, args.model)
    net = dsfd_net
    net_enh = RetinexNet()
    net_enh.load_state_dict(torch.load(args.save_folder + 'decomp.pth'))

    if args.resume:
        if local_rank == 0:
            print('Resuming training, loading {}...'.format(args.resume))
        start_epoch = net.load_weights(args.resume)
        iteration = start_epoch * per_epoch_size
    else:
        base_weights = torch.load(args.save_folder + basenet)
        if local_rank == 0:
            print('Load base network {}'.format(args.save_folder + basenet))
        if args.model == 'vgg' or args.model == 'dark':
            net.vgg.load_state_dict(base_weights)
        else:
            net.resnet.load_state_dict(base_weights)


    if not args.resume:
        if local_rank == 0:
            print('Initializing weights...')
        net.extras.apply(net.weights_init)
        net.fpn_topdown.apply(net.weights_init)
        net.fpn_latlayer.apply(net.weights_init)
        net.fpn_fem.apply(net.weights_init)
        net.loc_pal1.apply(net.weights_init)
        net.conf_pal1.apply(net.weights_init)
        net.loc_pal2.apply(net.weights_init)
        net.conf_pal2.apply(net.weights_init)
        net.ref.apply(net.weights_init)

    # Scaling the lr
    lr = args.lr * np.round(np.sqrt(args.batch_size / 4 * torch.cuda.device_count()),4)
    param_group = []
    param_group += [{'params': dsfd_net.vgg.parameters(), 'lr': lr}]
    param_group += [{'params': dsfd_net.extras.parameters(), 'lr': lr}]
    param_group += [{'params': dsfd_net.fpn_topdown.parameters(), 'lr': lr}]
    param_group += [{'params': dsfd_net.fpn_latlayer.parameters(), 'lr': lr}]
    param_group += [{'params': dsfd_net.fpn_fem.parameters(), 'lr': lr}]
    param_group += [{'params': dsfd_net.loc_pal1.parameters(), 'lr': lr}]
    param_group += [{'params': dsfd_net.conf_pal1.parameters(), 'lr': lr}]
    param_group += [{'params': dsfd_net.loc_pal2.parameters(), 'lr': lr}]
    param_group += [{'params': dsfd_net.conf_pal2.parameters(), 'lr': lr}]
    param_group += [{'params': dsfd_net.ref.parameters(), 'lr': lr / 10.}]

    optimizer = optim.SGD(param_group, lr=lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)

    if args.cuda:
        if args.multigpu:
            net = torch.nn.parallel.DistributedDataParallel(net.cuda(), find_unused_parameters=True)
            net_enh = torch.nn.parallel.DistributedDataParallel(net_enh.cuda())
        # net = net.cuda()
        cudnn.benckmark = True

    criterion = MultiBoxLoss(cfg, args.cuda)
    criterion_enhance = EnhanceLoss()
    if local_rank == 0:
        print('Loading wider dataset...')
        print('Using the specified args:')
        print(args)

    for step in cfg.LR_STEPS:
        if iteration > step:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)
    net_enh.eval()
    net.train()
    corr_mat = None
    for epoch in range(start_epoch, cfg.EPOCHES):
        losses = 0

        for batch_idx, (images, targets, _) in enumerate(train_loader):
            images = Variable(images.cuda() / 255.)
            targetss = [Variable(ann.cuda(), requires_grad=False)
                        for ann in targets]
            img_dark = torch.empty(size=(images.shape[0], images.shape[1], images.shape[2], images.shape[3])).cuda()
            # Generation of degraded data and AET groundtruth
            for i in range(images.shape[0]):
                img_dark[i], _ = Low_Illumination_Degrading(images[i])

            if iteration in cfg.LR_STEPS:
                step_index += 1
                adjust_learning_rate(optimizer, args.gamma, step_index)

            t0 = time.time()
            R_dark_gt, I_dark = net_enh(img_dark)
            R_light_gt, I_light = net_enh(images)

            out, out2, loss_mutual = net(img_dark, images, I_dark.detach(), I_light.detach())
            R_dark, R_light, R_dark_2, R_light_2 = out2

            # backprop
            optimizer.zero_grad()

            loss_l_pa1l, loss_c_pal1 = criterion(out[:3], targetss)
            loss_l_pa12, loss_c_pal2 = criterion(out[3:], targetss)

            loss_enhance = criterion_enhance([R_dark, R_light, R_dark_2, R_light_2, I_dark.detach(), I_light.detach()], images, img_dark) * 0.1
            loss_enhance2 = F.l1_loss(R_dark, R_dark_gt.detach()) + F.l1_loss(R_light, R_light_gt.detach()) + (
                        1. - ssim(R_dark, R_dark_gt.detach())) + (1. - ssim(R_light, R_light_gt.detach()))

            loss = loss_l_pa1l + loss_c_pal1 + loss_l_pa12 + loss_c_pal2 + loss_enhance2 + loss_enhance + loss_mutual
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=35, norm_type=2)
            optimizer.step()
            t1 = time.time()
            losses += loss.item()

            if iteration % 100 == 0:
                tloss = losses / (batch_idx + 1)
                if local_rank == 0:
                    print('Timer: %.4f' % (t1 - t0))
                    print('epoch:' + repr(epoch) + ' || iter:' +
                          repr(iteration) + ' || Loss:%.4f' % (tloss))
                    print('->> pal1 conf loss:{:.4f} || pal1 loc loss:{:.4f}'.format(
                        loss_c_pal1.item(), loss_l_pa1l.item()))
                    print('->> pal2 conf loss:{:.4f} || pal2 loc loss:{:.4f}'.format(
                        loss_c_pal2.item(), loss_l_pa12.item()))
                    print('->>lr:{}'.format(optimizer.param_groups[0]['lr']))

            if iteration != 0 and iteration % 5000 == 0:
                if local_rank == 0:
                    print('Saving state, iter:', iteration)
                    file = 'dsfd_' + repr(iteration) + '.pth'
                    torch.save(dsfd_net.state_dict(),
                               os.path.join(save_folder, file))
            iteration += 1
        # if local_rank == 0:
        if (epoch + 1) >= 0:
            val(epoch, net, dsfd_net, net_enh, criterion)
        if iteration >= cfg.MAX_STEPS:
            break


def val(epoch, net, dsfd_net, net_enh, criterion):
    net.eval()
    step = 0
    losses = torch.tensor(0.).cuda()
    losses_enh = torch.tensor(0.).cuda()
    t1 = time.time()

    for batch_idx, (images, targets, img_paths) in enumerate(val_loader):
        if args.cuda:
            images = Variable(images.cuda() / 255.)
            targets = [Variable(ann.cuda(), volatile=True)
                       for ann in targets]
        else:
            images = Variable(images / 255.)
            targets = [Variable(ann, volatile=True) for ann in targets]
        img_dark = torch.stack([Low_Illumination_Degrading(images[i])[0] for i in range(images.shape[0])],
                               dim=0)
        out, R = net.module.test_forward(img_dark)

        loss_l_pa1l, loss_c_pal1 = criterion(out[:3], targets)
        loss_l_pa12, loss_c_pal2 = criterion(out[3:], targets)
        loss = loss_l_pa12 + loss_c_pal2

        losses += loss.item()
        step += 1
    dist.reduce(losses, 0, op=dist.ReduceOp.SUM)

    tloss = losses / step / torch.cuda.device_count()
    t2 = time.time()
    if local_rank == 0:
        print('Timer: %.4f' % (t2 - t1))
        print('test epoch:' + repr(epoch) + ' || Loss:%.4f' % (tloss))

    global min_loss
    if tloss < min_loss:
        if local_rank == 0:
            print('Saving best state,epoch', epoch)
            torch.save(dsfd_net.state_dict(), os.path.join(
                save_folder, 'dsfd.pth'))
        min_loss = tloss

    states = {
        'epoch': epoch,
        'weight': dsfd_net.state_dict(),
    }
    if local_rank == 0:
        torch.save(states, os.path.join(save_folder, 'dsfd_checkpoint.pth'))


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    # lr = args.lr * args.batch_size / 4 * torch.cuda.device_count() * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * gamma


if __name__ == '__main__':
    train()
