#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

from layers import *
from data.config import cfg


class FEM(nn.Module):
    """docstring for FEM"""

    def __init__(self, in_planes):
        super(FEM, self).__init__()
        inter_planes = in_planes // 3
        inter_planes1 = in_planes - 2 * inter_planes
        self.branch1 = nn.Conv2d(
            in_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3)

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_planes, inter_planes, kernel_size=3,
                      stride=1, padding=3, dilation=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_planes, inter_planes, kernel_size=3,
                      stride=1, padding=3, dilation=3)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_planes, inter_planes1, kernel_size=3,
                      stride=1, padding=3, dilation=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_planes1, inter_planes1, kernel_size=3,
                      stride=1, padding=3, dilation=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_planes1, inter_planes1, kernel_size=3,
                      stride=1, padding=3, dilation=3)
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x1, x2, x3), dim=1)
        out = F.relu(out, inplace=True)
        return out


class DSFD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, base, extras, fem, head1, head2, num_classes):
        super(DSFD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.vgg = nn.ModuleList(base)

        self.L2Normof1 = L2Norm(256, 10)
        self.L2Normof2 = L2Norm(512, 8)
        self.L2Normof3 = L2Norm(512, 5)

        self.extras = nn.ModuleList(extras)
        self.fpn_topdown = nn.ModuleList(fem[0])
        self.fpn_latlayer = nn.ModuleList(fem[1])

        self.fpn_fem = nn.ModuleList(fem[2])

        self.L2Normef1 = L2Norm(256, 10)
        self.L2Normef2 = L2Norm(512, 8)
        self.L2Normef3 = L2Norm(512, 5)

        self.loc_pal1 = nn.ModuleList(head1[0])
        self.conf_pal1 = nn.ModuleList(head1[1])

        self.loc_pal2 = nn.ModuleList(head2[0])
        self.conf_pal2 = nn.ModuleList(head2[1])

        if self.phase=='test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(cfg)

    def _upsample_prod(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') * y

    def forward(self, x):
        size = x.size()[2:]
        pal1_sources = list()
        pal2_sources = list()
        loc_pal1 = list()
        conf_pal1 = list()
        loc_pal2 = list()
        conf_pal2 = list()

        # apply vgg up to conv4_3 relu
        for k in range(16):
            x = self.vgg[k](x)
        of1 = x
        s = self.L2Normof1(of1)
        pal1_sources.append(s)
        # apply vgg up to fc7
        for k in range(16, 23):
            x = self.vgg[k](x)
        of2 = x
        s = self.L2Normof2(of2)
        pal1_sources.append(s)

        for k in range(23, 30):
            x = self.vgg[k](x)
        of3 = x
        s = self.L2Normof3(of3)
        pal1_sources.append(s)

        for k in range(30, len(self.vgg)):
            x = self.vgg[k](x)
        of4 = x
        pal1_sources.append(of4)
        # apply extra layers and cache source layer outputs

        for k in range(2):
            x = F.relu(self.extras[k](x), inplace=True)
        of5 = x
        pal1_sources.append(of5)
        for k in range(2, 4):
            x = F.relu(self.extras[k](x), inplace=True)
        of6 = x
        pal1_sources.append(of6)

        conv7 = F.relu(self.fpn_topdown[0](of6), inplace=True)

        x = F.relu(self.fpn_topdown[1](conv7), inplace=True)
        conv6 = F.relu(self._upsample_prod(
            x, self.fpn_latlayer[0](of5)), inplace=True)

        x = F.relu(self.fpn_topdown[2](conv6), inplace=True)
        convfc7_2 = F.relu(self._upsample_prod(
            x, self.fpn_latlayer[1](of4)), inplace=True)

        x = F.relu(self.fpn_topdown[3](convfc7_2), inplace=True)
        conv5 = F.relu(self._upsample_prod(
            x, self.fpn_latlayer[2](of3)), inplace=True)

        x = F.relu(self.fpn_topdown[4](conv5), inplace=True)
        conv4 = F.relu(self._upsample_prod(
            x, self.fpn_latlayer[3](of2)), inplace=True)

        x = F.relu(self.fpn_topdown[5](conv4), inplace=True)
        conv3 = F.relu(self._upsample_prod(
            x, self.fpn_latlayer[4](of1)), inplace=True)

        ef1 = self.fpn_fem[0](conv3)
        ef1 = self.L2Normef1(ef1)
        ef2 = self.fpn_fem[1](conv4)
        ef2 = self.L2Normef2(ef2)
        ef3 = self.fpn_fem[2](conv5)
        ef3 = self.L2Normef3(ef3)
        ef4 = self.fpn_fem[3](convfc7_2)
        ef5 = self.fpn_fem[4](conv6)
        ef6 = self.fpn_fem[5](conv7)

        pal2_sources = (ef1, ef2, ef3, ef4, ef5, ef6)
        for (x, l, c) in zip(pal1_sources, self.loc_pal1, self.conf_pal1):
            loc_pal1.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf_pal1.append(c(x).permute(0, 2, 3, 1).contiguous())

        for (x, l, c) in zip(pal2_sources, self.loc_pal2, self.conf_pal2):
            loc_pal2.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf_pal2.append(c(x).permute(0, 2, 3, 1).contiguous())

        features_maps = []
        for i in range(len(loc_pal1)):
            feat = []
            feat += [loc_pal1[i].size(1), loc_pal1[i].size(2)]
            features_maps += [feat]

        loc_pal1 = torch.cat([o.view(o.size(0), -1)
                              for o in loc_pal1], 1)
        conf_pal1 = torch.cat([o.view(o.size(0), -1)
                               for o in conf_pal1], 1)

        loc_pal2 = torch.cat([o.view(o.size(0), -1)
                              for o in loc_pal2], 1)
        conf_pal2 = torch.cat([o.view(o.size(0), -1)
                               for o in conf_pal2], 1)

        priorbox = PriorBox(size, features_maps, cfg, pal=1)
        self.priors_pal1 = Variable(priorbox.forward(), volatile=True)

        priorbox = PriorBox(size, features_maps, cfg, pal=2)
        self.priors_pal2 = Variable(priorbox.forward(), volatile=True)

        if self.phase == 'test':
            output = self.detect.forward(
                loc_pal2.view(loc_pal2.size(0), -1, 4),
                self.softmax(conf_pal2.view(conf_pal2.size(0), -1,
                                            self.num_classes)),                # conf preds
                self.priors_pal2.type(type(x.data))
            )

        else:
            output = (
                loc_pal1.view(loc_pal1.size(0), -1, 4),
                conf_pal1.view(conf_pal1.size(0), -1, self.num_classes),
                self.priors_pal1,
                loc_pal2.view(loc_pal2.size(0), -1, 4),
                conf_pal2.view(conf_pal2.size(0), -1, self.num_classes),
                self.priors_pal2)
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            mdata = torch.load(base_file,
                               map_location=lambda storage, loc: storage)
            weights = mdata['weight']
            epoch = mdata['epoch']
            self.load_state_dict(weights)
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')
        return epoch

    def xavier(self, param):
        init.xavier_uniform(param)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            self.xavier(m.weight.data)
            m.bias.data.zero_()

        if isinstance(m,nn.ConvTranspose2d):
            self.xavier(m.weight.data)
            if 'bias' in m.state_dict().keys():
                m.bias.data.zero_()

        if isinstance(m,nn.BatchNorm2d):
            m.weight.data[...] = 1
            m.bias.data.zero_()


vgg_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
           512, 512, 512, 'M']

extras_cfg = [256, 'S', 512, 128, 'S', 256]

fem_cfg = [256, 512, 512, 1024, 512, 256]


def fem_module(cfg):
    topdown_layers = []
    lat_layers = []
    fem_layers = []

    topdown_layers += [nn.Conv2d(cfg[-1], cfg[-1],
                                 kernel_size=1, stride=1, padding=0)]
    for k, v in enumerate(cfg):
        fem_layers += [FEM(v)]
        cur_channel = cfg[len(cfg) - 1 - k]
        if len(cfg) - 1 - k > 0:
            last_channel = cfg[len(cfg) - 2 - k]
            topdown_layers += [nn.Conv2d(cur_channel, last_channel,
                                         kernel_size=1, stride=1, padding=0)]
            lat_layers += [nn.Conv2d(last_channel, last_channel,
                                     kernel_size=1, stride=1, padding=0)]
    return (topdown_layers, lat_layers, fem_layers)


def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=3, dilation=3)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                                     kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers


def multibox(vgg, extra_layers, num_classes):
    loc_layers = []
    conf_layers = []
    vgg_source = [14, 21, 28, -2]

    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                                  num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels,
                                 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels,
                                  num_classes, kernel_size=3, padding=1)]
    return (loc_layers, conf_layers)


def build_net_vgg(phase, num_classes=2):
    base = vgg(vgg_cfg, 3)
    extras = add_extras(extras_cfg, 1024)
    head1 = multibox(base, extras, num_classes)
    head2 = multibox(base, extras, num_classes)
    fem = fem_module(fem_cfg)
    return DSFD(phase, base, extras, fem, head1, head2, num_classes)

if __name__ == '__main__':
    inputs = Variable(torch.randn(1, 3, 640, 640))
    net = build_net('train', 2)
    out = net(inputs)
