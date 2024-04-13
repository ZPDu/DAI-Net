# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

from layers import *
from data.config import cfg


def conv3x3(in_planes, out_planes, stride=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        sources = []

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        sources += [x]

        x = self.layer2(x)
        sources += [x]

        x = self.layer3(x)
        sources += [x]

        x = self.layer4(x)
        sources += [x]

        return sources


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
    """docstring for SRN"""

    def __init__(self, phase, base, extras, fem_modules, head1, head2, num_classes=2):
        super(DSFD, self).__init__()
        self.resnet = base
        self.phase = phase
        self.num_classes = num_classes
        self.extras = nn.ModuleList(extras)

        self.fpn_topdown = nn.ModuleList(fem_modules[0])
        self.fpn_latlayer = nn.ModuleList(fem_modules[1])
        self.fpn_fem = nn.ModuleList(fem_modules[2])

        self.loc_pal1 = nn.ModuleList(head1[0])
        self.conf_pal1 = nn.ModuleList(head1[1])
        self.loc_pal2 = nn.ModuleList(head2[0])
        self.conf_pal2 = nn.ModuleList(head2[1])

        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(cfg)

    def _upsample_prod(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') * y

    def forward(self, x):
        size = x.size()[2:]
        of1, of2, of3, of4 = self.resnet(x)

        x = of4
        for i in range(2):
            x = F.relu(self.extras[i](x), inplace=True)
        of5 = x

        for i in range(2, len(self.extras)):
            x = F.relu(self.extras[i](x), inplace=True)
        of6 = x

        conv7 = F.relu(self.fpn_topdown[0](of6), inplace=True)

        x = F.relu(self.fpn_topdown[1](conv7), inplace=True)
        conv6 = F.relu(self._upsample_prod(
            x, self.fpn_latlayer[0](of5)), inplace=True)

        x = F.relu(self.fpn_topdown[2](conv6), inplace=True)
        conv5 = F.relu(self._upsample_prod(
            x, self.fpn_latlayer[1](of4)), inplace=True)

        x = F.relu(self.fpn_topdown[3](conv5), inplace=True)
        conv4 = F.relu(self._upsample_prod(
            x, self.fpn_latlayer[2](of3)), inplace=True)

        x = F.relu(self.fpn_topdown[4](conv4), inplace=True)
        conv3 = F.relu(self._upsample_prod(
            x, self.fpn_latlayer[3](of2)), inplace=True)

        x = F.relu(self.fpn_topdown[5](conv3), inplace=True)
        conv2 = F.relu(self._upsample_prod(
            x, self.fpn_latlayer[4](of1)), inplace=True)

        ef1 = self.fpn_fem[0](conv2)
        ef2 = self.fpn_fem[1](conv3)
        ef3 = self.fpn_fem[2](conv4)
        ef4 = self.fpn_fem[3](conv5)
        ef5 = self.fpn_fem[4](conv6)
        ef6 = self.fpn_fem[5](conv7)

        sources_pal1 = [of1, of2, of3, of4, of5, of6]
        sources_pal2 = [ef1, ef2, ef3, ef4, ef5, ef6]
        loc_pal1, conf_pal1 = list(), list()
        loc_pal2, conf_pal2 = list(), list()

        for (x, l, c) in zip(sources_pal1, self.loc_pal1, self.conf_pal1):
            loc_pal1.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf_pal1.append(c(x).permute(0, 2, 3, 1).contiguous())

        for (x, l, c) in zip(sources_pal2, self.loc_pal2, self.conf_pal2):
            loc_pal2.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf_pal2.append(c(x).permute(0, 2, 3, 1).contiguous())

        features_maps = []
        for i in range(len(loc_pal1)):
            feat = []
            feat += [loc_pal1[i].size(1), loc_pal1[i].size(2)]
            features_maps += [feat]

        loc_pal1 = torch.cat([o.view(o.size(0), -1) for o in loc_pal1], 1)
        conf_pal1 = torch.cat([o.view(o.size(0), -1) for o in conf_pal1], 1)

        loc_pal2 = torch.cat([o.view(o.size(0), -1) for o in loc_pal2], 1)
        conf_pal2 = torch.cat([o.view(o.size(0), -1) for o in conf_pal2], 1)

        priorbox = PriorBox(size, features_maps, cfg, pal=1)
        self.priors_pal1 = Variable(priorbox.forward(), volatile=True)

        priorbox = PriorBox(size, features_maps, cfg, pal=2)
        self.priors_pal2 = Variable(priorbox.forward(), volatile=True)

        if self.phase == 'test':
            output = self.detect(
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

        if isinstance(m, nn.ConvTranspose2d):
            self.xavier(m.weight.data)
            if 'bias' in m.state_dict().keys():
                m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data[...] = 1
            m.bias.data.zero_()

extras_cfg = [256, 'S', 512, 128, 'S', 256]

net_cfg = [256, 512, 1024, 2048, 512, 256]


def add_extras(cfg, i):
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


def multibox(cfg, num_classes=2):
    conf_layers = []
    loc_layers = []
    for k, v in enumerate(cfg):
        loc_layers += [nn.Conv2d(v, 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v, num_classes, kernel_size=3, padding=1)]
    return (loc_layers, conf_layers)


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


def resnet50():
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    return model


def resnet101():
    model = ResNet(Bottleneck, [3, 4, 23, 3])
    return model


def resnet152():
    model = ResNet(Bottleneck, [3, 8, 36, 3])
    return model


def model_map(net_name='resnet50'):
    _dicts = {'resnet50': resnet50,
              'resnet101': resnet101, 'resnet152': resnet152}
    return _dicts[net_name]()


def build_net_resnet(phase, num_classes=2, net_name='resnet50'):
    resnet = model_map(net_name)
    extras = add_extras(extras_cfg, 2048)
    head_pal1 = multibox(net_cfg, num_classes)
    head_pal2 = multibox(net_cfg, num_classes)
    fem_modules = fem_module(net_cfg)
    model = DSFD(phase, resnet, extras, fem_modules,
                 head_pal1, head_pal2, num_classes)
    return model

if __name__ == '__main__':
    inputs = Variable(torch.randn(1, 3, 640, 640))
    net = build_net('train', 2, 101)
    out = net(inputs)
