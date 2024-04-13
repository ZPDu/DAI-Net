# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import


from .DSFD_vgg import build_net_vgg
from .DSFD_resnet import build_net_resnet
from .DAINet import build_net_dark


def build_net(phase, num_classes=2, model='vgg'):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return

    if model == 'vgg':
        return build_net_vgg(phase, num_classes)
    elif model == 'dark':
        return build_net_dark(phase, num_classes)
    else:
        return build_net_resnet(phase, num_classes, model)



def basenet_factory(model='vgg'):
	if model=='vgg' or model=='dark':
		basenet = 'vgg16_reducedfc.pth'

	elif 'resnet' in model:
		basenet = '{}.pth'.format(model)
	return basenet

