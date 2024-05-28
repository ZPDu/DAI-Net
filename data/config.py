#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
from easydict import EasyDict
import numpy as np


_C = EasyDict()
cfg = _C
# data augument config
_C.expand_prob = 0.5
_C.expand_max_ratio = 4
_C.hue_prob = 0.5
_C.hue_delta = 18
_C.contrast_prob = 0.5
_C.contrast_delta = 0.5
_C.saturation_prob = 0.5
_C.saturation_delta = 0.5
_C.brightness_prob = 0.5
_C.brightness_delta = 0.125
_C.data_anchor_sampling_prob = 0.5
_C.min_face_size = 6.0
_C.apply_distort = True
_C.apply_expand = False
# Modified to fit the input requirement of DARK ISP
_C.img_mean = np.array([0., 0., 0.])[:, np.newaxis, np.newaxis].astype(
    'float32')
_C.resize_width = 640
_C.resize_height = 640
_C.scale = 1 / 127.0
_C.anchor_sampling = True
_C.filter_min_face = True

# train config
_C.LR_STEPS = (10000*2,12500*2,15000*2)
_C.MAX_STEPS = 150000
_C.EPOCHES = 100

# anchor config
_C.FEATURE_MAPS = [160, 80, 40, 20, 10, 5]
_C.INPUT_SIZE = 640
_C.STEPS = [4, 8, 16, 32, 64, 128]
_C.ANCHOR_SIZES1 = [8, 16, 32, 64, 128, 256]
_C.ANCHOR_SIZES2 = [16, 32, 64, 128, 256, 512]
_C.ASPECT_RATIO = [1.0]
_C.CLIP = False
_C.VARIANCE = [0.1, 0.2]

# detection config
_C.NMS_THRESH = 0.3
_C.NMS_TOP_K = 5000
_C.TOP_K = 750
_C.CONF_THRESH = 0.05

# loss config
_C.NEG_POS_RATIOS = 3
_C.NUM_CLASSES = 2

_C.WEIGHT = EasyDict()
_C.WEIGHT.EQUAL_R = 0.01
_C.WEIGHT.SMOOTH = 0.5
_C.WEIGHT.RC = 0.001
_C.WEIGHT.MC = 0.1

# dataset config
_C.HOME = ''


# face config
_C.FACE = EasyDict()
_C.FACE.TRAIN_FILE = ''
_C.FACE.VAL_FILE = ''

_C.FACE.OVERLAP_THRESH = 0.35

