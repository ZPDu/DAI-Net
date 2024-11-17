import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import constant_init, kaiming_init
from mmcv.cnn import ConvModule
from mmcv.runner import auto_fp16, load_checkpoint

from mmdet.models.builder import SHARED_HEADS, build_loss
from mmdet.utils import get_root_logger

class Interpolate(nn.Module):
    def __init__(self, scale_factor):
        super(Interpolate, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale_factor, mode='nearest')
        return x


@SHARED_HEADS.register_module()
class REF_head(nn.Module):
    '''The decoder of REF branches, input the feat of original images and
    feat of transformed images, passed by global pool and return the transformed
    results'''

    def __init__(self,
                 indim=64
                 ):
        super(REF_head, self).__init__()
        self.indim = indim
        self.ref = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            Interpolate(2),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                # m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, feat):
        x = self.ref(feat)
        return x

    def init_weights(self):
        """Initialize the weights of module."""
        # init is done in LinearBlock
        pass





