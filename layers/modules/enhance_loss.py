from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure as ssim
from data.config import cfg

def gradient(input_tensor, direction):
    smooth_kernel_x = torch.FloatTensor([[0, 0], [-1, 1]]).view((1, 1, 2, 2)).cuda()
    smooth_kernel_y = torch.transpose(smooth_kernel_x, 2, 3)

    if direction == "x":
        kernel = smooth_kernel_x
    elif direction == "y":
        kernel = smooth_kernel_y
    grad_out = torch.abs(F.conv2d(input_tensor, kernel,
                                  stride=1, padding=1))
    return grad_out

def ave_gradient(input_tensor, direction):
    return F.avg_pool2d(gradient(input_tensor, direction),
                        kernel_size=3, stride=1, padding=1)

def smooth(input_I, input_R):
    input_R = 0.299 * input_R[:, 0, :, :] + 0.587 * input_R[:, 1, :, :] + 0.114 * input_R[:, 2, :, :]
    input_R = torch.unsqueeze(input_R, dim=1)
    return torch.mean(gradient(input_I, "x") * torch.exp(-10 * ave_gradient(input_R, "x")) +
                      gradient(input_I, "y") * torch.exp(-10 * ave_gradient(input_R, "y")))

class EnhanceLoss(nn.Module):
    def __init__(self):
        super(EnhanceLoss, self).__init__()

    def forward(self, preds, img, img_dark):
        R_dark, R_light, R_dark_2, R_light_2, I_dark, I_light = preds

        losses_equal_R = (F.mse_loss(R_dark, R_light.detach())) * cfg.WEIGHT.EQUAL_R
        losses_recon_low = F.mse_loss(R_dark * I_dark, img_dark) * 1.+ (1. - ssim(R_dark * I_dark, img_dark))
        losses_recon_high = F.mse_loss(R_light * I_light, img) * 1.+ (1. - ssim(R_light * I_light, img))

        losses_smooth_low = smooth(I_dark, R_dark) * cfg.WEIGHT.SMOOTH
        losses_smooth_high = smooth(I_light, R_light) * cfg.WEIGHT.SMOOTH
        # Redecomposition cohering loss
        losses_rc = (F.mse_loss(R_dark_2, R_dark.detach()) + F.mse_loss(R_light_2, R_light.detach())) * cfg.WEIGHT.RC

        enhance_loss = losses_equal_R + losses_recon_low + losses_recon_high + losses_smooth_low \
                       + losses_smooth_high + losses_rc

        return enhance_loss