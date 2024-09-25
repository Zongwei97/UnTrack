import math
import logging
import pdb
from functools import partial
from collections import OrderedDict
from copy import deepcopy

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import to_2tuple

import functools
from torch import nn, Tensor
from info_nce import InfoNCE

def gradient(depth_tmp):
    step = 7

    B, C, H, W = depth_tmp.size()
    pad = (step - 1) // 2
    depth_tmp = F.pad(depth_tmp, [pad, pad, pad, pad], mode='constant', value=0)
    patches = depth_tmp.unfold(dimension=2, size=step, step=1)
    patches = patches.unfold(dimension=3, size=step, step=1)
    max_depth, _ = patches.reshape(B, C, H, W, -1).max(dim=-1)

    # 求解max_depth四个方向梯度, 随后concat depth, 以缓解深度跳变对深度预测模块的影响
    step = float(step)
    shift_list = [[step / H, 0.0 / W], [-step / H, 0.0 / W], [0.0 / H, step / W], [0.0 / H, -step / W]]
    output_list = []
    for shift in shift_list:
        transform_matrix = torch.tensor([[1, 0, shift[0]], [0, 1, shift[1]]]).unsqueeze(0).repeat(B, 1, 1).cuda()
        grid = F.affine_grid(transform_matrix, max_depth.shape).float()
        output = F.grid_sample(max_depth, grid, mode='nearest')  # 平移后图像
        output = max_depth - output
        output_mask = ((output == max_depth) == False)
        output = output * output_mask
        output_list.append(output)
    grad = torch.cat(output_list, dim=1)
    max_grad = torch.abs(grad).max(dim=1)[0].unsqueeze(1)

    return grad, max_grad

if __name__ == "__main__":
    path = '/home/zwu/Tracking/data/depthtrack/train/ball02_indoor/color/00001464.jpg'
    im = plt.imread(path)
    rgb = torch.from_numpy(im).unsqueeze(0).permute(0,3,1,2).cuda()
    rgb = F.interpolate(rgb, (256,256)).float()
    grad = gradient(rgb)
    max_grad = torch.abs(grad).max(dim=1)[0].unsqueeze(1)
    a=1