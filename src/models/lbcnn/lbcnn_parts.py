import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConvLBP(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, sparsity=0):
        super().__init__(in_channels, out_channels, kernel_size, padding=1, bias=False, dilation=1)
        weights = next(self.parameters())
        matrix_proba = torch.FloatTensor(weights.data.shape).fill_(0.5)  # same shape as weights filled with 0.5
        binary_weights = torch.bernoulli(matrix_proba) * 2 - 1  # 50/50 chance of being 1 or -1
        mask_inactive = torch.rand(matrix_proba.shape) > sparsity  # 'sparsity' (50/50 by default) chance of a vale being zero
        binary_weights.masked_fill_(mask_inactive, 0)
        weights.data = binary_weights
        weights.requires_grad_(False)


class BlockLBP(nn.Module):

    def __init__(self, n_channels, n_weights, sparsity=0.5):
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(n_channels)
        self.conv_lbp = ConvLBP(n_channels, n_weights, kernel_size=3, sparsity=sparsity)
        self.conv_1x1 = nn.Conv2d(n_weights, n_channels, kernel_size=1)

    def forward(self, x):
        res = x
        x = self.batch_norm(x)
        x = F.relu(self.conv_lbp(x))
        x = self.conv_1x1(x)
        x.add_(res)
        return x


class BlockLBPUNet(nn.Module):

    def __init__(self, n_channels, out_channels, sparsity=0.5):
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(n_channels)
        self.conv_lbp = ConvLBP(n_channels, n_channels, kernel_size=3, sparsity=sparsity)
        self.conv_1x1 = nn.Conv2d(n_channels, out_channels, kernel_size=1)

    def forward(self, x):
        res = x
        x2 = self.batch_norm(x)
        x3 = F.relu(self.conv_lbp(x2))
        # x_detached = x3.detach().cpu().numpy()
        # print(np.all(x_detached==0))
        x3.add_(res)
        x4 = self.conv_1x1(x3)
        return x4


class DownLBP(nn.Module):
    def __init__(self, in_channels, out_channels, sparsity=0.5):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            BlockLBPUNet(in_channels, out_channels, sparsity)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpLBP(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, sparsity=0.5):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = BlockLBPUNet(in_channels, out_channels, sparsity)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = BlockLBPUNet(in_channels, out_channels, sparsity)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)