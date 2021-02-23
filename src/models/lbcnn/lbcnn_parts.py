import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLBP(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, sparsity=0.5):
        super().__init__(in_channels, out_channels, kernel_size, padding=1, bias=False)
        weights = next(self.parameters())
        matrix_proba = torch.FloatTensor(weights.data.shape).fill_(0.5)
        binary_weights = torch.bernoulli(matrix_proba) * 2 - 1
        mask_inactive = torch.rand(matrix_proba.shape) > sparsity
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
