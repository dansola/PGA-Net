import torch
from torch import nn
from models.axial_attention.axial_attention import AxialAttention


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BlockAxial(nn.Module):
    def __init__(self, channels, embedding_dims):
        super(BlockAxial, self).__init__()
        self.channels = channels
        self.embedding_dims = embedding_dims
        self.embedding_dims_double = embedding_dims * 2

        self.conv1 = conv1x1(self.channels, self.embedding_dims)
        self.bn1 = nn.BatchNorm2d(self.embedding_dims)
        self.relu = nn.ReLU(inplace=True)

        self.attn = AxialAttention(dim=self.embedding_dims, dim_index=1, heads=2, num_dimensions=2,
                                   sum_axial_out=True)

        self.conv2 = conv1x1(self.embedding_dims_double, self.embedding_dims)
        self.bn2 = nn.BatchNorm2d(self.embedding_dims)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x_attn = self.attn(x)
        x_attn = self.relu(x_attn)

        x = torch.cat((x_attn, x), dim=1)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x


class BlockAxialDown(nn.Module):
    def __init__(self, channels, embedding_dims, heads=2):
        super(BlockAxialDown, self).__init__()
        self.channels = channels
        self.embedding_dims = embedding_dims
        self.heads = heads

        self.mp = nn.MaxPool2d(2)
        self.conv1 = conv1x1(self.embedding_dims, self.embedding_dims)
        self.bn1 = nn.BatchNorm2d(self.embedding_dims)
        self.relu = nn.ReLU(inplace=True)

        self.attn = AxialAttention(dim=self.channels, dim_index=1, heads=self.heads, num_dimensions=2,
                                   sum_axial_out=True)

    def forward(self, x):
        x = self.mp(x)

        x_attn = self.attn(x)
        x_attn = self.relu(x_attn)

        x = torch.cat((x_attn, x), dim=1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        return x


class BlockAxialUp(nn.Module):
    def __init__(self, channels, embedding_dims, heads=2):
        super(BlockAxialUp, self).__init__()
        self.channels = channels
        self.embedding_dims = embedding_dims
        self.cat_dims = channels * 2 + embedding_dims
        self.heads = heads

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv1 = conv1x1(self.cat_dims, self.embedding_dims)
        self.bn1 = nn.BatchNorm2d(self.embedding_dims)
        self.relu = nn.ReLU(inplace=True)

        self.attn = AxialAttention(dim=self.channels, dim_index=1, heads=self.heads, num_dimensions=2,
                                   sum_axial_out=True)

    def forward(self, x, res):
        x = self.up(x)

        x_attn = self.attn(x)
        x_attn = self.relu(x_attn)

        x = torch.cat((x_attn, x, res), dim=1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        return x
