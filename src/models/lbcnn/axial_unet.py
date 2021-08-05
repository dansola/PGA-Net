import torch
from torch import nn

from src.models.axial_attention.axial_attention import AxialAttention
from src.models.basic_axial.basic_axial_parts import BlockAxial
from src.models.lbcnn.axial_lbcnn import conv1x1


class AxialDown(nn.Module):
    def __init__(self, n_channels, embedding_dims, heads=2):
        super(AxialDown, self).__init__()
        self.n_channels = n_channels
        self.embedding_dims = embedding_dims
        self.cat_dims = embedding_dims
        self.heads = heads

        self.mp = nn.MaxPool2d(2)
        self.conv1 = conv1x1(self.cat_dims, self.embedding_dims)
        self.bn1 = nn.BatchNorm2d(self.embedding_dims)
        self.relu = nn.ReLU(inplace=True)

        self.attn = AxialAttention(dim=self.n_channels, dim_index=1, heads=self.heads, num_dimensions=2,
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


class AxialUp(nn.Module):
    def __init__(self, n_channels, embedding_dims, heads=2):
        super(AxialUp, self).__init__()
        self.n_channels = n_channels
        self.embedding_dims = embedding_dims
        self.cat_dims = n_channels * 2 + embedding_dims
        self.heads = heads

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv1 = conv1x1(self.cat_dims, self.embedding_dims)
        self.bn1 = nn.BatchNorm2d(self.embedding_dims)
        self.relu = nn.ReLU(inplace=True)

        self.attn = AxialAttention(dim=self.n_channels, dim_index=1, heads=self.heads, num_dimensions=2,
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


class AxialUNet(nn.Module):
    """
    New version of AxialUNet that differs from src/models/axial_unet/axial_unet to be more considtent with the axial
    unet lbc for a direct comparison.  This involves no positional encoding.
    """
    def __init__(self, n_channels, n_classes, embedding_dims):
        super(AxialUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.embedding_dims = embedding_dims

        self.encode = BlockAxial(self.n_channels, self.embedding_dims)

        self.down1 = AxialDown(self.embedding_dims, self.embedding_dims * 2)
        self.down2 = AxialDown(self.embedding_dims * 2, self.embedding_dims * 4)
        self.down3 = AxialDown(self.embedding_dims * 4, self.embedding_dims * 8)
        self.down4 = AxialDown(self.embedding_dims * 8, self.embedding_dims * 16)
        self.up1 = AxialUp(self.embedding_dims * 16, self.embedding_dims * 8)
        self.up2 = AxialUp(self.embedding_dims * 8, self.embedding_dims * 4)
        self.up3 = AxialUp(self.embedding_dims * 4, self.embedding_dims * 2)
        self.up4 = AxialUp(self.embedding_dims * 2, self.embedding_dims)

        self.decode = conv1x1(self.embedding_dims, self.n_classes)

    def forward(self, x):
        x1 = self.encode(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x6 = self.up1(x5, x4)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)

        logits = self.decode(x9)

        return logits


class SmallAxialUNet(nn.Module):
    def __init__(self, n_channels, n_classes, embedding_dims):
        super(SmallAxialUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.embedding_dims = embedding_dims

        self.encode = BlockAxial(self.n_channels, self.embedding_dims)

        self.down1 = AxialDown(self.embedding_dims, self.embedding_dims * 2)
        self.down2 = AxialDown(self.embedding_dims * 2, self.embedding_dims * 4)
        self.up1 = AxialUp(self.embedding_dims * 4, self.embedding_dims * 2)
        self.up2 = AxialUp(self.embedding_dims * 2, self.embedding_dims)

        self.decode = conv1x1(self.embedding_dims, self.n_classes)

    def forward(self, x):
        x1 = self.encode(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        x4 = self.up1(x3, x2)
        x5 = self.up2(x4, x1)

        logits = self.decode(x5)

        return logits