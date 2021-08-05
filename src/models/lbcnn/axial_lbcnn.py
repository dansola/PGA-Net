import torch
from torch import nn
from src.models.axial_attention.axial_attention import AxialAttention
from src.models.lbcnn.lbcnn_parts import ConvLBP, BlockLBP


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BlockAxialLBC(nn.Module):
    def __init__(self, n_channels, embedding_dims, heads=2):
        super(BlockAxialLBC, self).__init__()
        self.n_channels = n_channels
        self.embedding_dims = embedding_dims
        self.embedding_dims_trip = embedding_dims * 3

        self.conv1 = conv1x1(self.n_channels, self.embedding_dims)
        self.bn1 = nn.BatchNorm2d(self.embedding_dims)
        self.relu = nn.ReLU(inplace=True)

        self.attn = AxialAttention(dim=self.embedding_dims, dim_index=1, heads=heads, num_dimensions=2,
                                   sum_axial_out=True)
        self.bn_lbc = nn.BatchNorm2d(self.embedding_dims)
        self.conv_lbc = ConvLBP(self.embedding_dims, self.embedding_dims)

        self.conv2 = conv1x1(self.embedding_dims_trip, self.embedding_dims)
        self.bn2 = nn.BatchNorm2d(self.embedding_dims)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x_attn = self.attn(x)
        x_attn = self.relu(x_attn)

        x_lbc = self.bn_lbc(x)
        x_lbc = self.conv_lbc(x_lbc)
        x_lbc = self.relu(x_lbc)

        x = torch.cat((x_attn, x_lbc, x), dim=1)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn2(x)
        return x


class AxialDownLBC(nn.Module):
    def __init__(self, n_channels, embedding_dims, heads=2):
        super(AxialDownLBC, self).__init__()
        self.n_channels = n_channels
        self.embedding_dims = embedding_dims
        self.cat_dims = embedding_dims + n_channels
        self.heads = heads

        self.mp = nn.MaxPool2d(2)
        self.conv1 = conv1x1(self.cat_dims, self.embedding_dims)
        self.bn1 = nn.BatchNorm2d(self.embedding_dims)
        self.relu = nn.ReLU(inplace=True)

        self.attn = AxialAttention(dim=self.n_channels, dim_index=1, heads=self.heads, num_dimensions=2,
                                   sum_axial_out=True)
        self.bn_lbc = nn.BatchNorm2d(self.n_channels)
        self.conv_lbc = ConvLBP(self.n_channels, self.n_channels)

    def forward(self, x):
        x = self.mp(x)

        x_attn = self.attn(x)
        x_attn = self.relu(x_attn)

        x_lbc = self.bn_lbc(x)
        x_lbc = self.conv_lbc(x_lbc)
        x_lbc = self.relu(x_lbc)

        x = torch.cat((x_attn, x_lbc, x), dim=1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        return x


class AxialUpLBC(nn.Module):
    def __init__(self, n_channels, embedding_dims, heads=2):
        super(AxialUpLBC, self).__init__()
        self.n_channels = n_channels
        self.embedding_dims = embedding_dims
        self.cat_dims = n_channels * 3 + embedding_dims
        self.heads = heads

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv1 = conv1x1(self.cat_dims, self.embedding_dims)
        self.bn1 = nn.BatchNorm2d(self.embedding_dims)
        self.relu = nn.ReLU(inplace=True)

        self.attn = AxialAttention(dim=self.n_channels, dim_index=1, heads=self.heads, num_dimensions=2,
                                   sum_axial_out=True)
        self.bn_lbc = nn.BatchNorm2d(self.n_channels)
        self.conv_lbc = ConvLBP(self.n_channels, self.n_channels)

    def forward(self, x, res):
        x = self.up(x)

        x_attn = self.attn(x)
        x_attn = self.relu(x_attn)

        x_lbc = self.bn_lbc(x)
        x_lbc = self.conv_lbc(x_lbc)
        x_lbc = self.relu(x_lbc)

        x = torch.cat((x_attn, x_lbc, x, res), dim=1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        return x


class AxialUNetLBC(nn.Module):
    def __init__(self, n_channels, n_classes, embedding_dims):
        super(AxialUNetLBC, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.embedding_dims = embedding_dims

        self.encode = BlockAxialLBC(self.n_channels, self.embedding_dims)

        self.down1 = AxialDownLBC(self.embedding_dims, self.embedding_dims * 2)
        self.down2 = AxialDownLBC(self.embedding_dims * 2, self.embedding_dims * 4)
        self.down3 = AxialDownLBC(self.embedding_dims * 4, self.embedding_dims * 8)
        self.down4 = AxialDownLBC(self.embedding_dims * 8, self.embedding_dims * 16)
        self.up1 = AxialUpLBC(self.embedding_dims * 16, self.embedding_dims * 8)
        self.up2 = AxialUpLBC(self.embedding_dims * 8, self.embedding_dims * 4)
        self.up3 = AxialUpLBC(self.embedding_dims * 4, self.embedding_dims * 2)
        self.up4 = AxialUpLBC(self.embedding_dims * 2, self.embedding_dims)

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


class SmallAxialUNetLBC(nn.Module):
    def __init__(self, n_channels, n_classes, embedding_dims):
        super(SmallAxialUNetLBC, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.embedding_dims = embedding_dims

        self.encode = BlockAxialLBC(self.n_channels, self.embedding_dims)

        self.down1 = AxialDownLBC(self.embedding_dims, self.embedding_dims * 2)
        self.down2 = AxialDownLBC(self.embedding_dims * 2, self.embedding_dims * 4)
        self.up1 = AxialUpLBC(self.embedding_dims * 4, self.embedding_dims * 2)
        self.up2 = AxialUpLBC(self.embedding_dims * 2, self.embedding_dims)

        self.decode = conv1x1(self.embedding_dims, self.n_classes)

    def forward(self, x):
        x1 = self.encode(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        x4 = self.up1(x3, x2)
        x5 = self.up2(x4, x1)

        logits = self.decode(x5)

        return logits


class BlockAxialLBC_Add(nn.Module):
    def __init__(self, n_channels, embedding_dims, heads=2):
        super(BlockAxialLBC_Add, self).__init__()
        self.n_channels = n_channels
        self.embedding_dims = embedding_dims
        self.embedding_dims_trip = embedding_dims * 3

        self.conv1 = conv1x1(self.n_channels, self.embedding_dims)
        self.bn1 = nn.BatchNorm2d(self.embedding_dims)
        self.relu = nn.ReLU(inplace=True)

        self.attn = AxialAttention(dim=self.embedding_dims, dim_index=1, heads=heads, num_dimensions=2,
                                   sum_axial_out=True)
        self.bn_lbc = nn.BatchNorm2d(self.embedding_dims)
        self.conv_lbc = BlockLBP(self.embedding_dims, self.embedding_dims)

        self.conv2 = conv1x1(self.embedding_dims, self.embedding_dims)
        self.bn2 = nn.BatchNorm2d(self.embedding_dims)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x_attn = self.attn(x)
        x_attn = self.relu(x_attn)

        x = self.conv_lbc(x)

        # x = torch.cat((x_attn, x_lbc, x), dim=1)
        x.add_(x_attn)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn2(x)
        return x


class BasicAxialLBC(nn.Module):
    def __init__(self, n_channels, n_classes, embedding_dims):
        super(BasicAxialLBC, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.embedding_dims = embedding_dims

        self.block1 = BlockAxialLBC(self.n_channels, self.embedding_dims)
        self.block2 = BlockAxialLBC(self.embedding_dims, self.embedding_dims)
        self.block3 = BlockAxialLBC(self.embedding_dims, self.embedding_dims)
        self.block4 = BlockAxialLBC(self.embedding_dims, self.embedding_dims)

        self.outc = conv1x1(self.embedding_dims, self.n_classes, 1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        logits = self.outc(x)
        return logits


class BasicAxialLBC_Add(nn.Module):
    def __init__(self, n_channels, n_classes, embedding_dims):
        super(BasicAxialLBC_Add, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.embedding_dims = embedding_dims

        self.block1 = BlockAxialLBC_Add(self.n_channels, self.embedding_dims)
        self.block2 = BlockAxialLBC_Add(self.embedding_dims, self.embedding_dims)
        self.block3 = BlockAxialLBC_Add(self.embedding_dims, self.embedding_dims)
        self.block4 = BlockAxialLBC_Add(self.embedding_dims, self.embedding_dims)

        self.outc = conv1x1(self.embedding_dims, self.n_classes, 1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        logits = self.outc(x)
        return logits


class LargeAxialLBC(nn.Module):
    def __init__(self, n_channels, n_classes, embedding_dims):
        super(LargeAxialLBC, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.embedding_dims = embedding_dims

        self.block1 = BlockAxialLBC(self.n_channels, self.embedding_dims)
        self.block2 = BlockAxialLBC(self.embedding_dims, self.embedding_dims)
        self.block3 = BlockAxialLBC(self.embedding_dims, self.embedding_dims)
        self.block4 = BlockAxialLBC(self.embedding_dims, self.embedding_dims)
        self.block5 = BlockAxialLBC(self.embedding_dims, self.embedding_dims)
        self.block6 = BlockAxialLBC(self.embedding_dims, self.embedding_dims)
        self.block7 = BlockAxialLBC(self.embedding_dims, self.embedding_dims)
        self.block8 = BlockAxialLBC(self.embedding_dims, self.embedding_dims)

        self.outc = conv1x1(self.embedding_dims, self.n_classes, 1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)

        logits = self.outc(x)
        return logits