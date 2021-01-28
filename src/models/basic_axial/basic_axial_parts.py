import torch
from torch import nn
from models.axial_unet.utils import AxialPositionalEmbedding
from models.axial_unet.axial_attention import AxialAttention, AxialImageTransformer


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Block(nn.Module):
    def __init__(self, channels, embedding_dims, img_shape=(300, 300)):
        super(Block, self).__init__()
        self.channels = channels
        self.embedding_dims = embedding_dims
        self.embedding_dims_double = embedding_dims * 2
        self.img_shape = img_shape

        self.conv1 = conv1x1(self.channels, self.embedding_dims)
        self.bn1 = nn.BatchNorm2d(self.embedding_dims)
        self.relu = nn.ReLU(inplace=True)

        self.pos = AxialPositionalEmbedding(self.embedding_dims, self.img_shape)
        self.attn = AxialAttention(dim=self.embedding_dims, dim_index=1, heads=2, num_dimensions=2,
                                   sum_axial_out=True)

        self.conv2 = conv1x1(self.embedding_dims_double, self.embedding_dims)
        # self.conv2 = conv1x1(self.embedding_dims, self.embedding_dims)
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
