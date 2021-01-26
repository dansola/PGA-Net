from torch import nn
from models.utils import PositionalEncodingPermute2D, AxialPositionalEmbedding, elem_add
from models.axial_attention import AxialAttention


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Embed(nn.Module):
    def __init__(self, channels, embedding_dims, sine_pos=True, img_shape=(300, 300)):
        super(Embed, self).__init__()
        self.channels = channels
        self.embedding_dims = embedding_dims
        self.sine_pos = sine_pos
        self.img_shape = img_shape

        self.embed = nn.Linear(self.channels, self.embedding_dims, bias=False)
        if self.sine_pos:
            self.pos = PositionalEncodingPermute2D(self.embedding_dims)
        else:
            self.pos = AxialPositionalEmbedding(self.embedding_dims, self.img_shape)
        self.bn = nn.BatchNorm2d(self.embedding_dims)

    def forward(self, x):
        embedded = self.embed(x).permute(0, 3, 1, 2)
        if self.sine_pos:
            pos = self.pos(embedded)
            embedded_pos = elem_add(embedded, pos)
        else:
            embedded_pos = self.pos(embedded.contiguous())
        embedded_norm = self.bn(embedded_pos)

        return embedded_norm


class AttentionDown(nn.Module):
    def __init__(self, embedding_dims, do_downsample=True, stride=2):
        super(AttentionDown, self).__init__()
        self.embedding_dims = embedding_dims
        self.embedding_dims_half = int(embedding_dims / 2)
        self.do_downsample = do_downsample
        self.stride = stride

        self.conv_down1 = nn.Sequential(conv1x1(self.embedding_dims, self.embedding_dims_half, 1),
                                        nn.BatchNorm2d(self.embedding_dims_half),
                                        nn.ReLU(inplace=True))
        self.conv_down2 = nn.Sequential(conv1x1(self.embedding_dims, self.embedding_dims, 1),
                                        nn.BatchNorm2d(self.embedding_dims),
                                        nn.ReLU(inplace=True))
        self.attn = AxialAttention(dim=self.embedding_dims_half, dim_index=1, heads=2, num_dimensions=2,
                                   sum_axial_out=True)
        self.conv_up = conv1x1(self.embedding_dims_half, self.embedding_dims, 1)
        self.bn = nn.BatchNorm2d(self.embedding_dims)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(conv1x1(self.embedding_dims, self.embedding_dims * 2, self.stride),
                                        nn.BatchNorm2d(self.embedding_dims * 2))

    def forward(self, x):
        x_conv_down1 = self.conv_down1(x)
        x_conv_down2 = self.conv_down2(x)

        embedded_attn = self.attn(x_conv_down1)
        embedded_attn = self.relu(embedded_attn)

        x_conv_up = self.conv_up(embedded_attn)
        embedded_skip = elem_add(x_conv_up, x_conv_down2)
        embedded_final = self.bn(embedded_skip)

        if self.do_downsample:
            embedded_final = self.downsample(embedded_final)
        embedded_final = self.relu(embedded_final)

        return embedded_final


class AttentionUp(nn.Module):
    def __init__(self, embedding_dims, skip=False):
        super(AttentionUp, self).__init__()
        self.embedding_dims = embedding_dims
        self.embedding_dims_half = int(embedding_dims / 2)
        self.skip = skip

        self.conv_down1 = nn.Sequential(conv1x1(self.embedding_dims, self.embedding_dims_half, 1),
                                        nn.BatchNorm2d(self.embedding_dims_half),
                                        nn.ReLU(inplace=True),
                                        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        self.conv_down2 = nn.Sequential(conv1x1(self.embedding_dims, self.embedding_dims_half, 1),
                                        nn.BatchNorm2d(self.embedding_dims_half),
                                        nn.ReLU(inplace=True),
                                        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        self.attn = AxialAttention(dim=self.embedding_dims_half, dim_index=1, heads=2, num_dimensions=2,
                                   sum_axial_out=True)
        self.conv_up = conv1x1(self.embedding_dims_half, self.embedding_dims_half, 1)
        self.conv_up_encoder = conv1x1(self.embedding_dims, self.embedding_dims_half, 1)
        self.bn = nn.BatchNorm2d(self.embedding_dims_half)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1, x2):
        x_conv_down1 = self.conv_down1(x1)
        x_conv_down2 = self.conv_down2(x1)

        embedded_attn = self.attn(x_conv_down1)
        embedded_attn = self.relu(embedded_attn)

        x_conv_up = self.conv_up(embedded_attn)
        embedded_skip = elem_add(x_conv_up, x_conv_down2)

        if self.skip:
            conv_up_encoder = self.conv_up_encoder(x2)
            embedded_skip = elem_add(embedded_skip, conv_up_encoder)

        embedded_final = self.bn(embedded_skip)

        return embedded_final
