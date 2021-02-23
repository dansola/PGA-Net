from torch import nn
from models.basic_axial.basic_axial_parts import BlockAxial, conv1x1, BlockAxialDown, BlockAxialUp


class BasicAxial(nn.Module):
    def __init__(self, channels, n_classes, embedding_dims):
        super(BasicAxial, self).__init__()
        self.channels = channels
        self.n_classes = n_classes
        self.embedding_dims = embedding_dims

        self.block1 = BlockAxial(self.channels, self.embedding_dims)
        self.block2 = BlockAxial(self.embedding_dims, self.embedding_dims)
        self.block3 = BlockAxial(self.embedding_dims, self.embedding_dims)
        self.block4 = BlockAxial(self.embedding_dims, self.embedding_dims)

        self.outc = conv1x1(self.embedding_dims, self.n_classes, 1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        logits = self.outc(x)

        return logits


class AxialUNetSmall(nn.Module):
    def __init__(self, channels, n_classes, embedding_dims):
        super(AxialUNetSmall, self).__init__()
        self.channels = channels
        self.n_classes = n_classes
        self.embedding_dims = embedding_dims

        self.encode = BlockAxial(self.channels, self.embedding_dims)
        self.decode = conv1x1(self.embedding_dims, 3)
        self.down = BlockAxialDown(self.embedding_dims, self.embedding_dims * 2)
        self.up = BlockAxialUp(self.embedding_dims * 2, self.embedding_dims)

    def forward(self, x):
        x1 = self.encode(x)
        x2 = self.down(x1)
        x3 = self.up(x2, x1)
        logits = self.decode(x3)

        return logits


class AxialUNetMed(nn.Module):
    def __init__(self, channels, n_classes, embedding_dims):
        super(AxialUNetMed, self).__init__()
        self.channels = channels
        self.n_classes = n_classes
        self.embedding_dims = embedding_dims

        self.encode = BlockAxial(self.channels, self.embedding_dims)

        self.down1 = BlockAxialDown(self.embedding_dims, self.embedding_dims * 2)
        self.down2 = BlockAxialDown(self.embedding_dims * 2, self.embedding_dims * 4)
        self.up1 = BlockAxialUp(self.embedding_dims * 4, self.embedding_dims * 2)
        self.up2 = BlockAxialUp(self.embedding_dims * 2, self.embedding_dims)

        self.decode = conv1x1(self.embedding_dims, 3)

    def forward(self, x):
        x1 = self.encode(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        x4 = self.up1(x3, x2)
        x5 = self.up2(x4, x1)
        logits = self.decode(x5)

        return logits


class AxialUNet(nn.Module):
    def __init__(self, channels, n_classes, embedding_dims):
        super(AxialUNet, self).__init__()
        self.channels = channels
        self.n_classes = n_classes
        self.embedding_dims = embedding_dims

        self.encode = BlockAxial(self.channels, self.embedding_dims)

        self.down1 = BlockAxialDown(self.embedding_dims, self.embedding_dims * 2)
        self.down2 = BlockAxialDown(self.embedding_dims * 2, self.embedding_dims * 4)
        self.down3 = BlockAxialDown(self.embedding_dims * 4, self.embedding_dims * 8)
        self.down4 = BlockAxialDown(self.embedding_dims * 8, self.embedding_dims * 16)
        self.up1 = BlockAxialUp(self.embedding_dims * 16, self.embedding_dims * 8)
        self.up2 = BlockAxialUp(self.embedding_dims * 8, self.embedding_dims * 4)
        self.up3 = BlockAxialUp(self.embedding_dims * 4, self.embedding_dims * 2)
        self.up4 = BlockAxialUp(self.embedding_dims * 2, self.embedding_dims)

        self.decode = conv1x1(self.embedding_dims, 3)

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
