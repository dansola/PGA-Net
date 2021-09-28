from torch import nn
from src.models.lbcnn.lbcnn_parts import BlockLBPUNet, DownLBP, UpLBP


class UNetLBP(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNetLBP, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = BlockLBPUNet(n_channels, 64)
        self.down1 = DownLBP(64, 128)
        self.down2 = DownLBP(128, 256)
        self.down3 = DownLBP(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = DownLBP(512, 1024 // factor)
        self.up1 = UpLBP(1024, 512 // factor, bilinear)
        self.up2 = UpLBP(512, 256 // factor, bilinear)
        self.up3 = UpLBP(256, 128 // factor, bilinear)
        self.up4 = UpLBP(128, 64, bilinear)
        self.outc = BlockLBPUNet(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class SmallUNetLBP(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(SmallUNetLBP, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = BlockLBPUNet(n_channels, 64)
        self.down1 = DownLBP(64, 128)
        factor = 2 if bilinear else 1
        self.down2 = DownLBP(128, 256 // factor)
        self.up1 = UpLBP(256, 128 // factor, bilinear)
        self.up2 = UpLBP(128, 64, bilinear)
        self.outc = BlockLBPUNet(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        logits = self.outc(x)
        return logits


class SkinnySmallUNetLBP(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(SkinnySmallUNetLBP, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = BlockLBPUNet(n_channels, 32)
        self.down1 = DownLBP(32, 64)
        factor = 2 if bilinear else 1
        self.down2 = DownLBP(64, 128 // factor)
        self.up1 = UpLBP(128, 64 // factor, bilinear)
        self.up2 = UpLBP(64, 32, bilinear)
        self.outc = BlockLBPUNet(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        logits = self.outc(x)
        return logits