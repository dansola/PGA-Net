import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetDSC(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNetDSC, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConvDSC(n_channels, 64)
        self.down1 = DownDSC(64, 128)
        self.down2 = DownDSC(128, 256)
        self.down3 = DownDSC(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = DownDSC(512, 1024 // factor)
        self.up1 = UpDSC(1024, 512 // factor, bilinear)
        self.up2 = UpDSC(512, 256 // factor, bilinear)
        self.up3 = UpDSC(256, 128 // factor, bilinear)
        self.up4 = UpDSC(128, 64, bilinear)
        self.outc = OutConvDSC(64, n_classes)

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


class SmallUNetDSC(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(SmallUNetDSC, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConvDSC(n_channels, 64)
        self.down1 = DownDSC(64, 128)
        factor = 2 if bilinear else 1
        self.down2 = DownDSC(128, 256 // factor)
        self.up1 = UpDSC(256, 128 // factor, bilinear)
        self.up2 = UpDSC(128, 64, bilinear)
        self.outc = OutConvDSC(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        logits = self.outc(x)
        return logits


class Conv2dDSC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        depth_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, groups=in_channels, padding=1)
        point_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.depthwise_separable_conv = nn.Sequential(depth_conv, point_conv)

    def forward(self, x):
        return self.depthwise_separable_conv(x)


class DoubleConvDSC(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            Conv2dDSC(in_channels, mid_channels),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            Conv2dDSC(mid_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DownDSC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvDSC(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpDSC(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConvDSC(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConvDSC(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConvDSC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConvDSC, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
