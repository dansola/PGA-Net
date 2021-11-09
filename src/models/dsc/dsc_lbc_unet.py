import torch
import torch.nn as nn
import torch.nn.functional as F


class DSCUNetLBP(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(DSCUNetLBP, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DSCDSCBlockLBPUNet(n_channels, 64)
        self.down1 = DSCDownLBP(64, 128, sparsity=0.5)
        self.down2 = DSCDownLBP(128, 256, sparsity=0.5)
        self.down3 = DSCDownLBP(256, 512, sparsity=0.5)
        factor = 2 if bilinear else 1
        self.down4 = DSCDownLBP(512, 1024 // factor, sparsity=0.5)
        self.up1 = DSCUpLBP(1024, 512 // factor, bilinear, sparsity=0.5)
        self.up2 = DSCUpLBP(512, 256 // factor, bilinear, sparsity=0.5)
        self.up3 = DSCUpLBP(256, 128 // factor, bilinear, sparsity=0.5)
        self.up4 = DSCUpLBP(128, 64, bilinear, sparsity=0.5)
        self.outc = DSCDSCBlockLBPUNet(64, n_classes)

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


class DSCSmallUNetLBP(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(DSCSmallUNetLBP, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DSCDSCBlockLBPUNet(n_channels, 64)
        self.down1 = DSCDownLBP(64, 128)
        factor = 2 if bilinear else 1
        self.down2 = DSCDownLBP(128, 256 // factor)
        self.up1 = DSCUpLBP(256, 128 // factor, bilinear)
        self.up2 = DSCUpLBP(128, 64, bilinear)
        self.outc = DSCDSCBlockLBPUNet(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        logits = self.outc(x)
        return logits


class SkinnyDSCSmallUNetLBP(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, sparsity=0.8):
        super(SkinnyDSCSmallUNetLBP, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DSCDSCBlockLBPUNet(n_channels, 32, sparsity)
        self.down1 = DSCDownLBP(32, 64, sparsity)
        factor = 2 if bilinear else 1
        self.down2 = DSCDownLBP(64, 128 // factor, sparsity)
        self.up1 = DSCUpLBP(128, 64 // factor, bilinear, sparsity)
        self.up2 = DSCUpLBP(64, 32, bilinear, sparsity)
        self.outc = DSCDSCBlockLBPUNet(32, n_classes, sparsity)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        logits = self.outc(x)
        return logits


class SuperSkinnyDSCSmallUNetLBP(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(SuperSkinnyDSCSmallUNetLBP, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DSCDSCBlockLBPUNet(n_channels, 16)
        self.down1 = DSCDownLBP(16, 32)
        factor = 2 if bilinear else 1
        self.down2 = DSCDownLBP(32, 64 // factor)
        self.up1 = DSCUpLBP(64, 32 // factor, bilinear)
        self.up2 = DSCUpLBP(32, 16, bilinear)
        self.outc = DSCDSCBlockLBPUNet(16, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        logits = self.outc(x)
        return logits


class DSCConvLBP(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, sparsity=0.5):
        super().__init__(in_channels, out_channels, kernel_size, padding=1, bias=False, dilation=1, groups=in_channels)
        weights = next(self.parameters())
        matrix_proba = torch.FloatTensor(weights.data.shape).fill_(0.5)
        binary_weights = torch.bernoulli(matrix_proba) * 2 - 1
        mask_inactive = torch.rand(matrix_proba.shape) > sparsity
        binary_weights.masked_fill_(mask_inactive, 0)
        weights.data = binary_weights
        weights.requires_grad_(False)


class DSCBlockLBP(nn.Module):

    def __init__(self, n_channels, n_weights, sparsity=0.5):
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(n_channels)
        self.conv_lbp = DSCConvLBP(n_channels, n_weights, kernel_size=3, sparsity=sparsity)
        self.conv_1x1 = nn.Conv2d(n_weights, n_channels, kernel_size=1)

    def forward(self, x):
        res = x
        x = self.batch_norm(x)
        x = F.relu(self.conv_lbp(x))
        x = self.conv_1x1(x)
        x.add_(res)
        return x


class DSCDSCBlockLBPUNet(nn.Module):

    def __init__(self, n_channels, out_channels, sparsity=0.8):
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(n_channels)
        self.conv_lbp = DSCConvLBP(n_channels, n_channels, kernel_size=3, sparsity=sparsity)
        self.conv_1x1 = nn.Conv2d(n_channels, out_channels, kernel_size=1)

    def forward(self, x):
        res = x
        x = self.batch_norm(x)
        x = F.relu(self.conv_lbp(x))
        x.add_(res)
        x = self.conv_1x1(x)
        return x


class DSCDownLBP(nn.Module):
    def __init__(self, in_channels, out_channels, sparsity):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DSCDSCBlockLBPUNet(in_channels, out_channels, sparsity)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class DSCUpLBP(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, sparsity=0.8):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DSCDSCBlockLBPUNet(in_channels, out_channels, sparsity)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DSCDSCBlockLBPUNet(in_channels, out_channels, sparsity)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)