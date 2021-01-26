from torch import nn
from models.axial_unet_parts import Embed, AttentionDown, AttentionUp, conv1x1


class AxialUnet(nn.Module):
    def __init__(self, channels, n_classes, embedding_dims, device, do_downsample=True, stride=2, sine_pos=True):
        super(AxialUnet, self).__init__()
        self.channels = channels
        self.n_classes = n_classes
        self.embedding_dims = embedding_dims
        self.device = device
        self.do_downsample = do_downsample
        self.stride = stride
        self.sine_pos = sine_pos

        self.embed = Embed(self.channels, self.embedding_dims, sine_pos=self.sine_pos)
        self.down1 = AttentionDown(self.embedding_dims)
        self.down2 = AttentionDown(self.embedding_dims * 2)
        self.down3 = AttentionDown(self.embedding_dims * 4)
        self.down4 = AttentionDown(self.embedding_dims * 8)

        self.up1 = AttentionUp(self.embedding_dims * 16, skip=False)
        self.up2 = AttentionUp(self.embedding_dims * 8)
        self.up3 = AttentionUp(self.embedding_dims * 4)
        self.up4 = AttentionUp(self.embedding_dims * 2)

        self.outc = conv1x1(self.embedding_dims, self.n_classes, 1)

    def forward(self, x):
        x1 = self.embed(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x6 = self.up1(x5, x5)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)

        logits = self.outc(x9)

        return logits
