from torch import nn
from models.basic_axial.basic_axial_parts import BlockAxial, conv1x1

class BasicAxial(nn.Module):
    def __init__(self, channels, n_classes, embedding_dims, img_crop=320):
        super(BasicAxial, self).__init__()
        self.channels = channels
        self.n_classes = n_classes
        self.embedding_dims = embedding_dims
        self.img_crop = img_crop

        self.block1 = BlockAxial(self.channels, self.embedding_dims, img_shape=(self.img_crop, self.img_crop))
        self.block2 = BlockAxial(self.embedding_dims, self.embedding_dims, img_shape=(self.img_crop, self.img_crop))
        self.block3 = BlockAxial(self.embedding_dims, self.embedding_dims, img_shape=(self.img_crop, self.img_crop))
        self.block4 = BlockAxial(self.embedding_dims, self.embedding_dims, img_shape=(self.img_crop, self.img_crop))
        # self.block5 = Block(self.embedding_dims, self.embedding_dims, img_shape=(self.img_crop, self.img_crop))

        self.outc = conv1x1(self.embedding_dims, self.n_classes, 1)
        self.out = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        # x = self.block5(x)

        logits = self.outc(x)
        out = self.out(logits)

        return out