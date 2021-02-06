import torch
from torch import nn
from models.basic_axial.basic_axial_parts import BlockAxial, conv1x1
from models.basic_pga.basic_pga_parts import BlockPGA


class BasicAxialPGA(nn.Module):
    def __init__(self, channels, n_classes, embedding_dims, img_crop=320):
        super(BasicAxialPGA, self).__init__()
        self.channels = channels
        self.n_classes = n_classes
        self.embedding_dims = embedding_dims
        self.img_crop = img_crop

        self.block_a1 = BlockAxial(self.channels, self.embedding_dims, img_shape=(self.img_crop, self.img_crop))
        self.block_pga1 = BlockPGA(self.channels, self.embedding_dims, img_shape=(self.img_crop, self.img_crop))
        self.block_a2 = BlockAxial(self.embedding_dims, self.embedding_dims, img_shape=(self.img_crop, self.img_crop))
        self.block_pga2 = BlockPGA(self.embedding_dims, self.embedding_dims, img_shape=(self.img_crop, self.img_crop))
        # self.block5 = Block(self.embedding_dims, self.embedding_dims, img_shape=(self.img_crop, self.img_crop))

        self.down1 = conv1x1(self.embedding_dims * 2, self.embedding_dims, 1)
        # self.down2 = conv1x1(self.embedding_dims * 2, self.embedding_dims, 1)

        self.outc = conv1x1(self.embedding_dims * 2, self.n_classes, 1)
        self.out = nn.Softmax(dim=1)
    def set_indices(self,indices):
        self.block_pga1.indice = indices


    def forward(self, x, prop):
        x_a1 = self.block_a1(x)
        x_pga1 = self.block_pga1(x, prop)
        x = torch.cat((x_a1, x_pga1), dim=1)
        x = self.down1(x)

        x_a2 = self.block_a2(x)
        x_pga2 = self.block_pga2(x, prop)
        x = torch.cat((x_a2, x_pga2), dim=1)
        # x = self.down1(x)

        logits = self.outc(x)
        out = self.out(logits)

        return out