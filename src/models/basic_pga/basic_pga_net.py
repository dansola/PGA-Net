import torch
from torch import nn
from src.models.basic_axial.basic_axial_parts import BlockAxial, conv1x1
from src.models.basic_pga.basic_pga_parts import BlockPGA


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

        self.down1 = conv1x1(self.embedding_dims * 2, self.embedding_dims, 1)

        self.outc = conv1x1(self.embedding_dims * 2, self.n_classes, 1)

    def forward(self, x, obj_dict, bg_dict):
        x_a1 = self.block_a1(x)
        x_pga1 = self.block_pga1(x, obj_dict, bg_dict)
        x = torch.cat((x_a1, x_pga1), dim=1)
        x = self.down1(x)

        x_a2 = self.block_a2(x)
        x_pga2 = self.block_pga2(x, obj_dict, bg_dict)
        x = torch.cat((x_a2, x_pga2), dim=1)

        logits = self.outc(x)

        return logits


class OnlyPGA(nn.Module):
    def __init__(self, channels, n_classes, embedding_dims, img_crop=320):
        super(OnlyPGA, self).__init__()
        self.channels = channels
        self.n_classes = n_classes
        self.embedding_dims = embedding_dims
        self.img_crop = img_crop

        self.block_pga1 = BlockPGA(self.channels, self.embedding_dims, img_shape=(self.img_crop, self.img_crop))
        self.block_pga2 = BlockPGA(self.embedding_dims, self.embedding_dims, img_shape=(self.img_crop, self.img_crop))
        self.block_pga3 = BlockPGA(self.embedding_dims, self.embedding_dims, img_shape=(self.img_crop, self.img_crop))
        self.block_pga4 = BlockPGA(self.embedding_dims, self.embedding_dims, img_shape=(self.img_crop, self.img_crop))

        self.down1 = conv1x1(self.embedding_dims, self.embedding_dims, 1)
        self.down2 = conv1x1(self.embedding_dims, self.embedding_dims, 1)
        self.down3 = conv1x1(self.embedding_dims, self.embedding_dims, 1)

        self.outc = conv1x1(self.embedding_dims, self.n_classes, 1)

    def forward(self, x, obj_dict, bg_dict):
        x_pga1 = self.block_pga1(x, obj_dict, bg_dict)
        x = self.down1(x_pga1)

        x_pga2 = self.block_pga2(x, obj_dict, bg_dict)
        x = self.down2(x_pga2)

        x_pga3 = self.block_pga3(x, obj_dict, bg_dict)
        x = self.down3(x_pga3)

        x_pga4 = self.block_pga4(x, obj_dict, bg_dict)

        logits = self.outc(x_pga4)

        return logits


class BigOnlyPGA(nn.Module):
    def __init__(self, channels, n_classes, embedding_dims, img_crop=320):
        super(BigOnlyPGA, self).__init__()
        self.channels = channels
        self.n_classes = n_classes
        self.embedding_dims = embedding_dims
        self.img_crop = img_crop

        self.block_pga1 = BlockPGA(self.channels, self.embedding_dims, img_shape=(self.img_crop, self.img_crop))
        self.block_pga2 = BlockPGA(self.embedding_dims, self.embedding_dims, img_shape=(self.img_crop, self.img_crop))
        self.block_pga3 = BlockPGA(self.embedding_dims, self.embedding_dims, img_shape=(self.img_crop, self.img_crop))
        self.block_pga4 = BlockPGA(self.embedding_dims, self.embedding_dims, img_shape=(self.img_crop, self.img_crop))
        self.block_pga5 = BlockPGA(self.embedding_dims, self.embedding_dims, img_shape=(self.img_crop, self.img_crop))
        self.block_pga6 = BlockPGA(self.embedding_dims, self.embedding_dims, img_shape=(self.img_crop, self.img_crop))
        self.block_pga7 = BlockPGA(self.embedding_dims, self.embedding_dims, img_shape=(self.img_crop, self.img_crop))
        self.block_pga8 = BlockPGA(self.embedding_dims, self.embedding_dims, img_shape=(self.img_crop, self.img_crop))

        self.outc = conv1x1(self.embedding_dims, self.n_classes, 1)

    def forward(self, x, obj_dict, bg_dict):
        x = self.block_pga1(x, obj_dict, bg_dict)
        x = self.block_pga2(x, obj_dict, bg_dict)
        x = self.block_pga3(x, obj_dict, bg_dict)
        x = self.block_pga4(x, obj_dict, bg_dict)
        x = self.block_pga5(x, obj_dict, bg_dict)
        x = self.block_pga6(x, obj_dict, bg_dict)
        x = self.block_pga7(x, obj_dict, bg_dict)
        x = self.block_pga8(x, obj_dict, bg_dict)

        logits = self.outc(x)
        return logits