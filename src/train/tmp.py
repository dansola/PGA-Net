from time import time

import torch
from torch import nn
from torch.nn import Sequential

from src.models.lbcnn.axial_lbcnn import SmallAxialUNetLBC
from src.models.unet.unet_model import UNet


def net_time_test(net, times):
    duration = 0
    for i in range(times):
        out = net(input_)
        start = time()
        out.backward(gradient=torch.randn(1, 3, 256, 256))
        end = time()
        duration += end - start
    print(duration)


big_net = UNet(n_channels=3, n_classes=3, bilinear=True)
small_net = SmallAxialUNetLBC(3, 3, 10)

input_ = torch.randn(1, 3, 256, 256)

print(net_time_test(big_net, 1))
print(net_time_test(small_net, 1))
