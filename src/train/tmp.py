from time import time

import torch
from torch import nn
from torch.nn import Sequential

from src.models.dsc.dsc_unet import UNetDSC
from src.models.lbcnn.axial_lbcnn import SmallAxialUNetLBC
from src.models.lbcnn.axial_unet import AxialUNet
from src.models.lbcnn.lbc_unet import UNetLBP, SmallUNetLBP
from src.models.unet.unet_model import UNet
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101, lraspp_mobilenet_v3_large, deeplabv3_mobilenet_v3_large


def net_time_test(net, times, mobile=False):
    duration = 0
    for i in range(times):
        if mobile:
            out = net(input_)['out']
        else:
            out = net(input_)
        start = time()
        out.backward(gradient=torch.randn(1, 3, 256, 256).to(device=device))
        end = time()
        duration += end - start
    print(duration)

device = 'cpu'

big_net = UNet(n_channels=3, n_classes=3, bilinear=True).to(device=device)
big_net_dsc = UNetDSC(n_channels=3, n_classes=3, bilinear=True).to(device=device)
# small_net = AxialUNet(3, 3, 10).to(device=device)
small_net = UNetLBP(3, 3).to(device=device)
# small_net = SmallAxialUNetLBC(3, 3, 10).to(device=device)
mobile_net = lraspp_mobilenet_v3_large(num_classes=3).to(device=device)


input_ = torch.randn(1, 3, 256, 256).to(device=device)

net_time_test(big_net, 5)
net_time_test(small_net, 5)
net_time_test(mobile_net, 5, mobile=True)
net_time_test(big_net_dsc, 5)
