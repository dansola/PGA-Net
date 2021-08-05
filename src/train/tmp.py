from time import time

import torch
from torch import nn
from torch.nn import Sequential

from src.models.dsc.dsc_lbc_unet import DSCUNetLBP, DSCSmallUNetLBP
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
        out.backward(gradient=torch.randn(2, 3, 64, 64).to(device=device))
        end = time()
        duration += end - start
    print(duration)

device = 'cpu'

# big_net = UNet(n_channels=3, n_classes=3, bilinear=True).to(device=device)
# big_net_dsc = UNetDSC(n_channels=3, n_classes=3, bilinear=True).to(device=device)
# # small_net = AxialUNet(3, 3, 10).to(device=device)
# small_net = SmallUNetLBP(3, 3).to(device=device)
# # small_net = SmallAxialUNetLBC(3, 3, 10).to(device=device)
# mobile_net = deeplabv3_mobilenet_v3_large(num_classes=3).to(device=device)

unet_model = UNet(n_channels=3, n_classes=3, bilinear=True)
axial_unet_model = AxialUNet(3, 3, 64)
lbc_unet_model = UNetLBP(3, 3)
small_axial_lbc_unet_model = SmallAxialUNetLBC(3, 3, 10)
deeplab_mobile_net_model = deeplabv3_mobilenet_v3_large(num_classes=3)
lraspp_mobile_net_model = lraspp_mobilenet_v3_large(num_classes=3)
dsc_unet_model = UNetDSC(n_channels=3, n_classes=3, bilinear=True)
dsc_lbc_unet_model = DSCUNetLBP(3, 3)
small_dsc_lbc_unet_model = DSCSmallUNetLBP(3, 3)


input_ = torch.randn(2, 3, 64, 64).to(device=device)

net_time_test(unet_model, 50)
# net_time_test(axial_unet_model, 5)
net_time_test(lbc_unet_model, 50)
net_time_test(small_axial_lbc_unet_model, 50)
net_time_test(deeplab_mobile_net_model, 50, mobile=True)
net_time_test(lraspp_mobile_net_model, 50, mobile=True)
net_time_test(dsc_unet_model, 50)
net_time_test(dsc_lbc_unet_model, 50)
net_time_test(small_dsc_lbc_unet_model, 50)
