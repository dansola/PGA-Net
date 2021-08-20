from torch import nn
from torchvision.models.segmentation import lraspp_mobilenet_v3_large, deeplabv3_mobilenet_v3_large

lraspp_mobilenet_v3_large_one_channel = lraspp_mobilenet_v3_large(num_classes=2)
lraspp_mobilenet_v3_large_one_channel.backbone._modules['0']._modules['0'] = nn.Conv2d(1, 16, kernel_size=(3, 3),
                                                                                       stride=(2, 2), padding=(1, 1),
                                                                                       bias=False)

deeplabv3_mobilenet_v3_large_one_channel = deeplabv3_mobilenet_v3_large(num_classes=2)
deeplabv3_mobilenet_v3_large_one_channel.backbone._modules['0']._modules['0'] = nn.Conv2d(1, 16, kernel_size=(3, 3),
                                                                                          stride=(2, 2), padding=(1, 1),
                                                                                          bias=False)
