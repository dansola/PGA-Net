from torchvision import models
from torchsummary import summary
from src.models.lbcnn.axial_lbcnn import SmallAxialUNetLBC, AxialUNetLBC
from src.models.lbcnn.axial_unet import AxialUNet, SmallAxialUNet
from src.models.lbcnn.lbc_unet import UNetLBP, SmallUNetLBP
from src.models.unet.unet_model import UNet, SmallUNet
from src.models.dsc.dsc_lbc_unet import DSCSmallUNetLBP, DSCUNetLBP, SkinnyDSCSmallUNetLBP, SuperSkinnyDSCSmallUNetLBP
from src.models.dsc.dsc_unet import UNetDSC, SmallUNetDSC
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, lraspp_mobilenet_v3_large
import torch
import torch.nn as nn
from torch.autograd import Variable
import collections
from collections import OrderedDict
import numpy as np

# model = deeplabv3_mobilenet_v3_large(num_classes=3).to('cpu')
# model = DSCSmallUNetLBP(3, 3).to('cpu')
model = SkinnyDSCSmallUNetLBP(3, 3).to('cpu')
# model = SuperSkinnyDSCSmallUNetLBP(3, 3).to('cpu')
input_size = (3, 256, 256)
batch_size = -1
device = "cpu"

# summary(lraspp_mobile_net_model.to('cuda'), (3, 256, 256))
# summary(DSCSmallUNetLBP(3, 3).to('cuda'), (3, 256, 256))


def register_hook(module):
    def hook(module, input, output):
        class_name = str(module.__class__).split(".")[-1].split("'")[0]
        module_idx = len(summary)

        m_key = "%s-%i" % (class_name, module_idx + 1)
        summary[m_key] = OrderedDict()
        if isinstance(input[0], collections.OrderedDict):
            print(input[0].keys())
            summary[m_key]["input_shape"] = list(input[0]['out'].size())
        else:
            summary[m_key]["input_shape"] = list(input[0].size())
        summary[m_key]["input_shape"][0] = batch_size
        if isinstance(output, (list, tuple)):
            summary[m_key]["output_shape"] = [
                [-1] + list(o.size())[1:] for o in output
            ]
        else:
            if isinstance(output, collections.OrderedDict):
                print(output.keys())
                summary[m_key]["output_shape"] = list(output['out'].size())
            else:
                summary[m_key]["output_shape"] = list(output.size())
            summary[m_key]["output_shape"][0] = batch_size

        params = 0
        if hasattr(module, "weight") and hasattr(module.weight, "size"):
            params += torch.prod(torch.LongTensor(list(module.weight.size())))
            summary[m_key]["trainable"] = module.weight.requires_grad
        if hasattr(module, "bias") and hasattr(module.bias, "size"):
            params += torch.prod(torch.LongTensor(list(module.bias.size())))
        summary[m_key]["nb_params"] = params

    if (
        not isinstance(module, nn.Sequential)
        and not isinstance(module, nn.ModuleList)
        and not (module == model)
    ):
        hooks.append(module.register_forward_hook(hook))

device = device.lower()
assert device in [
    "cuda",
    "cpu",
], "Input device is not valid, please specify 'cuda' or 'cpu'"

if device == "cuda" and torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor

# multiple inputs to the network
if isinstance(input_size, tuple):
    input_size = [input_size]

# batch_size of 2 for batchnorm
x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
# print(type(x[0]))

# create properties
summary = OrderedDict()
hooks = []

# register hook
model.apply(register_hook)

# make a forward pass
# print(x.shape)
model(*x)

# remove these hooks
for h in hooks:
    h.remove()

print("----------------------------------------------------------------")
line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
print(line_new)
print("================================================================")
total_params = 0
total_output = 0
trainable_params = 0
for layer in summary:
    # input_shape, output_shape, trainable, nb_params
    line_new = "{:>20}  {:>25} {:>15}".format(
        layer,
        str(summary[layer]["output_shape"]),
        "{0:,}".format(summary[layer]["nb_params"]),
    )
    total_params += summary[layer]["nb_params"]
    total_output += np.prod(summary[layer]["output_shape"])
    if "trainable" in summary[layer]:
        if summary[layer]["trainable"] == True:
            trainable_params += summary[layer]["nb_params"]
    print(line_new)

# assume 4 bytes/number (float on cuda).
total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
total_size = total_params_size + total_output_size + total_input_size

print("================================================================")
print("Total params: {0:,}".format(total_params))
print("Trainable params: {0:,}".format(trainable_params))
print("Non-trainable params: {0:,}".format(total_params - trainable_params))
print("----------------------------------------------------------------")
print("Input size (MB): %0.2f" % total_input_size)
print("Forward/backward pass size (MB): %0.2f" % total_output_size)
print("Params size (MB): %0.2f" % total_params_size)
print("Estimated Total Size (MB): %0.2f" % total_size)
print("----------------------------------------------------------------")
