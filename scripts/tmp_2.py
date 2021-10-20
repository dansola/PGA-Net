from src.models.unet.unet_model import UNet
from torch.utils.data import DataLoader
import torch
from src.datasets.ice import Ice
from torch import nn
import os
import matplotlib.pyplot as plt

from src.metrics.segmentation import _fast_hist, per_class_pixel_accuracy, jaccard_index, fw_miou
from src.metrics.river_ice_metrics import pixel_accuracy, mean_accuracy, mean_IU, frequency_weighted_IU

net = UNet(n_channels=3, n_classes=3, bilinear=True)
checkpoint = 'frosty_sponge_239_unet_ice'
data_dir = '/home/dsola/repos/PGA-Net/data/'
batch_size = 1
img_scale = 0.35
img_crop = 256
epoch = 40

test_set = Ice(os.path.join(data_dir, 'imgs'), os.path.join(data_dir, 'masks'),
              os.path.join(data_dir, 'txt_files'), 'val', img_scale, img_crop)

test_loader = DataLoader(test_set, batch_size=batch_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net.to(device=device)

checkpoint_path = f'/home/dsola/repos/PGA-Net/checkpoints/{checkpoint}/epoch{epoch + 1}.pth'
net.load_state_dict(torch.load(checkpoint_path, map_location=device))
net.train()
out = nn.Softmax(dim=1)

batch = test_set[0]

img = batch['image'].to(device=device).unsqueeze(0)
mask = batch['mask'].to(device=device, dtype=torch.long).unsqueeze(0)

with torch.no_grad():
    output = net(img)
sftmx = out(output)
argmx = torch.argmax(sftmx, dim=1)

im = img.squeeze(0).permute(1,2,0).detach().cpu().numpy()[:,:,0]
pred = argmx.squeeze(0).detach().cpu().numpy()
gt = mask.squeeze(0).squeeze(0).detach().cpu().numpy()

frequency_weighted_IU(pred, gt, return_freq=1)