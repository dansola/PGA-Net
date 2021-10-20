import os

from torchvision.models.segmentation import lraspp_mobilenet_v3_large

from src.datasets.city import City
from torch.utils.data import DataLoader
import torch
from src.metrics.river_ice_metrics import pixel_accuracy, mean_accuracy, mean_IU, frequency_weighted_IU
from src.datasets.ice import BasicDatasetIce, Ice
from src.models.dsc.dsc_lbc_unet import SkinnyDSCSmallUNetLBP
from src.models.unet.unet_model import UNet, SmallUNet
from loguru import logger as log
from torch import nn
import json
import matplotlib.pyplot as plt
import numpy as np
from src.metrics.segmentation import _fast_hist, per_class_pixel_accuracy, jaccard_index
from tqdm import tqdm
from torch import optim

from src.train.utils import load_ckp

N_EPOCHS = 80

data_dir = '/home/dsola/repos/PGA-Net/data/'
batch_size = 1
img_scale = 0.35
img_crop = 256

# train_set = City(data_dir, split='train', is_transform=True, img_size=(128, 256))
# val_set = City(data_dir, split='val', is_transform=True, img_size=(128, 256))
# train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
# val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True,
#                         drop_last=True)

train_set = Ice(os.path.join(data_dir, 'imgs'), os.path.join(data_dir, 'masks'),
                os.path.join(data_dir, 'txt_files'), 'train', img_scale, img_crop)
val_set = Ice(os.path.join(data_dir, 'imgs'), os.path.join(data_dir, 'masks'),
              os.path.join(data_dir, 'txt_files'), 'val', img_scale, img_crop)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pre_dict, rec_dict = {}, {}

for epoch in range(N_EPOCHS):
    log.info(f'Evaluating Epoch {epoch+1}')
    # model = UNet(n_channels=3, n_classes=3, bilinear=True).to(device=device)
    model = SkinnyDSCSmallUNetLBP(3, 3, sparsity=0.8).to(device=device)
    # model = lraspp_mobilenet_v3_large(num_classes=3).to(device=device)
    optimizer = optim.RMSprop(model.parameters(), lr=0.0001, weight_decay=1e-8, momentum=0.9)
    # checkpoint_path = f'/home/dsola/repos/PGA-Net/checkpoints/honest_salad_428_unet_ice_original_split/epoch{epoch+1}.pth'
    checkpoint_path = f'/home/dsola/repos/PGA-Net/checkpoints/lilac_sky_429_skinny_dsc_lbc_unet_ice_original_split/epoch{epoch+1}.pth'
    # checkpoint_path = f'/home/dsola/repos/PGA-Net/checkpoints/peach_glitter_436_skinny_dsc_lbc_ice_original_split/epoch{epoch+1}.pth'
    # checkpoint_path = f'/home/dsola/repos/PGA-Net/checkpoints/fresh_vally_433_lraspp_mobilenet_ice_original_split/epoch{epoch+1}.pth'
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.train()
    out = nn.Softmax(dim=1)

    mask_list, pred_list = [], []
    avg_mean_IU_ice = avg_mean_IU_ice_1 = avg_mean_IU_ice_2 = 0
    avg_mean_acc_ice = avg_mean_acc_ice_1 = avg_mean_acc_ice_2 = 0
    fw_iou_list, mean_iu_list = [], []
    mean_acc_list, pix_acc_list = [], []


    for i, batch in enumerate(val_loader):
        img = batch['image'].to(device=device)
        mask = batch['mask'].to(device=device, dtype=torch.long)

        with torch.no_grad():
            output = model(img)#['out']
        sftmx = out(output)
        argmx = torch.argmax(sftmx, dim=1)

        mask_list.append(mask.squeeze(0))
        pred_list.append(argmx)

        im = img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()[:, :, 0]
        pred = argmx.squeeze(0).detach().cpu().numpy()
        gt = mask.squeeze(0).squeeze(0).detach().cpu().numpy()

        fw_iou = frequency_weighted_IU(pred, gt)
        fw_iou_list.append(fw_iou)

        mean_iu = mean_IU(pred, gt)
        mean_iu_list.append(mean_iu)

        mean_acc = mean_accuracy(pred, gt)
        mean_acc_list.append(mean_acc)

        pix_acc = pixel_accuracy(pred, gt)
        pix_acc_list.append(pix_acc)

        _acc, _ = mean_accuracy(pred, gt, return_acc=1)
        mean_acc_ice = np.mean(list(_acc.values())[1:])
        avg_mean_acc_ice += (mean_acc_ice - avg_mean_acc_ice) / (i + 1)
        mean_acc_ice_1 = _acc[1]
        avg_mean_acc_ice_1 += (mean_acc_ice_1 - avg_mean_acc_ice_1) / (i + 1)
        mean_acc_ice_2 = _acc[2]
        avg_mean_acc_ice_2 += (mean_acc_ice_2 - avg_mean_acc_ice_2) / (i + 1)

        _IU, _ = mean_IU(pred, gt, return_iu=1)
        mean_IU_ice = np.mean(list(_IU.values())[1:])
        avg_mean_IU_ice += (mean_IU_ice - avg_mean_IU_ice) / (i + 1)
        mean_IU_ice_1 = _IU[1]
        avg_mean_IU_ice_1 += (mean_IU_ice_1 - avg_mean_IU_ice_1) / (i + 1)
        mean_IU_ice_2 = _IU[2]
        avg_mean_IU_ice_2 += (mean_IU_ice_2 - avg_mean_IU_ice_2) / (i + 1)

    pre_dict[epoch+1] = [avg_mean_acc_ice_1, avg_mean_acc_ice_2, np.mean(mean_acc_list), np.mean(pix_acc_list)]
    rec_dict[epoch + 1] = [avg_mean_IU_ice_1, avg_mean_IU_ice_2, np.mean(mean_iu_list), np.mean(fw_iou_list)]

    del model
    del mask_list
    del pred_list
    del img
    del mask
    del output
    del sftmx
    del argmx
    torch.cuda.empty_cache()

model_name = checkpoint_path.split('/')[-2]

with open(f'../results/original_test/{model_name}-mean-acc-epoch.json', 'w') as fp:
    json.dump(pre_dict, fp)

with open(f'../results/original_test/{model_name}-mean-iou-epoch.json', 'w') as fp:
    json.dump(rec_dict, fp)