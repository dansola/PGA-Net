import os

from src.datasets.city import City
from torch.utils.data import DataLoader
import torch

from src.datasets.ice import BasicDatasetIce, Ice
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
acc_dict, iou_dict = {}, {}

for epoch in range(N_EPOCHS):
    log.info(f'Evaluating Epoch {epoch+1}')
    model = UNet(n_channels=3, n_classes=3, bilinear=True).to(device=device)
    # model = SmallUNet(n_channels=3, n_classes=3, bilinear=True).to(device=device)
    optimizer = optim.RMSprop(model.parameters(), lr=0.0001, weight_decay=1e-8, momentum=0.9)
    checkpoint_path = f'/home/dsola/repos/PGA-Net/checkpoints/frosty_sponge_239_unet_ice/epoch{epoch+1}.pth'
    # model, optimizer, _ = load_ckp(checkpoint_path, model, optimizer)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.train()
    out = nn.Softmax(dim=1)

    mask_list, pred_list = [], []

    for batch in tqdm(val_loader):
        img = batch['image'].to(device=device)
        mask = batch['mask'].to(device=device, dtype=torch.long)

        with torch.no_grad():
            output = model(img)
        sftmx = out(output)
        argmx = torch.argmax(sftmx, dim=1)

        mask_list.append(mask.squeeze(0))
        pred_list.append(argmx)

    masks = torch.stack(mask_list, dim=0)
    preds = torch.stack(pred_list, dim=0)

    hist = _fast_hist(masks.to(dtype=torch.long, device='cpu'), preds.to(dtype=torch.long, device='cpu'), 3)

    acc_dict[epoch+1] = per_class_pixel_accuracy(hist)[0].item()
    iou_dict[epoch + 1] = jaccard_index(hist)[0].item()

    del model
    del masks
    del preds
    del mask_list
    del pred_list
    del img
    del mask
    del output
    del sftmx
    del argmx
    torch.cuda.empty_cache()

model_name = checkpoint_path.split('/')[-2]

with open(f'../results/{model_name}-mean-acc-epoch.json', 'w') as fp:
    json.dump(acc_dict, fp)

with open(f'../results/{model_name}-mean-iou-epoch.json', 'w') as fp:
    json.dump(iou_dict, fp)