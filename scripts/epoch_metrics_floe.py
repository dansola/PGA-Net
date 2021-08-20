import argparse
import os
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, lraspp_mobilenet_v3_large

from src.datasets.city import City
from torch.utils.data import DataLoader
import torch

from src.datasets.floe import DatasetValidateFloe
from src.datasets.ice import Ice
from src.models.dsc.dsc_lbc_unet import DSCUNetLBP, DSCSmallUNetLBP
from src.models.dsc.dsc_unet import UNetDSC, SmallUNetDSC
from src.models.lbcnn.axial_lbcnn import AxialUNetLBC, SmallAxialUNetLBC
from src.models.lbcnn.axial_unet import AxialUNet, SmallAxialUNet
from src.models.lbcnn.lbc_unet import UNetLBP, SmallUNetLBP
from src.models.mobilenets import lraspp_mobilenet_v3_large_one_channel
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

def get_args():
    parser = argparse.ArgumentParser(description='get metrics for test.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model', dest='model', type=str, default='lraspp_mobile_net',
                        help='Model to use.')

    return parser.parse_args()


def get_model_and_checkpoint(model_name):
    if model_name == 'unet':
        net = UNet(n_channels=1, n_classes=2, bilinear=True)
        checkpoint = 'glowing_armadillo_326_unet_floe'
        e = 19
    elif model_name == 'small_unet':
        net = SmallUNet(n_channels=1, n_classes=2, bilinear=True)
        checkpoint = 'playful_snowball_331_small_unet_floe'
        e = 20
    elif model_name == 'lbc_unet':
        net = UNetLBP(n_channels=1, n_classes=2)
        checkpoint = 'royal_darkness_330_lbc_unet_floe'
        e = 20
    elif model_name == 'small_lbc_unet':
        net = SmallUNetLBP(n_channels=1, n_classes=2)
        checkpoint = 'elated_violet_333_small_lbc_unet_floe'
        e = 20
    elif model_name == 'lraspp_mobile_net':
        net = lraspp_mobilenet_v3_large_one_channel
        checkpoint = 'vocal_brook_343_lraspp_floe'
        e = 20
    elif model_name == 'dsc_unet':
        net = UNetDSC(n_channels=1, n_classes=2, bilinear=True)
        checkpoint = 'smart_silence_327_dsc_unet_floe'
        e = 35
    elif model_name == 'small_dsc_unet':
        net = SmallUNetDSC(n_channels=1, n_classes=2, bilinear=True)
        checkpoint = 'polar_shadow_332_small_dsc_unet_floe'
        e = 20
    elif model_name == 'dsc_lbc_unet':
        net = DSCUNetLBP(n_channels=1, n_classes=2)
        checkpoint = 'gentle_yogurt_328_dsc_unet_lbp_floe'
        e = 23
    elif model_name == 'small_dsc_lbc_unet':
        net = DSCSmallUNetLBP(n_channels=1, n_classes=2)
        checkpoint = 'dulcet_meadow_329_dsc_lbc_small_unet_floe'
        e = 70
    else:
        raise ValueError('Please enter a valid model name.')
    return net, checkpoint, e


if __name__ == '__main__':
    args = get_args()

    log.info(f'Evaluating {args.model}')

    batch_size = 1
    val_set = DatasetValidateFloe()
    val_loader = DataLoader(val_set, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    acc_dict, iou_dict = {}, {}
    _, _, e = get_model_and_checkpoint(args.model)

    for epoch in range(e):
        net, checkpoint, _ = get_model_and_checkpoint(args.model)
        net.to(device=device)
        log.info(f'Evaluating Epoch {epoch + 1}')
        checkpoint_path = f'/home/dsola/repos/PGA-Net/checkpoints/{checkpoint}/epoch{epoch + 1}.pth'
        net.load_state_dict(torch.load(checkpoint_path, map_location=device))
        net.train()
        out = nn.Softmax(dim=1)

        mask_list, pred_list = [], []
        tot_iou, tot_acc = 0, 0

        for batch in tqdm(val_loader):
            img = batch['image'][:, 0, :, :].unsqueeze(1).to(device=device, dtype=torch.float32)
            mask = batch['mask'].to(device=device, dtype=torch.long)

            with torch.no_grad():
                if 'mobile' in args.model:
                    output = net(img)['out']
                else:
                    output = net(img)
            sftmx = out(output)
            argmx = torch.argmax(sftmx, dim=1).to(dtype=torch.float32)

            hist = _fast_hist(mask.squeeze(0).squeeze(0), argmx.squeeze(0).to(dtype=torch.long), 2)

            tot_iou += jaccard_index(hist)[0]
            tot_acc += per_class_pixel_accuracy(hist)[0]

        acc_dict[epoch + 1] = tot_acc / len(val_loader)
        iou_dict[epoch + 1] = tot_iou / len(val_loader)

        del net
        del mask_list
        del pred_list
        del img
        del mask
        del output
        del sftmx
        del argmx
        torch.cuda.empty_cache()

    model_name = checkpoint_path.split('/')[-2]

    with open(f'../results/floe/val_set/{model_name}-mean-acc-epoch.json', 'w') as fp:
        json.dump(acc_dict, fp)

    with open(f'../results/floe/val_set/{model_name}-mean-iou-epoch.json', 'w') as fp:
        json.dump(iou_dict, fp)