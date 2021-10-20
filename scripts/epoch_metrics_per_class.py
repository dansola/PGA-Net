import argparse
import os

from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, lraspp_mobilenet_v3_large

from src.datasets.city import City
from torch.utils.data import DataLoader
import torch

from src.datasets.ice import Ice
from src.models.dsc.dsc_lbc_unet import DSCUNetLBP, DSCSmallUNetLBP
from src.models.dsc.dsc_unet import UNetDSC, SmallUNetDSC
from src.models.lbcnn.axial_lbcnn import AxialUNetLBC, SmallAxialUNetLBC
from src.models.lbcnn.axial_unet import AxialUNet, SmallAxialUNet
from src.models.lbcnn.lbc_unet import UNetLBP, SmallUNetLBP
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
    parser.add_argument('-m', '--model', dest='model', type=str, default='deeplab_mobile_net',
                        help='Model to use.')

    return parser.parse_args()


def get_model_and_checkpoint(model_name):
    if model_name == 'unet':
        net = UNet(n_channels=3, n_classes=3, bilinear=True)
        checkpoint = 'frosty_sponge_239_unet_ice'
    elif model_name == 'small_unet':
        net = SmallUNet(n_channels=3, n_classes=3, bilinear=True)
        checkpoint = 'laced_disco_238_small_unet_ice'
    elif model_name == 'axial_unet':
        net = AxialUNet(3, 3, 64)
    elif model_name == 'small_axial_unet':
        net = SmallAxialUNet(3, 3, 64)
    elif model_name == 'lbc_unet':
        net = UNetLBP(3, 3)
        checkpoint = 'worthy_wood_227_lbc_unet_ice'
    elif model_name == 'small_lbc_unet':
        net = SmallUNetLBP(3, 3)
        checkpoint = 'sandy_moon_228_small_lbc_unet_ice'
    elif model_name == 'axial_lbc_unet':
        net = AxialUNetLBC(3, 3, 32)
    elif model_name == 'small_axial_lbc_unet':
        net = SmallAxialUNetLBC(3, 3, 32)
    elif model_name == 'small_axial_lbc_unet_10':
        net = SmallAxialUNetLBC(3, 3, 10)
    elif model_name == 'deeplab_mobile_net':
        net = deeplabv3_mobilenet_v3_large(num_classes=3)
        checkpoint = 'vivid_donkey_305_deeplab_mobilenet_ice'
    elif model_name == 'lraspp_mobile_net':
        net = lraspp_mobilenet_v3_large(num_classes=3)
        checkpoint = 'clean_darkness_312_laraspp_mobilenet_ice'
    elif model_name == 'dsc_unet':
        net = UNetDSC(n_channels=3, n_classes=3, bilinear=True)
        checkpoint = 'helpful_fire_308_unet_dsc_ice'
    elif model_name == 'small_dsc_unet':
        net = SmallUNetDSC(n_channels=3, n_classes=3, bilinear=True)
        checkpoint = 'cosmic_waterfall_315_small_dsc_unet_ice'
    elif model_name == 'dsc_lbc_unet':
        net = DSCUNetLBP(3, 3)
        checkpoint = 'youthful_microwave_310_lbc_dsc_unet_ice'
    elif model_name == 'small_dsc_lbc_unet':
        net = DSCSmallUNetLBP(3, 3)
        checkpoint = 'still_paper_311_small_lbc_dsc_unet_ice'
    else:
        raise ValueError('Please enter a valid model name.')
    return net, checkpoint


if __name__ == '__main__':
    args = get_args()

    log.info(f'Evaluating {args.model}')
    N_EPOCHS = 80

    data_dir = '/home/dsola/repos/PGA-Net/data/'
    batch_size = 1
    img_scale = 0.35
    img_crop = 256

    test_set = Ice(os.path.join(data_dir, 'imgs'), os.path.join(data_dir, 'masks'),
                  os.path.join(data_dir, 'txt_files'), 'val', img_scale, img_crop)

    test_loader = DataLoader(test_set, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    acc_dict, iou_dict = {}, {}

    for epoch in range(N_EPOCHS):
        net, checkpoint = get_model_and_checkpoint(args.model)
        net.to(device=device)
        log.info(f'Evaluating Epoch {epoch + 1}')
        checkpoint_path = f'/home/dsola/repos/PGA-Net/checkpoints/{checkpoint}/epoch{epoch + 1}.pth'
        net.load_state_dict(torch.load(checkpoint_path, map_location=device))
        net.train()
        out = nn.Softmax(dim=1)

        mask_list, pred_list = [], []

        for batch in tqdm(test_loader):
            img = batch['image'].to(device=device)
            mask = batch['mask'].to(device=device, dtype=torch.long)

            with torch.no_grad():
                if 'mobile' in args.model:
                    output = net(img)['out']
                else:
                    output = net(img)
            sftmx = out(output)
            argmx = torch.argmax(sftmx, dim=1)

            mask_list.append(mask.squeeze(0))
            pred_list.append(argmx)

        masks = torch.stack(mask_list, dim=0)
        preds = torch.stack(pred_list, dim=0)

        hist = _fast_hist(masks.to(dtype=torch.long, device='cpu'), preds.to(dtype=torch.long, device='cpu'), 3)

        l_acc_temp = per_class_pixel_accuracy(hist)[1].tolist()
        l_acc_temp.append(per_class_pixel_accuracy(hist)[0].item())
        acc_dict[epoch + 1] = l_acc_temp
        l_iou_temp = jaccard_index(hist)[1].tolist()
        l_iou_temp.append(jaccard_index(hist)[0].item())
        iou_dict[epoch + 1] = l_iou_temp
        #     sftmx = out(output)
        #     argmx = torch.argmax(sftmx, dim=1)
        #
        #     mask_list.append(mask.squeeze(0))
        #     pred_list.append(argmx)
        #
        # masks = torch.stack(mask_list, dim=0)
        # preds = torch.stack(pred_list, dim=0)
        #
        # hist = _fast_hist(masks.squeeze(2).to(dtype=torch.long, device='cpu'), preds.to(dtype=torch.long, device='cpu'), 3)
        #
        # l_acc_temp = per_class_pixel_accuracy(hist)[1].tolist()
        # l_acc_temp.append(per_class_pixel_accuracy(hist)[0].item())
        # acc_dict[epoch + 1] = l_acc_temp
        # l_iou_temp = jaccard_index(hist)[1].tolist()
        # l_iou_temp.append(jaccard_index(hist)[0].item())
        # iou_dict[epoch + 1] = l_iou_temp

        del net
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

    with open(f'../results/val_set_per_class/{model_name}-mean-acc-epoch.json', 'w') as fp:
        json.dump(acc_dict, fp)

    with open(f'../results/val_set_per_class/{model_name}-mean-iou-epoch.json', 'w') as fp:
        json.dump(iou_dict, fp)