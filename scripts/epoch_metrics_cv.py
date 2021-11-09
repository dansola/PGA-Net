import argparse
import os

from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, lraspp_mobilenet_v3_large

from src.datasets.city import City
from torch.utils.data import DataLoader
import torch

from src.datasets.ice import Ice
from src.models.dsc.dsc_lbc_unet import DSCUNetLBP, DSCSmallUNetLBP, SkinnyDSCSmallUNetLBP
from src.models.dsc.dsc_unet import UNetDSC, SmallUNetDSC, SkinnySmallUNetDSC
from src.models.lbcnn.axial_lbcnn import AxialUNetLBC, SmallAxialUNetLBC
from src.models.lbcnn.axial_unet import AxialUNet, SmallAxialUNet
from src.models.lbcnn.lbc_unet import UNetLBP, SmallUNetLBP, SkinnySmallUNetLBP
from src.models.unet.unet_model import UNet, SmallUNet, SkinnySmallUNet
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
    parser.add_argument('-m', '--model', dest='model', type=str, default='unet',
                        help='Model to use.')
    parser.add_argument('-c', '--checkpoint', dest='checkpoint', type=str, default='tmp',
                        help='Checkpoint to use.')
    parser.add_argument('-e', '--evalset', dest='evalset', type=str, default='imgs',
                        help='Eval dataset to use.')
    parser.add_argument('-cv', '--cv', dest='cv', type=int, default=1,
                        help='Cross val set to use.')


    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    checkpoint = args.checkpoint

    log.info(f'Evaluating {args.model}')
    N_EPOCHS = 80

    data_dir = '/home/dsola/repos/PGA-Net/data/'
    batch_size = 1
    img_scale = 0.35
    img_crop = 256

    val_set = Ice(os.path.join(data_dir, args.evalset), os.path.join(data_dir, 'masks'),
                  os.path.join(data_dir, f'txt_files/cross_val/cv{args.cv}'), 'val', img_scale, img_crop)

    val_loader = DataLoader(val_set, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    acc_dict, iou_dict, loss_dict = {}, {}, {}

    criterion = nn.CrossEntropyLoss()

    for epoch in range(N_EPOCHS):
        if args.model == 'unet':
            net = UNet(n_channels=3, n_classes=3, bilinear=True)
        elif args.model == 'small_dsc_lbc_unet':
            net = SkinnyDSCSmallUNetLBP(3, 3, sparsity=0.8)
        elif args.model == 'lraspp_mobile_net':
            net = lraspp_mobilenet_v3_large(num_classes=3)
        else:
            raise ValueError('Please enter a valid model name.')

        net.to(device=device)
        log.info(f'Evaluating Epoch {epoch + 1}')
        checkpoint_path = f'/home/dsola/repos/PGA-Net/checkpoints/{checkpoint}/epoch{epoch + 1}.pth'
        net.load_state_dict(torch.load(checkpoint_path, map_location=device))
        net.train()
        out = nn.Softmax(dim=1)

        mask_list, pred_list, output_list = [], [], []

        for batch in tqdm(val_loader):
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
            output_list.append(output.squeeze(1))

        masks = torch.stack(mask_list, dim=0)
        preds = torch.stack(pred_list, dim=0)
        outputs = torch.stack(output_list, dim=0)

        hist = _fast_hist(masks.squeeze(2).to(dtype=torch.long, device='cpu'), preds.to(dtype=torch.long, device='cpu'), 3)

        acc_dict[epoch + 1] = per_class_pixel_accuracy(hist)[0].item()
        iou_dict[epoch + 1] = jaccard_index(hist)[0].item()
        loss_dict[epoch + 1] = criterion(outputs.squeeze(1), masks.squeeze(1)).item()

        del net
        del masks
        del preds
        del outputs
        del mask_list
        del pred_list
        del output_list
        del img
        del mask
        del output
        del sftmx
        del argmx
        torch.cuda.empty_cache()

    model_name = checkpoint_path.split('/')[-2]

    with open(f'../results/cv/{model_name}_cv_{args.cv}-mean-acc-epoch.json', 'w') as fp:
        json.dump(acc_dict, fp)

    with open(f'../results/cv/{model_name}_cv_{args.cv}-mean-iou-epoch.json', 'w') as fp:
        json.dump(iou_dict, fp)

    with open(f'../results/cv/{model_name}_cv_{args.cv}-mean-loss-epoch.json', 'w') as fp:
        json.dump(loss_dict, fp)