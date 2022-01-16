import os
import sys

import time

from src.models.dsc.dsc_lbc_unet import DSCUNetLBP, DSCSmallUNetLBP, SkinnyDSCSmallUNetLBP
from src.models.dsc.dsc_unet import UNetDSC, SmallUNetDSC, SkinnySmallUNetDSC
from src.models.lbcnn.axial_lbcnn import AxialUNetLBC, SmallAxialUNetLBC
from src.models.lbcnn.axial_unet import AxialUNet, SmallAxialUNet
from src.models.lbcnn.lbc_unet import UNetLBP, SmallUNetLBP, SkinnySmallUNetLBP
from src.models.unet.unet_model import UNet, SmallUNet, SkinnySmallUNet
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, lraspp_mobilenet_v3_large

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm
from src.eval.eval_axial import eval_net
from src.datasets.ice import Ice
from torch.utils.data import DataLoader
# import wandb
import json
from loguru import logger as log

# wandb.init()


def get_args():
    parser = argparse.ArgumentParser(description='Train AxialUnet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data_directory', metavar='D', type=str, default='/home/dsola/repos/PGA-Net/data/',
                        help='Directory where images, masks, and txt files reside.', dest='data_dir')
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=2,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.35,
                        help='Downscaling factor of the images')
    parser.add_argument('-c', '--crop', dest='crop', type=int, default=256,
                        help='Height and width of images and masks.')
    parser.add_argument('-m', '--model', dest='model', type=str, default='deeplab_mobile_net',
                        help='Model to use.')
    parser.add_argument('-dev', '--device', dest='device', type=str, default='cuda',
                        help='Train on gpu vs cpu.')
    parser.add_argument('-file', '--file_number', dest='file_number', type=str, default='0',
                        help='Suffix number of output file.')

    return parser.parse_args()


def eval_net(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.TRAIN()
    n_val = len(loader)
    tot_loss, tot_iou, tot_acc = 0, 0, 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, _ = batch['image'], batch['mask']

            imgs = imgs.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                mask_pred = net(imgs)


if __name__ == '__main__':
    args = get_args()
    device = args.device

    if args.model == 'unet':
        net = UNet(n_channels=3, n_classes=3, bilinear=True)
    elif args.model == 'small_unet':
        net = SmallUNet(n_channels=3, n_classes=3, bilinear=True)
    elif args.model == 'axial_unet':
        net = AxialUNet(3, 3, 64)
    elif args.model == 'small_axial_unet':
        net = SmallAxialUNet(3, 3, 64)
    elif args.model == 'lbc_unet':
        net = UNetLBP(3, 3)
    elif args.model == 'small_lbc_unet':
        net = SmallUNetLBP(3, 3)
    elif args.model == 'axial_lbc_unet':
        net = AxialUNetLBC(3, 3, 32)
    elif args.model == 'small_axial_lbc_unet':
        net = SmallAxialUNetLBC(3, 3, 32)
    elif args.model == 'small_axial_lbc_unet_10':
        net = SmallAxialUNetLBC(3, 3, 10)
    elif args.model == 'deeplab_mobile_net':
        net = deeplabv3_mobilenet_v3_large(num_classes=3)
    elif args.model == 'lraspp_mobile_net':
        net = lraspp_mobilenet_v3_large(num_classes=3)
    elif args.model == 'dsc_unet':
        net = UNetDSC(n_channels=3, n_classes=3, bilinear=True)
    elif args.model == 'small_dsc_unet':
        net = SmallUNetDSC(n_channels=3, n_classes=3, bilinear=True)
    elif args.model == 'dsc_lbc_unet':
        net = DSCUNetLBP(3, 3)
    elif args.model == 'small_dsc_lbc_unet':
        net = DSCSmallUNetLBP(3, 3)
    elif args.model == 'skinny_small_dsc_lbc_unet':
        net = SkinnyDSCSmallUNetLBP(3, 3)
    elif args.model == 'skinny_small_unet':
        net = SkinnySmallUNet(n_channels=3, n_classes=3, bilinear=True)
    elif args.model == 'skinny_small_dsc_unet':
        net = SkinnySmallUNetDSC(n_channels=3, n_classes=3, bilinear=True)
    elif args.model == 'skinny_small_lbc_unet':
        net = SkinnySmallUNetLBP(3, 3)
    else:
        raise ValueError('Please enter a valid model name.')

    log.info(f'Training {args.model}.')
    # wandb.watch(net)

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))

    net.to(device=device)
    img_scale = 0.35
    img_crop = 320
    batch_size = 2

    val_set = Ice(os.path.join(args.data_dir, 'imgs'), os.path.join(args.data_dir, 'masks'),
                  os.path.join(args.data_dir, 'txt_files'), 'val', img_scale, img_crop)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    try:
        st = time.time()
        eval_net(net, val_loader, device)
        run_time = time.time() - st

        if os.path.exists(f'./times/model_profile_{args.device}_v{args.file_number}.json'):
            with open(f'./times/model_profile_{args.device}_v{args.file_number}.json') as f:
                data = json.load(f)
            data[args.model] = run_time
            with open(f'./times/model_profile_{args.device}_v{args.file_number}.json', 'w') as outfile:
                json.dump(data, outfile)
        else:
            data = {args.model: run_time}
            with open(f'./times/model_profile_{args.device}_v{args.file_number}.json', 'w') as outfile:
                json.dump(data, outfile)

        # with torch.autograd.profiler.profile(use_cuda=True) as prof:
        #     train_net(net=net, data_dir=args.data_dir, epochs=args.epochs, batch_size=args.batchsize, lr=args.lr,
        #               device=device,
        #               img_scale=args.scale, img_crop=args.crop)
        # # print(prof)
        # cpu_time = float(str(prof).split('\n')[-3].split(' ')[-1][:-1])
        # cuda_time = float(str(prof).split('\n')[-2].split(' ')[-1][:-1])
        #
        # print(f'CPU Time: {cpu_time}, Cuda Time: {cuda_time}')
        #
        # if os.path.exists(f'model_profile_{args.device}_v2.json'):
        #     with open(f'model_profile_{args.device}_v2.json') as f:
        #         data = json.load(f)
        #     data[args.model] = {'cpu_time': cpu_time, 'cuda_time': cuda_time}
        #     with open(f'model_profile_{args.device}_v2.json', 'w') as outfile:
        #         json.dump(data, outfile)
        # else:
        #     data = {args.model: {'cpu_time': cpu_time, 'cuda_time': cuda_time}}
        #     with open(f'model_profile_{args.device}_v2.json', 'w') as outfile:
        #         json.dump(data, outfile)

    except KeyboardInterrupt:
        torch.save(net.state_dict(), '../INTERRUPTED.pth')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
