import os
import sys

import time

from src.models.dsc.dsc_lbc_unet import DSCUNetLBP, DSCSmallUNetLBP
from src.models.dsc.dsc_unet import UNetDSC, SmallUNetDSC
from src.models.lbcnn.axial_lbcnn import AxialUNetLBC, SmallAxialUNetLBC
from src.models.lbcnn.axial_unet import AxialUNet, SmallAxialUNet
from src.models.lbcnn.lbc_unet import UNetLBP, SmallUNetLBP
from src.models.unet.unet_model import UNet, SmallUNet
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
from src.eval.eval_curves import eval_net
from src.eval.eval_mobilenet import eval_net as eval_net_mobile
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
    parser.add_argument('-m', '--model', dest='model', type=str, default='unet',
                        help='Model to use.')
    parser.add_argument('-dev', '--device', dest='device', type=str, default='cuda',
                        help='Train on gpu vs cpu.')
    parser.add_argument('-file', '--file_number', dest='file_number', type=str, default='0',
                        help='Suffix number of output file.')

    return parser.parse_args()


def train_net(net, data_dir, device, name, epochs=20, batch_size=1, lr=0.0001, save_cp=True, img_scale=0.35,
              img_crop=320):
    accs, ious, losses = [], [], []

    train_set = Ice(os.path.join(data_dir, 'imgs'), os.path.join(data_dir, 'masks'),
                    os.path.join(data_dir, 'txt_files'), 'train', img_scale, img_crop)
    val_set = Ice(os.path.join(data_dir, 'imgs'), os.path.join(data_dir, 'masks'),
                  os.path.join(data_dir, 'txt_files'), 'val', img_scale, img_crop)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    global_step = 0

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=len(train_set), desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']

                imgs = imgs.to(device=device, dtype=torch.float32)
                target = true_masks.to(device=device, dtype=torch.long)

                if 'mobile' in args.model:
                    masks_pred = net(imgs)['out']
                else:
                    masks_pred = net(imgs)

                probs = F.softmax(masks_pred, dim=1)
                argmx = torch.argmax(probs, dim=1).to(dtype=torch.float32)
                loss = criterion(masks_pred, target.squeeze(1))
                epoch_loss += loss.item()

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                if len(train_set) > 10:
                    n = 10
                else:
                    n = 1
                if global_step % (len(train_set) // (n * batch_size)) == 0:
                    if 'mobile' in name:
                        val_loss, val_iou, val_acc = eval_net_mobile(net, val_loader, device)
                    else:
                        val_loss, val_iou, val_acc = eval_net(net, val_loader, device)
                    accs.append(val_acc.item())
                    ious.append(val_iou.item())
                    losses.append(val_loss)
                    # scheduler.step(val_loss)

        if save_cp:
            try:
                os.mkdir('../checkpoints/')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       '../checkpoints/' + f'epoch{epoch + 1}.pth')
    return accs, ious, losses


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
    else:
        raise ValueError('Please enter a valid model name.')

    log.info(f'Training {args.model}.')
    # wandb.watch(net)

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))

    net.to(device=device)

    try:
        accs, ious, losses = train_net(net=net, data_dir=args.data_dir, epochs=args.epochs, batch_size=args.batchsize,
                                       lr=args.lr, device=device, img_scale=args.scale, img_crop=args.crop,
                                       name=args.model)
        curve_dict = {'loss': losses,
                      'iou': ious,
                      'acc': accs}

        if os.path.exists(f'model_curves.json'):
            with open(f'model_curves.json') as f:
                data = json.load(f)
            data[args.model] = curve_dict
            with open(f'model_curves.json', 'w') as outfile:
                json.dump(data, outfile)
        else:
            data = {args.model: curve_dict}
            with open(f'model_curves.json', 'w') as outfile:
                json.dump(data, outfile)

    except KeyboardInterrupt:
        torch.save(net.state_dict(), '../INTERRUPTED.pth')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
