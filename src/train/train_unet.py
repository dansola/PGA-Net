import os
import sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm
from src.eval.eval_unet import eval_net
from src.models.unet.unet_model import UNet
from src.datasets.ice import BasicDatasetIce
from torch.utils.data import DataLoader
import wandb

wandb.init()


def get_args():
    parser = argparse.ArgumentParser(description='Train AxialUnet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data_directory', metavar='D', type=str, default='../data',
                        help='Directory where images, masks, and txt files reside.', dest='data_dir')
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=20,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.35,
                        help='Downscaling factor of the images')
    parser.add_argument('-c', '--crop', dest='crop', type=int, default=220,
                        help='Height and width of images and masks.')

    return parser.parse_args()


def train_net(net, data_dir, device, epochs=20, batch_size=1, lr=0.0001, save_cp=True, img_scale=0.35, img_crop=320):
    dir_img = os.path.join(data_dir, 'imgs')
    dir_mask = os.path.join(data_dir, 'masks')
    dir_txt = os.path.join(data_dir, 'txt_files')
    train_set = BasicDatasetIce(dir_img, dir_mask, dir_txt, 'train', img_scale)
    val_set = BasicDatasetIce(dir_img, dir_mask, dir_txt, 'val', img_scale)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True,
                            drop_last=True)

    global_step = 0

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=len(train_set), desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']

                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                target = true_masks.to(device=device, dtype=torch.long)

                masks_pred = net(imgs)
                probs = F.softmax(masks_pred, dim=1)
                argmx = torch.argmax(probs, dim=1).to(dtype=torch.float32)

                example_images = [wandb.Image(imgs[0], caption='Image'),
                                  wandb.Image(target.to(dtype=torch.float)[0],
                                              caption='True Mask'),
                                  wandb.Image(argmx[0],
                                              caption='Predicted Mask')]

                wandb.log({"Examples": example_images})

                loss = criterion(masks_pred, target.squeeze(1))
                wandb.log({"Training Loss": loss})
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
                    val_loss, val_iou, val_acc = eval_net(net, val_loader, device)
                    wandb.log({"Validation Loss": val_loss})
                    wandb.log({"Validation IoU": val_iou})
                    wandb.log({"Validation Accuracy": val_acc})
                    # scheduler.step(val_loss)

        if save_cp:
            try:
                os.mkdir('../checkpoints/')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       '../checkpoints/' + f'epoch{epoch + 1}.pth')


if __name__ == '__main__':
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet(n_channels=3, n_classes=3, bilinear=True)
    wandb.watch(net)

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))

    net.to(device=device)

    try:
        train_net(net=net, data_dir=args.data_dir, epochs=args.epochs, batch_size=args.batchsize, lr=args.lr,
                  device=device,
                  img_scale=args.scale, img_crop=args.crop)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), '../INTERRUPTED.pth')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
