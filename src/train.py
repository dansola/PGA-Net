import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from eval import eval_net
from models.axial_unet import AxialUnet
from datasets.ice import Ice
from torch.utils.data import DataLoader
import wandb
wandb.init()


def get_args():
    parser = argparse.ArgumentParser(description='Train AxialUnet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data_directory', metavar='D', type=str, default='./data',
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
    parser.add_argument('-c', '--crop', dest='crop', type=int, default=320,
                        help='Height and width of images and masks.')

    return parser.parse_args()


def train_net(net, data_dir, device, epochs=20, batch_size=1, lr=0.0001, save_cp=True, img_scale=0.35, img_crop=320):
    train_set = Ice(os.path.join(data_dir, 'imgs'), os.path.join(data_dir, 'labels'),
                    os.path.join(data_dir, 'txt_files'), 'train', img_scale, img_crop)
    val_set = Ice(os.path.join(data_dir, 'imgs'), os.path.join(data_dir, 'labels'),
                  os.path.join(data_dir, 'txt_files'), 'val', img_scale, img_crop)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

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

                assert imgs.shape[-1] == net.channels, \
                    f'Network has been defined with {net.channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)

                masks_pred = net(imgs)

                target = torch.argmax(true_masks.to(dtype=torch.long), dim=1)
                loss = criterion(masks_pred, target)
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
                    val_score = eval_net(net, val_loader, device)
                    wandb.log({"Validation Loss": val_score})
                    scheduler.step(val_score)

        if save_cp:
            try:
                os.mkdir('checkpoints/')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       'checkpoints/' + f'CP_epoch{epoch + 1}.pth')


if __name__ == '__main__':
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = AxialUnet(channels=3, n_classes=3, embedding_dims=20, sine_pos=False, img_crop=args.crop)
    wandb.watch(net)



    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))

    net.to(device=device)

    try:
        train_net(net=net, data_dir=args.data_dir, epochs=args.epochs, batch_size=args.batchsize, lr=args.lr, device=device,
                  img_scale=args.scale, img_crop=args.crop)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)