#!/bin/bash
cd ..
cd src/train

python train_val_loss.py --epochs 30 --device cuda --model unet
python train_val_loss.py --epochs 30 --device cuda --model small_unet
python train_val_loss.py --epochs 30 --device cuda --model lbc_unet
python train_val_loss.py --epochs 30 --device cuda --model small_lbc_unet
python train_val_loss.py --epochs 30 --device cuda --model deeplab_mobile_net
python train_val_loss.py --epochs 30 --device cuda --model lraspp_mobile_net
python train_val_loss.py --epochs 30 --device cuda --model dsc_unet
python train_val_loss.py --epochs 30 --device cuda --model small_dsc_unet
python train_val_loss.py --epochs 30 --device cuda --model dsc_lbc_unet
python train_val_loss.py --epochs 30 --device cuda --model small_dsc_lbc_unet

