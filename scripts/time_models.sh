#!/bin/bash
cd ..
cd src/train
python train_timer.py --epochs 10 --device cuda --model unet
python train_timer.py --epochs 10 --device cuda --model small_unet
python train_timer.py --epochs 10 --device cuda --model axial_unet
python train_timer.py --epochs 10 --device cuda --model small_axial_unet
python train_timer.py --epochs 10 --device cuda --model lbc_unet
python train_timer.py --epochs 10 --device cuda --model small_lbc_unet
python train_timer.py --epochs 10 --device cuda --model axial_lbc_unet
python train_timer.py --epochs 10 --device cuda --model small_axial_lbc_unet
python train_timer.py --epochs 10 --device cuda --model small_axial_lbc_unet_10

python train_timer.py --epochs 10 --device cpu --model unet
python train_timer.py --epochs 10 --device cpu --model small_unet
python train_timer.py --epochs 10 --device cpu --model axial_unet
python train_timer.py --epochs 10 --device cpu --model small_axial_unet
python train_timer.py --epochs 10 --device cpu --model lbc_unet
python train_timer.py --epochs 10 --device cpu --model small_lbc_unet
python train_timer.py --epochs 10 --device cpu --model axial_lbc_unet
python train_timer.py --epochs 10 --device cpu --model small_axial_lbc_unet
python train_timer.py --epochs 10 --device cpu --model small_axial_lbc_unet_10