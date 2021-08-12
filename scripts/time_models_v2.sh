#!/bin/bash
cd ..
cd src/train

for NUMBER in 0 1 2 3 4 5 6 7 8 9
do
  python train_timer.py --epochs 3 --device cuda --model small_unet --file_number $NUMBER
  python train_timer.py --epochs 3 --device cuda --model small_lbc_unet --file_number $NUMBER

  python train_timer.py --epochs 3 --device cpu --model small_unet --file_number $NUMBER
  python train_timer.py --epochs 3 --device cpu --model small_lbc_unet --file_number $NUMBER

#	python train_timer.py --epochs 3 --device cuda --model unet --file_number $NUMBER
#  python train_timer.py --epochs 3 --device cuda --model lbc_unet --file_number $NUMBER
#  python train_timer.py --epochs 3 --device cuda --model small_axial_lbc_unet --file_number $NUMBER
#  python train_timer.py --epochs 3 --device cuda --model deeplab_mobile_net --file_number $NUMBER
#  python train_timer.py --epochs 3 --device cuda --model lraspp_mobile_net --file_number $NUMBER
#  python train_timer.py --epochs 3 --device cuda --model dsc_unet --file_number $NUMBER
#  python train_timer.py --epochs 3 --device cuda --model small_dsc_unet --file_number $NUMBER
#  python train_timer.py --epochs 3 --device cuda --model dsc_lbc_unet --file_number $NUMBER
#  python train_timer.py --epochs 3 --device cuda --model small_dsc_lbc_unet --file_number $NUMBER
#
#  python train_timer.py --epochs 3 --device cpu --model unet --file_number $NUMBER
#  python train_timer.py --epochs 3 --device cpu --model lbc_unet --file_number $NUMBER
#  python train_timer.py --epochs 3 --device cpu --model small_axial_lbc_unet --file_number $NUMBER
#  python train_timer.py --epochs 3 --device cpu --model deeplab_mobile_net --file_number $NUMBER
#  python train_timer.py --epochs 3 --device cpu --model lraspp_mobile_net --file_number $NUMBER
#  python train_timer.py --epochs 3 --device cpu --model dsc_unet --file_number $NUMBER
#  python train_timer.py --epochs 3 --device cpu --model small_dsc_unet --file_number $NUMBER
#  python train_timer.py --epochs 3 --device cpu --model dsc_lbc_unet --file_number $NUMBER
#  python train_timer.py --epochs 3 --device cpu --model small_dsc_lbc_unet --file_number $NUMBER
done