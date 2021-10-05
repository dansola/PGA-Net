#!/bin/bash

for NUMBER in 12 25 50 75 87
do
  cd ..
  cd src/train
  python train_unet.py --epochs 80 --sparsity $NUMBER
  cd ../..
  mkdir ./checkpoints/skinny_small_dsc_lbc_unet_sparcity_$NUMBER/
  cp -a ./src/checkpoints/. ./checkpoints/skinny_small_dsc_lbc_unet_sparcity_$NUMBER
  rm -r ./src/checkpoints
  cd scripts
  python sparsity_metrics.py --model small_dsc_lbc_unet --checkpoint skinny_small_dsc_lbc_unet_sparcity_$NUMBER --epochs 80 --sparsity $NUMBER
done