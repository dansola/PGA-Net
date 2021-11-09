#!/bin/bash

for MODEL in unet small_unet dsc_unet small_dsc_unet lbc_unet small_lbc_unet dsc_lbc_unet small_dsc_lbc_unet deeplab_mobile_net lraspp_mobile_net
do
  for ROUND in 1 2 3
  do
    cd ..
    cd src/train
    NAME="${MODEL}_${ROUND}"
    python train_ice.py --epochs 80 --model $MODEL --imgdir imgs --wandbname $NAME
    cd ../..
    mkdir ./checkpoints/$NAME/
    cp -a ./src/checkpoints/. ./checkpoints/$NAME
    rm -r ./src/checkpoints
    cd scripts
    python epoch_metrics_ice.py --model $MODEL --checkpoint $NAME --evalset imgs
  done
done