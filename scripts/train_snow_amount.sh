#!/bin/bash

for MODEL in unet small_dsc_lbc_unet lraspp_mobile_net
do
  for IMG_DIR in imgs imgs_one_third imgs_two_third imgs_snow
  do
    cd ..
    cd src/train
    NAME="${MODEL}_${IMG_DIR}_dir_3"
    python train_ice.py --epochs 80 --model $MODEL --imgdir $IMG_DIR --wandbname $NAME
    cd ../..
    mkdir ./checkpoints/$NAME/
    cp -a ./src/checkpoints/. ./checkpoints/$NAME
    rm -r ./src/checkpoints
    cd scripts
    for IMG_DIR_EVAL in imgs imgs_eval_half_snow imgs_snow
    do
      python epoch_metrics_snow_amount.py --model $MODEL --checkpoint $NAME --evalset $IMG_DIR_EVAL
    done
  done
done