#!/bin/bash

for MODEL in unet small_dsc_lbc_unet lraspp_mobile_net
do
  for CV in 1 2 3 4 5
  do
    cd ..
    cd src/train
    NAME="${MODEL}_cv${CV}"
    python train_ice_cv.py --epochs 80 --model $MODEL --imgdir imgs --wandbname $NAME --cv $CV
    cd ../..
    mkdir ./checkpoints/$NAME/
    cp -a ./src/checkpoints/. ./checkpoints/$NAME
    rm -r ./src/checkpoints
    cd scripts
    python epoch_metrics_cv.py --model $MODEL --checkpoint $NAME --evalset imgs --cv $CV
  done
done