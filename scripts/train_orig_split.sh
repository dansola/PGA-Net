#!/bin/bash

for MODEL in unet
do
  for CV in 2 3 4 5
  do
    cd ..
    cd src/train
    NAME="${MODEL}_original_split_${CV}"
    python train_ice_orig_split.py --epochs 80 --model $MODEL --imgdir imgs --wandbname $NAME
    cd ../..
    mkdir ./checkpoints/$NAME/
    cp -a ./src/checkpoints/. ./checkpoints/$NAME
    rm -r ./src/checkpoints
    cd scripts
    python epoch_metrics_original_split.py --model $MODEL --checkpoint $NAME
  done
done