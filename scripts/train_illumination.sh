#!/bin/bash

for MODEL in unet small_unet dsc_unet small_dsc_unet lbc_unet small_lbc_unet dsc_lbc_unet small_dsc_lbc_unet lraspp_mobile_net
do
  cd ..
  cd src/train
  TRAIN_DIR="illumination_shift_200"
  EVAL_DIR="illumination_shift_25"
  NAME="${MODEL}_${TRAIN_DIR}"
  python train_ice.py --epochs 80 --model $MODEL --imgdir $TRAIN_DIR --wandbname $NAME
  cd ../..
  mkdir ./checkpoints/$NAME/
  cp -a ./src/checkpoints/. ./checkpoints/$NAME
  rm -r ./src/checkpoints
  cd scripts
  python epoch_metrics_illumination.py --model $MODEL --checkpoint $NAME --evalset $EVAL_DIR

  cd ..
  cd src/train
  TRAIN_DIR="illumination_shift_25"
  EVAL_DIR="illumination_shift_200"
  NAME="${MODEL}_${TRAIN_DIR}"
  python train_ice.py --epochs 80 --model $MODEL --imgdir $TRAIN_DIR --wandbname $NAME
  cd ../..
  mkdir ./checkpoints/$NAME/
  cp -a ./src/checkpoints/. ./checkpoints/$NAME
  rm -r ./src/checkpoints
  cd scripts
  python epoch_metrics_illumination.py --model $MODEL --checkpoint $NAME --evalset $EVAL_DIR
done