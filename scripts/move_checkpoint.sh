#!/bin/bash

mkdir ./repos/PGA-Net/checkpoints/classic_deluge_213_full_axial_lbc_unet_2e-1_weight_decay/
cp -a ./repos/PGA-Net/src/checkpoints/. ./repos/PGA-Net/checkpoints/classic_deluge_213_full_axial_lbc_unet_2e-1_weight_decay
rm -r ./repos/PGA-Net/src/checkpoints