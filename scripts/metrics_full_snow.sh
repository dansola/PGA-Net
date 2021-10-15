#!/bin/bash

python epoch_metrics_full_snow.py --model unet
python epoch_metrics_full_snow.py --model dsc_unet
python epoch_metrics_full_snow.py --model lbc_unet
python epoch_metrics_full_snow.py --model dsc_lbc_unet
#python epoch_metrics_full_snow.py --model deeplab_mobile_net
python epoch_metrics_full_snow.py --model lraspp_mobile_net


