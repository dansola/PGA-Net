#!/bin/bash

python epoch_metrics_per_class.py --model unet
#python epoch_metrics_per_class.py --model small_unet
#python epoch_metrics_per_class.py --model dsc_unet
#python epoch_metrics_per_class.py --model small_dsc_unet
#python epoch_metrics_per_class.py --model lbc_unet
#python epoch_metrics_per_class.py --model small_lbc_unet
#python epoch_metrics_per_class.py --model dsc_lbc_unet
#python epoch_metrics_per_class.py --model small_dsc_lbc_unet
#python epoch_metrics_per_class.py --model deeplab_mobile_net
python epoch_metrics_per_class.py --model lraspp_mobile_net


