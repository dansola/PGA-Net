#!/bin/bash

python epoch_metrics_frequency_weighted.py --model unet
python epoch_metrics_frequency_weighted.py --model small_unet
python epoch_metrics_frequency_weighted.py --model dsc_unet
python epoch_metrics_frequency_weighted.py --model small_dsc_unet
python epoch_metrics_frequency_weighted.py --model lbc_unet
python epoch_metrics_frequency_weighted.py --model small_lbc_unet
python epoch_metrics_frequency_weighted.py --model dsc_lbc_unet
python epoch_metrics_frequency_weighted.py --model small_dsc_lbc_unet
python epoch_metrics_frequency_weighted.py --model deeplab_mobile_net
python epoch_metrics_frequency_weighted.py --model lraspp_mobile_net


