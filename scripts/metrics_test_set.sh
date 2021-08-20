#!/bin/bash

python epoch_metrics_test_set.py --model unet
python epoch_metrics_test_set.py --model small_unet
python epoch_metrics_test_set.py --model dsc_unet
python epoch_metrics_test_set.py --model small_dsc_unet
python epoch_metrics_test_set.py --model lbc_unet
python epoch_metrics_test_set.py --model small_lbc_unet
python epoch_metrics_test_set.py --model dsc_lbc_unet
python epoch_metrics_test_set.py --model small_dsc_lbc_unet
python epoch_metrics_test_set.py --model deeplab_mobile_net
python epoch_metrics_test_set.py --model lraspp_mobile_net


