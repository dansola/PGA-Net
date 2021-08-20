#!/bin/bash

python epoch_metrics_floe_test.py --model unet
python epoch_metrics_floe_test.py --model small_unet
python epoch_metrics_floe_test.py --model dsc_unet
python epoch_metrics_floe_test.py --model small_dsc_unet
python epoch_metrics_floe_test.py --model lbc_unet
python epoch_metrics_floe_test.py --model small_lbc_unet
python epoch_metrics_floe_test.py --model dsc_lbc_unet
python epoch_metrics_floe_test.py --model small_dsc_lbc_unet
python epoch_metrics_floe_test.py --model lraspp_mobile_net


