#!/bin/bash

python epoch_metrics_skinny.py --model small_unet
python epoch_metrics_skinny.py --model small_dsc_unet
python epoch_metrics_skinny.py --model small_lbc_unet
python epoch_metrics_skinny.py --model small_dsc_lbc_unet

