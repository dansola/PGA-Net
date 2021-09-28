#!/bin/bash
cd ..
cd src/eval

for NUMBER in 0 1 2 3 4 5 6 7 8 9
do
#	python eval_timer.py --device cuda --model unet --file_number $NUMBER
#	python eval_timer.py --device cuda --model dsc_unet --file_number $NUMBER
#  python eval_timer.py --device cuda --model lbc_unet --file_number $NUMBER
#  python eval_timer.py --device cuda --model dsc_lbc_unet --file_number $NUMBER
#	python eval_timer.py --device cuda --model small_unet --file_number $NUMBER
#	python eval_timer.py --device cuda --model small_dsc_unet --file_number $NUMBER
#  python eval_timer.py --device cuda --model small_lbc_unet --file_number $NUMBER
#  python eval_timer.py --device cuda --model small_dsc_lbc_unet --file_number $NUMBER
  python eval_timer.py --device cuda --model skinny_small_dsc_lbc_unet --file_number $NUMBER
#  python eval_timer.py --device cuda --model deeplab_mobile_net --file_number $NUMBER
#  python eval_timer.py --device cuda --model lraspp_mobile_net --file_number $NUMBER

#	python eval_timer.py --device cpu --model unet --file_number $NUMBER
#	python eval_timer.py --device cpu --model dsc_unet --file_number $NUMBER
#  python eval_timer.py --device cpu --model lbc_unet --file_number $NUMBER
#  python eval_timer.py --device cpu --model dsc_lbc_unet --file_number $NUMBER
#	python eval_timer.py --device cpu --model small_unet --file_number $NUMBER
#	python eval_timer.py --device cpu --model small_dsc_unet --file_number $NUMBER
#  python eval_timer.py --device cpu --model small_lbc_unet --file_number $NUMBER
#  python eval_timer.py --device cpu --model small_dsc_lbc_unet --file_number $NUMBER
  python eval_timer.py --device cpu --model skinny_small_dsc_lbc_unet --file_number $NUMBER
#  python eval_timer.py --device cpu --model deeplab_mobile_net --file_number $NUMBER
#  python eval_timer.py --device cpu --model lraspp_mobile_net --file_number $NUMBER
done