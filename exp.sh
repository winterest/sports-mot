#!/bin/zsh

#python train_seg.py --add_task rot10 --weight_add_task 1.0
python train_seg.py --add_task rot10 --weight_add_task 1.0 --smooth_label
#python train_seg.py --add_task nextframe --weight_add_task 1.0 --smooth_label
