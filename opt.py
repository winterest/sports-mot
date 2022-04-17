"""opts for main.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys


class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        # basic experiment setting
        self.parser.add_argument(
            "--task", default="alignment", help="alignment|align_seg ",
        )
        self.parser.add_argument(
            "--train_dataset", default="stacks", help="stacks | pairs",
        )
        self.parser.add_argument(
            "--test_dataset",
            default="",
            help="coco | kitti | coco_hp | pascal",
        )
        self.parser.add_argument("--exp_id", default="default")
        self.parser.add_argument("--test", action="store_true")
        self.parser.add_argument(
            "--load_model", default="", help="path to pretrained model"
        )
        self.parser.add_argument(
            "--resume",
            action="store_true",
            help="resume an experiment. "
            "Reloaded the optimizer parameter and "
            "set load_model to model_last.pth "
            "in the exp dir if load_model is empty.",
        )
        self.parser.add_argument(
            "--epochs", default=50, type=int, help="num of epochs"
        )

        self.parser.add_argument(
            "--gpus",
            default=-1,
            type=int,
            help="-2:cpu; -1:by usage; other:by opt",
        )
        self.parser.add_argument(
            "--lr", default=1e-5, type=float, help="learning rate"
        )
        self.parser.add_argument(
            "--bs", default=8, type=int, help="batch size"
        )
        self.parser.add_argument(
            "--downsize", default=1, type=int, help="dataset down scale size"
        )
        self.parser.add_argument(
            "--align_inc", default=6, type=int, help="alignment input channel"
        )
        self.parser.add_argument(
            "--align_p", default=124, type=int, help="alignment p"
        )
        self.parser.add_argument(
            "--videos_path",
            default="/mounted/mnt-gluster/cdl-data/xliu/mot/sports_video_tracking/FieldAlignSegmentation/data/nfl_frames/",
            help="path to video frames",
        )

    def parse(self, args=""):
        if args == "":
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)

        return opt
