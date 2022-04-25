"""seg_train_opts for main.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse


class seg_train_opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        # basic experiment setting
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

        self.parser.add_argument(
            "--train_num_frames",
            default=7,
            type=int,
            help="number of frames for each sample in training",
        )
        self.parser.add_argument(
            "--valid_num_frames",
            default=5,
            type=int,
            help="number of frames for each sample in validation",
        )
        self.parser.add_argument(
            "--weight_add_task",
            default=1.0,
            type=float,
            help="weight for addtional training task",
        )
        self.parser.add_argument(
            "--add_task",
            default="reconstruct",
            help="reconstruct | rot10 | colorizing | nextframe",
        )

        self.parser.add_argument(
            "--loss_fn", default="BCELoss", help="MSELoss | BCELoss",
        )
        self.parser.add_argument("--debug_mode", action="store_true")
        self.parser.add_argument("--smooth_label", action="store_true")

    def parse(self, args=""):
        if args == "":
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)

        return opt
