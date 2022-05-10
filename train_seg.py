from __future__ import print_function

import os
import time

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

# import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF

from data.nfl_dataset import NflImageStacks
from data.videos_train_valid_split import split_videos

# from PIL import Image, ImageDraw
from _init_paths import init_paths

init_paths()
from aligning import ResSTN, pureSTN
from utils import get_most_idle_gpu, rgb_to_l
import matplotlib.pyplot as plt
from segmentation import UNet, UNet_two_head
from seg_train_opts import seg_train_opts

plt.ion()  # interactive mode

if __name__ == "__main__":
    #### Hyper Parameters
    opt = seg_train_opts().parse()
    if opt.gpus == -2:
        device = "cpu"
    elif opt.gpus == -1:
        device = get_most_idle_gpu()  # device = "cuda:2"
    else:
        device = "cuda:{}".format(opt.gpus)
    opt.device = device
    EPOCHS = opt.epochs
    LR = opt.lr
    BS = opt.bs

    print("Now Using device {}".format(device))
    train_ds, valid_ds, test_ds = split_videos(
        videos_path=opt.videos_path, patterns=opt.vid_patterns, soccernet=opt.soccernet
    )
    TRAIN_NUM_FRAMES = opt.train_num_frames
    VALID_NUM_FRAMES = opt.valid_num_frames
    WEIGHT_ADD = opt.weight_add_task
    ADD_TASK = opt.add_task  # "rot10" | "colorizing" | "reconstruct"
    ALIGN_MODEL_PATH = opt.align_model_path

    BS = 16
    LR = 0.00001
    EPOCHS = 100
    ANGLE = 10
    DOWN = 10 if opt.debug_mode else 1

    sm = nn.AvgPool2d(3, stride=1, padding=1)

    # checkpoints folder
    stamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    checkpoint_path = os.path.join("./checkpoints", stamp)
    os.makedirs(checkpoint_path)
    path_log = os.path.join(checkpoint_path, "log")
    with open(path_log, "a") as log:
        log.write(
            "epoch | iteration | segmentation loss | additional loss | total loss\n"
        )
    with open(os.path.join(checkpoint_path, "opt"), "a") as log:
        log.write(str(opt))

    if opt.loss_fn == "MSELoss":
        seg_loss_fn = lambda segmm, seg_ref: F.mse_loss(segmm, seg_ref)
    elif opt.loss_fn == "BCELoss":
        seg_loss_fn = lambda segmm, seg_ref: F.binary_cross_entropy(
            seg_ref.view(-1), segmm.view(-1)
        )
        """
        seg_loss_fn = lambda segmm, seg_ref: F.binary_cross_entropy(
            torch.sigmoid(seg_ref).view(-1), segmm.view(-1)
        )
        """

    # alignment model loading
    input_c = 512
    output_c = ((input_c - 7 + 1) // 2 - 5 + 1) // 2

    only_alpha = False  # get 6 affine paras, only_alpha get only 1 (alpha)
    align_inc = 6  # input channels
    align_p = output_c  # feature channels
    align_model = ResSTN(inc=align_inc, p=align_p)  # , only_alpha=only_alpha)

    align_model = align_model.to(device)
    TO_LOAD = torch.load(ALIGN_MODEL_PATH, map_location=lambda storage, loc: storage)
    """
    if isinstance(TO_LOAD, dict):
        align_model.load_state_dict(TO_LOAD["state_dict"])
    else:
        align_model.load_state_dict(TO_LOAD)
    """
    try:
        align_model.load_state_dict(TO_LOAD)
    except:
        align_model.load_state_dict(TO_LOAD["state_dict"])


    print("alignment loaded")
    align_model.eval()

    # data set definition
    training_dataset = NflImageStacks(
        nfl_dir=train_ds,
        num_frames=TRAIN_NUM_FRAMES,
        size=(512, 512),
        max_interval=3,
        min_interval=1,
        mode="train",
        down_size=DOWN,
    )

    validation_dataset = NflImageStacks(
        nfl_dir=valid_ds,
        num_frames=VALID_NUM_FRAMES,
        size=(512, 512),
        max_interval=3,
        min_interval=1,
        mode="validation",
        down_size=DOWN,
    )

    train_loader = DataLoader(
        training_dataset, batch_size=BS, shuffle=True, num_workers=max(8, BS)
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=BS, shuffle=False, num_workers=max(8, BS),
    )

    # segmentation model definition
    unet_inc = 1 if ADD_TASK == "colorizing" else 3
    unet_feat = 32
    unet_outc = 1

    seg_model = UNet_two_head(
        in_channels=unet_inc,
        out_channels=unet_outc,
        init_features=unet_feat,
        out_channels_add=3,
    )

    seg_model = seg_model.to(device)

    optimizer = optim.Adam(seg_model.parameters(), lr=LR)
    losses = {"seg": [], "rot": [], "train": [], "validation": []}

    for epoch in range(EPOCHS):
        seg_model.train()
        epoch_loss, cnt_sample = 0, 0
        train_loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)
        for batch_idx, sample in train_loop:

            with torch.no_grad():
                align_model.eval()
                n = len(sample)
                assert (
                    n == TRAIN_NUM_FRAMES
                ), "sampling number is not the same as defined"

                ref = sample[n // 2].to(device)
                b, _, h, w = ref.size()

                segmm = torch.zeros((b, 1, h, w)).to(device)

                for target in sample[: n // 2] + sample[n // 2 + 1 :]:
                    target = target.to(device)
                    alg_target, theta0 = align_model(target, ref)
                    seg = (alg_target - ref) > 0.5

                    canvas = torch.ones((b, 1, h, w)).to(device)
                    grid = F.affine_grid(theta0, canvas.size())
                    canvas = F.grid_sample(canvas, grid)

                    segm = 1.0 * (seg.any(axis=1).reshape((b, 1, h, w)))  # b, 1, h, w
                    if opt.smooth_label:
                        segm = sm(segm)
                    segm = segm * canvas
                    # segm = segm.repeat(1,3,1,1) # b, 3, h, w
                    segmm = torch.maximum(segmm, segm)
            optimizer.zero_grad()

            if ADD_TASK == "colorizing":
                sample_l = rgb_to_l(sample)
                ref_l = sample_l[n // 2].to(device)
                seg_ref, add_ref = seg_model(ref_l)  # seg_ref: b,1,h,w
                loss_add = F.mse_loss(ref, add_ref)

            elif ADD_TASK == "rot10":
                seg_ref, add_ref = seg_model(ref)  # seg_ref: b,1,h,w
                rotated = TF.rotate(ref, ANGLE)
                loss_add = F.mse_loss(rotated, add_ref)

            elif ADD_TASK == "reconstruct":
                seg_ref, add_ref = seg_model(ref)  # seg_ref: b,1,h,w
                loss_add = F.mse_loss(ref, add_ref)

            else:
                seg_ref, add_ref = seg_model(ref)  # seg_ref: b,1,h,w
                next_frame = sample[n // 2 + 1].to(device)
                loss_add = F.mse_loss(next_frame, add_ref)

            loss_seg = seg_loss_fn(segmm, seg_ref)
            loss = loss_seg + WEIGHT_ADD * loss_add

            losses["seg"].append(loss_seg.item())
            losses["rot"].append(loss_add.item())

            loss.backward()
            optimizer.step()

            train_loop.set_description(f"Training Epoch [{epoch}/{EPOCHS}]")
            train_loop.set_postfix(
                losses="seg: {:.6f} rot: {:.6f}".format(
                    loss_seg.item(), loss_add.item()
                )
            )

            epoch_loss += loss.item() * n
            cnt_sample += n

            with open(path_log, "a") as log:
                log.write(
                    "{}   {}   {:.6f}   {:.6f}   {:.6f}\n".format(
                        epoch, batch_idx, loss_seg.item(), loss_add.item(), loss.item(),
                    )
                )

        losses["train"].append(epoch_loss / cnt_sample)
        print(epoch, epoch_loss / cnt_sample)

        #### validation

        seg_model.eval()

        epoch_loss_add, epoch_loss_seg, cnt_sample = 0, 0, 0
        val_loop = tqdm(
            enumerate(validation_loader), total=len(validation_loader), leave=True,
        )
        for batch_idx, sample in val_loop:
            with torch.no_grad():
                #### get alignment fake label
                align_model.eval()
                n = len(sample)
                assert (
                    n == VALID_NUM_FRAMES
                ), "sampling number is not the same as defined"

                ref = sample[n // 2].to(device)
                b, _, h, w = ref.size()
                segmm = torch.zeros((b, 1, h, w)).to(device)
                for target in sample[: n // 2] + sample[n // 2 + 1 :]:
                    target = target.to(device)
                    alg_target, theta0 = align_model(target, ref)
                    seg = (alg_target - ref) > 0.5
                    segm = 1.0 * (seg.any(axis=1).reshape((b, 1, h, w)))  # b, 1, h, w
                    # segm = segm.repeat(1,3,1,1) # b, 3, h, w
                    segmm = torch.maximum(segmm, segm)
                #### get alignment fake label

                n = len(sample)
                ref = sample[n // 2 + 1].to(device)

                if ADD_TASK == "colorizing":
                    sample_l = rgb_to_l(sample)
                    ref_l = sample_l[n // 2 + 1].to(device)
                    seg_ref, add_ref = seg_model(ref_l)  # seg_ref: b,1,h,w
                    loss_add = F.mse_loss(ref, add_ref)

                elif ADD_TASK == "rot10":
                    seg_ref, add_ref = seg_model(ref)  # seg_ref: b,1,h,w
                    rotated = TF.rotate(ref, ANGLE)
                    loss_add = F.mse_loss(rotated, add_ref)

                elif ADD_TASK == "reconstruct":
                    seg_ref, add_ref = seg_model(ref)  # seg_ref: b,1,h,w
                    loss_add = F.mse_loss(ref, add_ref)

                else:
                    seg_ref, add_ref = seg_model(ref)  # seg_ref: b,1,h,w
                    next_frame = sample[n // 2 + 1].to(device)
                    loss_add = F.mse_loss(next_frame, add_ref)

                loss_seg = seg_loss_fn(segmm, seg_ref)
                loss = loss_seg + WEIGHT_ADD * loss_add

                epoch_loss_add += loss_add.item() * n
                epoch_loss_seg += loss_seg.item() * n
                cnt_sample += n
                val_loop.set_description(f"Validation Epoch [{epoch}/{EPOCHS}]")
                val_loop.set_postfix(
                    losses="add: {:.6f} seg: {:.6f} ".format(
                        loss_add.item(), loss_seg.item()
                    )
                )

        with open(path_log, "a") as log:
            log.write(
                "validation: {} {:.6f} {:.6f}\n".format(
                    epoch, epoch_loss_add / cnt_sample, epoch_loss_seg / cnt_sample,
                )
            )

        torch.save(
            seg_model.state_dict(),
            "./checkpoints/{}/seg_model_{}.pth".format(stamp, epoch),
        )
