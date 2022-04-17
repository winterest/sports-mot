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
from utils import get_most_idle_gpu
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
    train_ds, valid_ds, test_ds = split_videos()
    TRAIN_NUM_FRAMES = opt.train_num_frames
    VALID_NUM_FRAMES = opt.valid_num_frames
    WEIGHT_ADD = opt.weight_add_task
    ADD_TASK = opt.add_task  # "rot10" | "colorizing" | "reconstruct"
    COLOR = "L" if ADD_TASK == "colorizing" else "RGB"

    train_single_fake_label = False

    input_c = 512
    output_c = ((input_c - 7 + 1) // 2 - 5 + 1) // 2

    only_alpha = False  # get 6 affine paras, only_alpha get only 1 (alpha)
    align_inc = 6  # input channels
    align_p = output_c  # feature channels
    align = ResSTN(inc=align_inc, p=align_p)  # , only_alpha=only_alpha)

    model = align.to(device)

    training_dataset = NflImageStacks(
        nfl_dir=train_ds,
        num_frames=TRAIN_NUM_FRAMES,
        size=(512, 512),
        max_interval=3,
        min_interval=1,
        mode="train",
        colortype=COLOR,
    )

    validation_dataset = NflImageStacks(
        nfl_dir=valid_ds,
        num_frames=VALID_NUM_FRAMES,
        size=(512, 512),
        max_interval=3,
        min_interval=1,
        mode="validation",
        colortype=COLOR,
    )

    smoothed = lambda s, l: [sum(s[i : i + l]) / l for i in range(0, len(s), l)]

    m, s = np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5])
    preprocess = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=m, std=s),]
    )

    inv_normalize = transforms.Normalize(
        mean=[-mm / ss for mm, ss in zip(m, s)], std=[1 / ss for ss in s]
    )

    # segmentation

    unet_inc = 1 if COLOR == "L" else 3
    unet_feat = 32
    unet_outc = 1

    # unet = UNet(in_channels=unet_inc, out_channels=unet_outc, init_features=unet_feat)
    unet = UNet_two_head(
        in_channels=unet_inc,
        out_channels=unet_outc,
        init_features=unet_feat,
        out_channels_add=3,
    )

    align_model = model
    align_model.load_state_dict(
        torch.load("./best_valid_model", map_location=lambda storage, loc: storage)
    )
    print("alignment loaded")
    align_model.eval()

    seg_model = unet.to(device)

    BS = 16
    LR = 0.00001
    EPOCHS = 100

    train_loader = DataLoader(
        training_dataset, batch_size=BS, shuffle=True, num_workers=max(8, BS)
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=BS, shuffle=False, num_workers=max(8, BS),
    )

    optimizer = optim.Adam(seg_model.parameters(), lr=LR)

    losses = {"seg": [], "rot": [], "train": [], "validation": []}

    def union(x, y, union_fn="max"):
        if union_fn == "max":
            return torch.max(x, y)
        elif union_fn == "relu":
            relu = nn.ReLU()
            return relu(x - y) + y

    stamp = time.strftime("%Y-%m-%d-%H-%M")
    path = os.path.join("./checkpoints", stamp)
    # Create the directory
    os.makedirs(path)
    with open(os.path.join(path, "log"), "a") as log:
        log.write(
            "epoch | iteration | segmentation loss | additional loss | total loss\n"
        )

    ## fake seg label
    """
    if train_single_fake_label:
        for epoch in range(EPOCHS):
            seg_model.train()
            epoch_loss, cnt_sample = 0, 0
            train_loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)
            for batch_idx, sample in train_loop:
                with torch.no_grad():
                        align_model.eval()
                        ref, target = sample[:2]
                        ref, target = ref.to(device), target.to(device)
                        alg_ref, theta0 = align_model(ref, target)

                        seg =  (1.0*((alg_ref.permute(0,2,3,1).cpu().detach().numpy() -
                                        target.permute(0,2,3,1).cpu().detach().numpy()) > 0.5))  
                        segm = 1.0 * (seg.any(axis=3).reshape((-1,512,512,1)))

                optimizer.zero_grad()
                seg_ref = seg_model(alg_ref)
                
                # CE loss
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_ce(seg_ref, segmm[:,0,:,:].long())
                # MSE loss
                #loss = F.mse_loss(torch.Tensor(segm).to(device).permute(0,3,1,2), seg_ref)
                
                losses["back"].append(loss.item())
                loss.backward()
                optimizer.step()

                train_loop.set_description(f"Training Epoch [{epoch}/{EPOCHS}]")
                train_loop.set_postfix(loss="{:.6f}".format(loss.item()))

                epoch_loss += loss.item()*len(sample)
                cnt_sample += len(sample)

            losses["train"].append(epoch_loss/cnt_sample)
            torch.save(seg_model.state_dict(), "./checkpoints/seg/new_model_{}_{}.pth".format(epoch,epoch_loss/cnt_sample))

    else:
    """
    if True:

        ANGLE = 10

        for epoch in range(EPOCHS):
            seg_model.train()
            epoch_loss, cnt_sample = 0, 0
            train_loop = tqdm(
                enumerate(train_loader), total=len(train_loader), leave=True
            )
            for batch_idx, (sample, sample_l) in train_loop:

                with torch.no_grad():
                    align_model.eval()
                    n = len(sample)
                    assert (
                        n == TRAIN_NUM_FRAMES
                    ), "sampling number is not the same as defined"

                    ref = sample[n // 2].to(device)
                    b, _, h, w = ref.size()

                    # segmm = torch.zeros_like(ref)[:,0,:,:]    # B,1,H,W
                    segmm = torch.zeros((b, 1, h, w)).to(device)

                    for target in sample[: n // 2] + sample[n // 2 + 1 :]:
                        target = target.to(device)
                        alg_target, theta0 = align_model(target, ref)
                        seg = (alg_target - ref) > 0.5

                        canvas = torch.ones_like(target).to(device)
                        grid = F.affine_grid(theta0, canvas.size())
                        canvas = F.grid_sample(canvas, grid)

                        segm = 1.0 * (
                            seg.any(axis=1).reshape((b, 1, h, w))
                        )  # b, 1, h, w
                        segm = segm * canvas
                        # segm = segm.repeat(1,3,1,1) # b, 3, h, w
                        segmm = torch.maximum(segmm, segm)

                    """
                    ref, target = sample[:2]                                                   # B, C, H, W: B*3*512*512
                    ref, target = ref.to(device), target.to(device)
                    alg_ref, theta0 = align_model(ref, target)

                    seg =  (1.0*((alg_ref.permute(0,2,3,1).cpu().detach().numpy() -
                                    target.permute(0,2,3,1).cpu().detach().numpy()) > 0.5))  # B, H, W, 3: B*512*512*3
                    segm = 1.0 * (seg.any(axis=3).reshape((-1,512,512,1)))
                    """
                ref_l = sample_l[n // 2].to(device)

                optimizer.zero_grad()
                seg_ref, add_ref = seg_model(ref_l)
                loss_seg = F.mse_loss(segmm, seg_ref)

                # rotated = TF.rotate(ref, ANGLE)
                # loss_add = F.mse_loss(rotated, add_ref)

                # next_frame = sample[n//2+1].to(device)
                # loss_add = F.mse_loss(next_frame, add_ref)

                loss_add = F.mse_loss(ref, add_ref)

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

                with open(os.path.join(path, "log"), "a") as log:
                    log.write(
                        "{}   {}   {:.6f}   {:.6f}   {:.6f}\n".format(
                            epoch,
                            batch_idx,
                            loss_seg.item(),
                            loss_add.item(),
                            loss.item(),
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
            for batch_idx, (sample, sample_l) in val_loop:
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
                        segm = 1.0 * (
                            seg.any(axis=1).reshape((b, 1, h, w))
                        )  # b, 1, h, w
                        # segm = segm.repeat(1,3,1,1) # b, 3, h, w
                        segmm = torch.maximum(segmm, segm)
                    #### get alignment fake label

                    n = len(sample)
                    ref = sample[n // 2 + 1].to(device)
                    ref_l = sample_l[n // 2 + 1].to(device)

                    seg_ref, add_ref = seg_model(ref_l)
                    # rotated = TF.rotate(ref, ANGLE)
                    # loss_add = F.mse_loss(rotated, add_ref)

                    # next_frame = sample[n//2+1].to(device)
                    # loss_add = F.mse_loss(next_frame, add_ref)

                    loss_add = F.mse_loss(ref, add_ref)
                    loss_seg = F.mse_loss(segmm, seg_ref)

                    epoch_loss_add += loss_add.item() * n
                    epoch_loss_seg += loss_seg.item() * n
                    cnt_sample += n
                    val_loop.set_description(f"Validation Epoch [{epoch}/{EPOCHS}]")
                    val_loop.set_postfix(
                        losses="add: {:.6f} seg: {:.6f} ".format(
                            loss_add.item(), loss_seg.item()
                        )
                    )

            with open(os.path.join(path, "log"), "a") as log:
                log.write(
                    "validation: {} {:.6f} {:.6f}\n".format(
                        epoch, epoch_loss_add / cnt_sample, epoch_loss_seg / cnt_sample,
                    )
                )

            torch.save(
                seg_model.state_dict(),
                "./checkpoints/{}/seg_model_{}.pth".format(stamp, epoch),
            )
