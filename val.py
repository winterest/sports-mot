from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os, random, PIL
from PIL import Image, ImageDraw

plt.ion()  # interactive mode
from main import get_most_idle_gpu

device = get_most_idle_gpu()
print("Now Using device {}".format(device))

import _init_paths

from data.nfl_dataset import NflImageStacks
from data.videos_train_valid_split import split_videos

train_ds, valid_ds, test_ds = split_videos()
NUM_FRAMES = 1

validation_dataset = NflImageStacks(
    nfl_dir=valid_ds,
    num_frames=NUM_FRAMES,
    size=(512, 512),
    max_interval=3,
    min_interval=1,
    mode="validation",
)

from tqdm import tqdm

m, s = np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5])
preprocess = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean=m, std=s),]
)

inv_normalize = transforms.Normalize(
    mean=[-mm / ss for mm, ss in zip(m, s)], std=[1 / ss for ss in s]
)

# segmentation

from segmentation import UNet, UNet_two_head

unet_inc = 3
unet_feat = 32
unet_outc = 1

# unet = UNet(in_channels=unet_inc, out_channels=unet_outc, init_features=unet_feat)
unet = UNet_two_head(
    in_channels=unet_inc,
    out_channels=unet_outc,
    init_features=unet_feat,
    out_channels_add=3,
)

seg_model = unet.to(device)

BS = 16
LR = 0.00001
EPOCHS = 100

validation_loader = DataLoader(
    validation_dataset, batch_size=BS, shuffle=False, num_workers=BS
)

losses = {"seg": [], "rot": [], "train": [], "validation": []}

model_dir = "./checkpoints/2022-04-03-06-54/"

import time

stamp = time.strftime("valid-%Y-%m-%d-%H-%M")
path = os.path.join("./checkpoints", stamp)
# Create the directory
os.makedirs(path)

if True:
    import torchvision.transforms.functional as TF

    ANGLE = 10
    seg_model.eval()
    with torch.no_grad():
        for epoch in range(EPOCHS):
            model_path = os.path.join(
                model_dir, "seg_model_{}.pth".format(epoch)
            )
            seg_model.load_state_dict(
                torch.load(
                    model_path, map_location=lambda storage, loc: storage
                )
            )

            #### validation
            epoch_loss, cnt_sample = 0, 0
            val_loop = tqdm(
                enumerate(validation_loader),
                total=len(validation_loader),
                leave=True,
            )
            for batch_idx, sample in val_loop:
                n = len(sample)
                ref = sample[n // 2].to(device)
                rotated = TF.rotate(ref, ANGLE)

                seg_ref, rot_ref = seg_model(ref)
                loss_add = F.mse_loss(rotated, rot_ref)
                epoch_loss += loss_add.item() * n
                cnt_sample += n
                val_loop.set_description(
                    f"Validation Epoch [{epoch}/{EPOCHS}]"
                )
                val_loop.set_postfix(
                    losses="rot: {:.6f}".format(loss_add.item())
                )

            with open(os.path.join(path, "val_log"), "a") as log:
                log.write(
                    "validation: {} {:.6f}\n".format(
                        epoch, epoch_loss / cnt_sample
                    )
                )
