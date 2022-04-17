"""
main module
"""
import time
import os
import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader

import _init_paths
from aligning import ResSTN
from model import Model
from loss import SegmentationLoss, AlignmentLoss
from trainer import Trainer, AlignTrainer
from data.nfl_dataset import NflImagePairs, NflImageStacks
from data.videos_train_valid_split import split_videos
from logger import Logging

# opt
from opt import opts

# get the gpu id
def get_most_idle_gpu():
    """
    get the most idle gpus id:
    return the gpu id with the least memory usage
    TODO: get the gpu id with the most unused memory, but it would be dangerous to introduce memory competing
    """
    used_memory = os.popen("nvidia-smi|grep MiB|awk '{print $9}'").read()
    # total_memory = os.popen("nvidia-smi|grep MiB|awk '{print $10}'").read()

    used_memory = used_memory.split("\n")
    while "" in used_memory:
        used_memory.remove("")
    used_memory = [int(i.replace("MiB", "")) for i in used_memory if "MiB" in i]
    # total_memory = total_memory.split("\n")
    # total_memory.remove("")
    # total_memory = [int(i.replace('MiB','')) for i in total_memory]

    min_used_memory_idx = [
        i for i in range(len(used_memory)) if used_memory[i] == min(used_memory)
    ]
    device = torch.device(
        "cuda:{}".format(min_used_memory_idx[0]) if torch.cuda.is_available() else "cpu"
    )
    print("Now Using device {}".format(device))
    return device


if __name__ == "__main__":
    #### Hyper Parameters
    # DEVICE = torch.DEVICE(opt.device)
    opt = opts().parse()
    if opt.gpus == -2:
        DEVICE = "cpu"
    elif opt.gpus == -1:
        DEVICE = get_most_idle_gpu()  # DEVICE = "cuda:2"
    else:
        DEVICE = "cuda:{}".format(opt.gpus)
    opt.device = DEVICE
    EPOCHS = opt.epochs
    LR = opt.lr
    BS = opt.bs

    #### Dataset construction

    train_ds, valid_ds, test_ds = split_videos(videos_path=opt.videos_path)
    training_dataset = NflImageStacks(
        nfl_dir=train_ds,
        num_frames=2,
        size=(512, 512),
        max_interval=3,
        min_interval=0,
        mode="train",
        down_size=opt.downsize
    )

    validation_dataset = NflImageStacks(
        nfl_dir=valid_ds,
        num_frames=2,
        size=(512, 512),
        max_interval=3,
        min_interval=0,
        mode="validation",
        down_size=opt.downsize
    )

    train_loader = DataLoader(training_dataset, batch_size=BS, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=BS, shuffle=False)

    #### Model and loss definition
    if opt.task == "alignment":
        align_inc = opt.align_inc
        align_p = opt.align_p
        model = ResSTN(inc=align_inc, p=align_p)
        loss_fn = AlignmentLoss()
    else:
        model = Model(only_alpha=True)
        loss_fn = SegmentationLoss(loss_type=nn.MSELoss, union_type="max")

    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    # optimizer = optimizer.to(DEVICE)

    loss_fn = loss_fn.to(DEVICE)

    #### Trainer
    if opt.task == "alignment":
        trainer = AlignTrainer(model, optimizer, loss_fn, opt)
    else:
        trainer = Trainer(model, optimizer, loss_fn)

    #### Logging
    loss_hist = []
    time_str = time.strftime("%Y-%m-%d-%H-%M")
    logger = Logging(time_str)

    #### Training
    for epoch in range(EPOCHS):
        trainer.run_epoch(epoch, train_loader, logger, mode="train")

        trainer.run_epoch(epoch, validation_loader, logger, mode="validate")
