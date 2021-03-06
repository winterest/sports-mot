import os

import torch
import torch.nn as nn

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
    print("The best device is {}".format(device))
    return device


smoothed = lambda s, l: [sum(s[i : i + l]) / l for i in range(0, len(s), l)]


def union(x, y, union_fn="max"):
    if union_fn == "max":
        return torch.max(x, y)
    elif union_fn == "relu":
        relu = nn.ReLU()
        return relu(x - y) + y


def rgb_to_l(rgb_sample):
    """"
    rgb_sample: [b,c,h,w]
    gray_sample: [b,1,h,w]
    Y = 0.299 R + 0.587 G + 0.114 B
    """

    b, _, h, w = rgb_sample.size()
    gray_sample = torch.empty_like(b, 1, h, w)
    gray_sample = (
        0.299 * rgb_sample[:, 0, :, :]
        + 0.587 * rgb_sample[:, 1, :, :]
        + 0.114 * rgb_sample[:, 2, :, :]
    )
    gray_sample = gray_sample.unsqueeze(1)
