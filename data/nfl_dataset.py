"""
NFLImageStacks
pytorch data set
"""
import random
import os
import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import PurePath
from PIL import Image  # , ImageDraw

# import torchvision.transforms.functional as TF


SEED = 999
m, s = np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5])
m_l, s_l = np.array([0.5]), np.array([0.5])

preprocess = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean=m, std=s),]
)
preprocess_l = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean=m_l, std=s_l),]
)


class NflImageStacks(Dataset):
    """
    data set for nfl frames
    """

    def __init__(
        self,
        nfl_dir=[],
        # single_video=False,
        max_interval=10,
        min_interval=1,
        num_frames=5,  # return only reference fram if 1
        size=(512, 512),
        mode="train",  # train: random interval; test: consecutive // every possible interval
        down_size=1,
    ):
        assert isinstance(nfl_dir, list), "nfl_dir should be a list of paths"
        self.dir = str(PurePath(nfl_dir[0].parent))
        self.videos = [str(PurePath(video)) for video in nfl_dir]
        self.frames = {}
        for video in self.videos:
            if os.listdir(video):
                self.frames[video] = os.listdir(video)
                self.frames[video].sort()
            else:
                self.videos.remove(video)

        """
        if isinstance(nfl_dir, str):
            self.dir = nfl_dir
            if single_video:
                self.videos = [""]
                self.frames = {"": sorted(os.listdir(self.dir + ""))}
                print(
                    "There are {} frames in this single video dataset.".format(
                        len(self.frames[""])
                    )
                )
            else:
                self.videos = os.listdir(nfl_dir)
                self.frames = {}
                for video in self.videos:
                    self.frames[video] = os.listdir(self.dir + video)
                    self.frames[video].sort()
                print("There are {} videos in this dataset.".format(len(self.videos)))
        """
        self.max_intv = max_interval
        self.min_intv = min_interval
        self.num_fram = num_frames
        if self.num_fram == 1:
            self.max_intv = 0
            self.min_intv = 0
        self.size = size
        self.down_size = down_size
        assert mode in {
            "train",
            "validation",
            "test",
        }, "dataset mode must be either train, validation or test!"
        self.mode = mode
        random.seed(SEED)

    def __len__(self):
        # return self.interval * (len(self.all_images) - (self.interval+1)//2)
        return sum([len(i) for i in self.frames.values()]) // self.down_size

    def __getitem__(self, idx):
        video = random.choice(self.videos)

        if self.mode == "validation":
            random.seed(idx + 999)

        if self.mode == "test":
            cur_frame_name = self.frames[video][idx : idx + self.num_fram]
        else:
            ref_frame_idx = random.randint(
                self.max_intv, len(self.frames[video]) - self.max_intv - 1
            )
            pool = (
                self.frames[video][
                    ref_frame_idx
                    - self.max_intv : ref_frame_idx
                    - self.min_intv
                    + 1
                ]
                + [self.frames[video][ref_frame_idx]]
                + self.frames[video][
                    ref_frame_idx
                    + self.min_intv : ref_frame_idx
                    + self.max_intv
                    + 1
                ]
            )
            cur_frame_name = sorted(random.sample(pool, k=self.num_fram))

        resized_frames = [
            Image.open(video + "/" + nm).convert("RGB").resize(self.size)
            for nm in cur_frame_name
        ]
        processed_frames = [preprocess(frm) for frm in resized_frames]

        resized_frames_l = [
            Image.open(video + "/" + nm).convert("L").resize(self.size)
            for nm in cur_frame_name
        ]
        processed_frames_l = [preprocess_l(frm) for frm in resized_frames_l]

        return processed_frames, processed_frames_l


class NflImagePairs(Dataset):
    """
    depreciated
    only pairs, which is replaced by ~Stacks with num_frames=2
    """

    def __init__(
        self,
        nfl_dir="/mounted/mnt-gluster/cdl-data/xliu/mot/sports_video_tracking/FieldAlignSegmentation/data/nfl_frames/",
        max_interval=5,
    ):
        self.dir = nfl_dir
        self.videos = os.listdir(nfl_dir)
        self.frames = {}
        for video in self.videos:
            self.frames[video] = os.listdir(self.dir + video)
            self.frames[video].sort()
        self.interval = max_interval

    def __len__(self):
        # return self.interval * (len(self.all_images) - (self.interval+1)//2)
        return sum([len(i) for i in self.frames.values()])

    def __getitem__(self, idx):
        video = random.choice(self.videos)

        ref_frame_idx = random.randint(0, len(self.frames[video]) - 1)
        pool = self.frames[video][
            max(0, ref_frame_idx - self.interval) : ref_frame_idx
            + self.interval
        ]

        cur_frame_name = random.choice(pool)

        ref_frame = Image.open(
            self.dir + video + "/" + self.frames[video][ref_frame_idx]
        )
        ref_frame = ref_frame.convert("RGB").resize((512, 512))
        cur_frame = Image.open(self.dir + video + "/" + cur_frame_name)
        cur_frame = cur_frame.convert("RGB").resize((512, 512))

        return preprocess(ref_frame), preprocess(cur_frame)


"""
class NflImageStacks_depreciated(Dataset):
    def __init__(
        self,
        nfl_dir="/mounted/mnt-gluster/cdl-data/xliu/mot/sports_video_tracking/FieldAlignSegmentation/data/nfl_frames/",
        single_video=False,
        max_interval=10,
        min_interval=0,
        num_frames=5,
        size=(512, 512),
        opt=None,
    ):
        self.dir = nfl_dir
        if single_video:
            self.videos = [""]
            self.frames = {"": sorted(os.listdir(self.dir + ""))}

        else:
            self.videos = os.listdir(nfl_dir)
            self.frames = {}
            for video in self.videos:
                self.frames[video] = os.listdir(self.dir + video)
                self.frames[video].sort()

        self.max_intv = max_interval
        self.min_intv = min_interval
        self.num_fram = num_frames
        self.size = size
        self.opt = opt

    def __len__(self):
        # return self.interval * (len(self.all_images) - (self.interval+1)//2)
        return sum([len(i) for i in self.frames.values()])

    def __getitem__(self, idx):
        video = random.choice(self.videos)

        ref_frame_idx = random.randint(
            self.max_intv, len(self.frames[video]) - self.max_intv - 1
        )
        # pool = self.frames[video][max(0,ref_frame_idx-self.max_intv):ref_frame_idx+self.max_intv]

        pool = (
            self.frames[video][
                ref_frame_idx - self.max_intv : ref_frame_idx - self.min_intv
            ]
            + self.frames[video][
                ref_frame_idx + self.min_intv : ref_frame_idx + self.max_intv
            ]
        )

        # cur_frame_name = random.choice(pool)
        cur_frame_name = sorted(random.sample(pool, k=self.num_fram))

        resized_frames = [
            Image.open(self.dir + video + "/" + nm).convert("RGB").resize(self.size)
            for nm in cur_frame_name
        ]
        processed_frames = [preprocess(frm) for frm in resized_frames]
        return processed_frames


from torch.utils.data import Dataset

m, s = np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5])
preprocess = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean=m, std=s),]
)

# frame_dir = "/mounted/mnt-gluster/cdl-data/xliu/mot/sports_video_tracking/FieldAlignSegmentation/data/nfl_frames/57583_000082_Sideline/"

"""
