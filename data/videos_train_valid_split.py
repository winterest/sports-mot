"""
script to split video files to train/valid/test
"""
import random
from pathlib import Path

RANSEED = 999

VALID_PATH = "/mounted/mnt-gluster/cdl-data/xliu/mot/sports_video_tracking/\
FieldAlignSegmentation/data/nfl_frames/"


def split_videos(
    videos_path=VALID_PATH, patterns="*Sideline", valid=0.1, test=0.0
):
    """
    function to split
    """
    train = 1 - valid - test
    p = Path(videos_path)
    side_videos = sorted(list(p.glob(patterns)))

    print("side videos: {}".format(len(side_videos)))

    random.shuffle(side_videos)
    num_videos = len(side_videos)
    train_ds, valid_ds, test_ds = (
        side_videos[: int(num_videos * train)],
        side_videos[
            int(num_videos * train) : int(num_videos * (train + valid))
        ],
        side_videos[int(num_videos * (train + valid)) :],
    )

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":
    split_videos()
