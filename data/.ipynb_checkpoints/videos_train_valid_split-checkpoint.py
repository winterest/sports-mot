"""
script to split video files to train/valid/test
"""
import random
from pathlib import Path

RANSEED = 999

VALID_PATH = "/mounted/mnt-gluster/cdl-data/xliu/mot/sports_video_tracking/\
FieldAlignSegmentation/data/nfl_frames/"


def split_videos(videos_path=VALID_PATH, patterns="*Sideline", valid=0.1, test=0.0, soccernet=False):
    """
    function to split
    """
    train = 1 - valid - test
    p = Path(videos_path)
    valid_videos = sorted(list(p.glob(patterns)))
    if soccernet:
        valid_videos = [v/"img1" for v in valid_videos]

    print("valid videos: {}".format(len(valid_videos)))

    random.shuffle(valid_videos)
    num_videos = len(valid_videos)
    train_ds, valid_ds, test_ds = (
        valid_videos[: int(num_videos * train)],
        valid_videos[int(num_videos * train) : int(num_videos * (train + valid))],
        valid_videos[int(num_videos * (train + valid)) :],
    )

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":
    split_videos()
