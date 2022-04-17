import os

# video to frames
def mkdir_ifnotexists(dir):
    if os.path.exists(dir):
        return
    os.mkdir(dir)


def video_to_frame(
    vid_file="./input_videos/sample.mp4", frame_path="./frames/"
):
    """
    docstring
    """
    mkdir_ifnotexists(frame_path)
    video_name = vid_file.split("/")[-1].split(".")[0]
    mkdir_ifnotexists(frame_path + video_name)
    cmd = "ffmpeg -i %s -start_number 0 -vsync 0 %s/frame_%%06d.png" % (
        vid_file,
        frame_path + video_name,
    )
    os.system(cmd)


if __name__ == "__main__":
    videos_path = (
        "/mounted/mnt-gluster/cdl-data/xliu/nfl-impact-detection/train/"
    )

    videos = os.listdir(videos_path)

    for video in videos:
        video_to_frame(
            vid_file=videos_path + video, frame_path="./nfl_frames/"
        )
