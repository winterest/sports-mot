#python train_seg.py --gpus -1 --videos_path /mounted/mnt-gluster/cdl-data/liu/SoccerNet/tracking/train/ --soccernet  --vid_patterns SNMOT* --align_model_path ./best_soccer_model

python train_seg.py --gpus -1 --videos_path /mounted/mnt-gluster/cdl-data/liu/SoccerNet/tracking/train/ --soccernet  --vid_patterns SNMOT* --align_model_path ./best_valid_model