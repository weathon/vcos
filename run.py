import os
import numpy as np
import cv2 
import tqdm
import sys

available = sorted(os.listdir("/home/wg25r/fastdata/fullmoca/MoCA-Video-Test/"))[::-1]
import random
random.seed(484)
random.shuffle(available)
cmds = []

for i, video in enumerate(tqdm.tqdm(available)):
    # ret = os.system(f"python3 main.old.py --video_name {video} --log_path log3.csv --output_dir qwen --threshold qwen --use_motion_detection --use_bgs --positive_prompt 'NAME highlighted in blue'")  
    ret = os.system(f"python3 main_qwen.py --video_name {video} --log_path log3.csv --output_dir qwen --threshold 0.0 --use_motion_detection --use_bgs")  
    if ret != 0:
        with open("error.txt", "a") as f:
            f.write(f"error when processing {video} with threshold qwen\n")
            f.write(f"error code: {ret}\n")
            f.write("\n")
        