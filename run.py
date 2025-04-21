import os
import numpy as np
import cv2 
import tqdm
import sys

available = sorted(os.listdir("/home/wg25r/fastdata/fullmoca/MoCA-Video-Test/"))[::-1]

cmds = []
for thresh in [0.09, 0.13, 0.05, 0.11, 0.07, 0.03]:
    os.makedirs(f"davis_{thresh}/", exist_ok=True)
    for i, video in enumerate(tqdm.tqdm(available)):
        # ret = os.system(f"python3 main.old.py --video_name {video} --log_path log3.csv --output_dir gpt_{thresh} --threshold {thresh} --use_motion_detection --use_bgs --positive_prompt 'NAME highlighted in blue'")  
        ret = os.system(f"python3 main.old.py --video_name {video} --log_path log3.csv --output_dir gpt_{thresh} --threshold {thresh} --use_motion_detection --use_bgs")  
        if ret != 0:
            with open("error.txt", "a") as f:
                f.write(f"error when processing {video} with threshold {thresh}\n")
                f.write(f"error code: {ret}\n")
                f.write("\n")
            