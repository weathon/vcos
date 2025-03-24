import os
import numpy as np
import cv2 
import tqdm
import sys
available = os.listdir("/home/wg25r/fastdata/fullmoca/MoCA-Video-Test/") 
cmds = []
for thresh in [0.03, 0.13, 0.05, 0.11, 0.07, 0.09]:
    os.makedirs(f"no_back_tracking_{thresh}/", exist_ok=True)
    for i, video in enumerate(tqdm.tqdm(available)):
        ret = os.system(f"python3 main.old.py --video_name {video} --momentum 0.9 --log_path log3.csv --output_dir no_back_tracking_{thresh} --threshold {thresh} --use_motion_detection --use_bgs --no_back_tracking") 
        if ret != 0:
            with open("error.txt", "a") as f:
                f.write(f"error when processing {video} with threshold {thresh}\n")
                f.write(f"error code: {ret}\n")
                f.write("\n")
                
    available = set(os.listdir("/home/wg25r/fastdata/fullmoca/MoCA-Video-Test/"))
    have = set(os.listdir(f"no_back_tracking_{thresh}/"))
    print("Difference between available and have:")
    print(available - have)

    for video_name in available - have:
        gt = os.listdir(f"/home/wg25r/fastdata/fullmoca/MoCA-Video-Test/{video_name}/GT")
        for i in gt:
            gt = cv2.imread(os.path.join(f"/home/wg25r/fastdata/fullmoca/MoCA-Video-Test/{video_name}/GT", i))
            pred = np.zeros_like(gt)
            os.makedirs(os.path.join(f"no_back_tracking_{thresh}/", video_name), exist_ok=True)
            cv2.imwrite(os.path.join(f"no_back_tracking_{thresh}/", video_name, i), pred)
            print(i)