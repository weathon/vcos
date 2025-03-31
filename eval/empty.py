import numpy as np
import os
import cv2

available = set(os.listdir("/home/wg25r/fastdata/fullmoca/MoCA-Video-Test/"))
have = set(os.listdir("/home/wg25r/flowsam/no_momentum_full_0.12/"))
print("Difference between available and have:")
print(available - have)

for video_name in available - have:
    gt = os.listdir(f"/home/wg25r/fastdata/real/MoCA_Video/TestDataset_per_sq/{video_name}/GT")
    for i in gt:
        gt = cv2.imread(os.path.join(f"/home/wg25r/fastdata/real/MoCA_Video/TestDataset_per_sq/{video_name}/GT", i))
        pred = np.zeros_like(gt)
        os.makedirs(os.path.join("/home/wg25r/flowsam/no_momentum_full_0.12/", video_name), exist_ok=True)
        cv2.imwrite(os.path.join("/home/wg25r/flowsam/no_momentum_full_0.12/", video_name, i), pred)
        print(i)