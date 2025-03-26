# %%
from skimage import data, img_as_float
bp = breakpoint
import sys
import os
import cv2
# if too far away from mask do not accpt
from PIL import Image
os.environ["DISPLAY"] = "localhost:10.0"
# %%
import requests
import argparse

# when camera is partical stationary, we can use BGS to assist, no need stationary whole time? Or we can always use moving background BGS!!!!!
# no need for moving camera bgs, check if there is place camera is fixed, use that part to do bgs and then tracking
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM 

import requests
from PIL import Image
import torch

from transformers import Owlv2Processor, Owlv2ForObjectDetection
from scipy.ndimage import center_of_mass

# check camera motion, if no motion, use BGS to assist, still use optical flow. experement on this
from camera_motion import predict_camera_motion


# args: video_name, log_path
parser = argparse.ArgumentParser(description="Video segmentation pipeline")
parser.add_argument("--video_name", type=str, required=True, help="Name of the video to process")
parser.add_argument("--log_path", type=str, default="output.log", help="Path to save the log file")
parser.add_argument("--use_motion_detection", action="store_true", help="Use motion detection to assist segmentation")
parser.add_argument("--output_dir", type=str, default="output", help="Directory to save the output video")
parser.add_argument("--positive_prompt", type=str, default="an animal or insect being highlighted in blue", help="Positive prompt for object detection")
parser.add_argument("--threshold", type=float, default=0.12, help="Threshold for object detection")
parser.add_argument("--use_bgs", action="store_true", help="Use background subtraction to assist segmentation")
parser.add_argument("--no_back_tracking", action="store_true", help="Do not use back tracking for segmentation")
parser.add_argument("--momentum", type=float, default=0, help="Momentum for optical flow")
parser.add_argument("--no_mean_sub", action="store_true", help="Do not use mean subtraction for optical flow")
parser.add_argument("--no_negative_prompt", action="store_true", help="Do not use negative prompt for VLM")
args = parser.parse_args()

video_name = args.video_name
log_path = args.log_path

processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to("cuda")

try:
  import google.colab
  IN_COLAB = True
except:
  IN_COLAB = False
  
if IN_COLAB:
    flow_dir = "./fullmoca/FlowImages_gap1/"
    img_dir = "./fullmoca/MoCA-Video-Test/"
    sam2_checkpoint = ".sam2/checkpoints/sam2.1_hiera_small.pt"
    
else:
    flow_dir = "/home/wg25r/fastdata/fullmoca/FlowImages_gap1/"
    img_dir = "/home/wg25r/fastdata/fullmoca/MoCA-Video-Test/"
    sam2_checkpoint = "../../grounded_mog/.sam2/checkpoints/sam2.1_hiera_small.pt"
    
# use points not box, because box could not encapsulate the whole object

model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
from sam2.build_sam import build_sam2_video_predictor
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device="cuda")

class BinaryConfusion:
    def __init__(self, backend="torch"):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
        assert backend in ["torch", "numpy"], "Invalid backend"
        if backend == "torch":
            import torch
            self.torch = torch
        self.backend = backend

    def update(self, y_true, y_pred):
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        assert y_true.shape == y_pred.shape
        if self.backend == "torch":
            self.tp += self.torch.sum((y_true == 1) & (y_pred == 1))
            self.fn += self.torch.sum((y_true == 1) & (y_pred == 0))
            self.fp += self.torch.sum((y_true == 0) & (y_pred == 1))
            self.tn += self.torch.sum((y_true == 0) & (y_pred == 0))
        elif self.backend == "numpy":
            self.tp += np.sum((y_true == 1) & (y_pred == 1))
            self.fn += np.sum((y_true == 1) & (y_pred == 0))
            self.fp += np.sum((y_true == 0) & (y_pred == 1))
            self.tn += np.sum((y_true == 0) & (y_pred == 0))
        else:
            raise ValueError("Invalid backend")
            
            

    def get_f1(self):
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0
        if precision + recall == 0:
            return 0
        return 2 * (precision * recall) / (precision + recall)
    
    def get_recall(self):
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0

    def get_precision(self):
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0
    
    def get_iou(self):
        return self.tp / (self.tp + self.fp + self.fn) if (self.tp + self.fp + self.fn) else 0
    
# %%
import pylab
import numpy as np

flow_images = sorted(os.listdir(flow_dir+video_name))
input_images = sorted(os.listdir(img_dir+video_name+"/Frame"))
frame0 = cv2.imread(os.path.join(img_dir+video_name+"/Frame", input_images[0]))
video_writer = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"XVID"), 5, (frame0.shape[1], frame0.shape[0]))
from tqdm import tqdm
from skimage.feature import peak_local_max

running_dx = None
running_dy = None
past_boxes = []
points = {}
history_boxes = {}
first_box = None
blended_images = []

_, _, _, _, moved = predict_camera_motion([os.path.join(img_dir, video_name, "Frame", input_images[i]) for i in range(0, len(input_images))])
bgsub = cv2.createBackgroundSubtractorMOG2()
for idx in tqdm(range(0, len(input_images) - 1)):
    flow_image = cv2.imread(os.path.join(flow_dir, video_name, flow_images[idx]))
    # read as BGR
    input_image = cv2.imread(os.path.join(img_dir+video_name+"/Frame", input_images[idx]), cv2.IMREAD_COLOR)
    bgsub.apply(input_image)
    flow_image = cv2.resize(flow_image, (input_image.shape[1], input_image.shape[0]))
    # convert flow into dx and dy
    hsv = cv2.cvtColor(flow_image, cv2.COLOR_BGR2HSV)

    angle = hsv[:, :, 0] * (2 * np.pi / 180)
    magnitude = hsv[:, :, 1] / 255.0

    dx = magnitude * np.cos(angle)
    dy = magnitude * np.sin(angle)

    if not args.no_mean_sub:
        dx = dx - dx.mean()
        dy = dy - dy.mean()
    if running_dx is None:
        running_dx = dx
        running_dy = dy
    else:
        running_dx = args.momentum * running_dx + (1 - args.momentum) * dx
        running_dy = args.momentum * running_dy + (1 - args.momentum) * dy
    # convert back to RGB
    mag, ang = cv2.cartToPolar(running_dx, running_dy)

    intensity_gray = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if args.use_bgs:
        if not moved:
            bg = bgsub.getBackgroundImage()
            diff = cv2.absdiff(input_image, bg).astype(float).mean(axis=2)
            # alpha = min(20, 255/(diff.mean() + diff.std()))
            diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            intensity_gray = diff
            intensity_gray = cv2.normalize(intensity_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # rgb_output = cv2.cvtColor(hsv_out.astype(np.uint8), cv2.COLOR_HSV2BGR)
    # THIS IS NOT INTENSITY, WHITE IS NO MOVMENT
    # intensity_gray = cv2.cvtColor(rgb_output, cv2.COLOR_BGR2GRAY)
    # color intensity is not movement intensity, need to fix it. much better. use that as seg
    
    intensity = np.stack([np.zeros_like(intensity_gray), np.zeros_like(intensity_gray), intensity_gray], axis=-1)
    # blended = input_image
    # overlay kmeans image?
    if args.use_motion_detection:
        blended = cv2.addWeighted(input_image, 1, intensity, 0.3, 0)
    else:
        blended = input_image
    blended_images.append(input_image) # try both

    image = Image.fromarray(blended)
    if args.no_negative_prompt:
      text_labels = [[args.positive_prompt]]
    else:
      text_labels = [[args.positive_prompt, "background", "logo or sign", "plant"]] # add more negative prompts
    inputs = processor(text=text_labels, images=image, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([(image.height, image.width)])
    results = processor.post_process_object_detection(
        outputs=outputs, target_sizes=target_sizes, threshold=args.threshold
    )
    result = results[0]
    boxes, scores, labels = result["boxes"], result["scores"], result["labels"]
    boxes, scores = boxes[labels == 0], scores[labels == 0]
    this_frame_points = []
    if boxes.shape[0] > 0:
        boxes = boxes[scores == scores.max()]
        x1, y1, x2, y2 = boxes[0]
        cv2.rectangle(blended, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        roi_mask = np.zeros_like(intensity_gray)
        roi_mask[int(y1):int(y2), int(x1):int(x2)] = 1
        intensity_gray = intensity_gray * roi_mask
        center = center_of_mass(np.where(intensity_gray > 10, intensity_gray, 0))
        if ~np.isnan(center[0]):
            cv2.circle(blended, (int(center[1]), int(center[0])), 5, (0, 255, 0), -1)
            this_frame_points.append((int(center[1]), int(center[0])))
        if first_box is None:
            first_box = boxes[0]
            
    points[idx] = this_frame_points
    history_boxes[idx] = boxes
    video_writer.write(blended)
    past_boxes.append(boxes)
    if len(past_boxes) > 10:
        past_boxes.pop(0)
video_writer.release()
os.system("ffmpeg -y -i output.avi -c:v libx264 -crf 23 -preset medium -movflags +faststart output.mp4 > /dev/null 2>&1")      

import shutil

try:
    os.makedirs("output")
except:
    shutil.rmtree("output")
    os.makedirs("output")
    
for i, blended in enumerate(blended_images):
    cv2.imwrite(os.path.join("output", f"{i:05d}.jpg"), blended)
    
inference_state = predictor.init_state(video_path="output")
predictor.reset_state(inference_state)


for idx in points.keys():
    ann_frame_idx = idx
    point = points[idx]
    if len(point) > 0:
        pointi = np.array([point], dtype=np.float32)
        labels = np.array([1], np.int32)
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=1,
            points=pointi,
            labels=labels,
        )
    
for idx in history_boxes.keys():
    ann_frame_idx = idx
    box = history_boxes[idx]
    if box.shape[0] > 0:
        box = box.cpu().numpy()
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=1,
            box=box,
        )
        
video_segments = {} 
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }


# do it again but on a reversed video
frames = video_path=img_dir+video_name+"/Frame"
try:
    os.makedirs("output_rev")
except:
    shutil.rmtree("output_rev")
    os.makedirs("output_rev")
    
# for i, frame_name in enumerate(sorted(os.listdir(frames))[::-1]): #need sort
#     frame = cv2.imread(os.path.join(frames, frame_name))
#     cv2.imwrite(os.path.join("output_rev", f"{i:05d}.jpg"), frame)

for i, blended in enumerate(blended_images[::-1]):
    cv2.imwrite(os.path.join("output_rev", f"{i:05d}.jpg"), blended)

inference_state = predictor.init_state(video_path="output_rev")
predictor.reset_state(inference_state)


frame_images = [cv2.imread(os.path.join("output_rev", frame_name)) for frame_name in sorted(os.listdir("output_rev"))]
for idx in points.keys():
    ann_frame_idx = len(frame_images) - idx - 1
    point = points[idx]
    if len(point) > 0:
        frame_images[ann_frame_idx] = cv2.circle(frame_images[ann_frame_idx], point[0], 5, (0, 255, 0), -1)
    
for idx in history_boxes.keys():
    ann_frame_idx = len(frame_images) - idx - 1
    box = history_boxes[idx].cpu().numpy()
    if box.shape[0] > 0:
        box = box[0]
        frame_images[ann_frame_idx] = cv2.rectangle(frame_images[ann_frame_idx], (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
    
# os.makedirs("tmp", exist_ok=True)
# for i, frame in enumerate(frame_images):
#     cv2.imwrite(os.path.join("tmp", f"{i:05d}.jpg"), frame)
    



for idx in points.keys():
    # ann_frame_idx = len(frames) - idx - 1 frame is the file name
    ann_frame_idx = len(frame_images) - idx - 1 
    point = points[idx]
    if len(point) > 0:
        pointi = np.array([point], dtype=np.float32)
        labels = np.array([1], np.int32)
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=1,
            points=pointi,
            labels=labels,
        )
    
for idx in history_boxes.keys():
    ann_frame_idx = len(frame_images) - idx - 1 
    box = history_boxes[idx]
    if box.shape[0] > 0:
        box = box.cpu().numpy()
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=1,
            box=box,
        )
        
        
        

if not args.no_back_tracking:
    video_segments_rev = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments_rev[len(frame_images) - out_frame_idx - 1] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }


# res_writter = cv2.VideoWriter(f"{video_name}.avi", cv2.VideoWriter_fourcc(*"XVID"), 10, (frame0.shape[1] * 2, frame0.shape[0] * 2))
gt_path = os.path.join(img_dir, video_name, "GT")
gt_frames = sorted(os.listdir(gt_path))
gt_images = [cv2.imread(os.path.join(gt_path, frame)) for frame in gt_frames]

confusion = BinaryConfusion(backend="numpy")
os.makedirs(f"{args.output_dir}/{video_name}", exist_ok=True)
for i in range(0, len(gt_images) - 1):
    gt = gt_images[i]
    # input_frame = cv2.imread(os.path.join(img_dir, video_name, "Frame", input_images[i]))
    input_frame = blended_images[i]
    frame_id = int(gt_frames[i].replace(".png",""))
    try:
        pred_for = video_segments[frame_id][1][0]
    except:
        pred_for = np.zeros_like(gt[:,:,0])
    
    try:
        pred_rev = video_segments_rev[frame_id][1][0]
    except:
        pred_rev = np.zeros_like(gt[:,:,0])
    pred = pred_for | pred_rev
    
    # try:
    #     pred = video_segments[frame_id][1][0]
    # except:
    #     pred = video_segments_rev[frame_id][1][0]


        
    # could just pick one instead of both 
    # could overlay seg or raw image or flow only
    
    
    
    pred_rgb = cv2.cvtColor(pred.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
    assert input_frame.shape == gt.shape == pred_rgb.shape, "Shape mismatch {} {} {}".format(input_frame.shape, gt.shape, pred_rgb.shape)
    # confusion.update(gt[..., 0] > 128, pred_rgb[..., 0] > 128)
    # slt = cv2.imread(os.path.join(img_dir, video_name, "Pred", input_images[i - 1].replace("jpg", "png")))
    cv2.imwrite(os.path.join(args.output_dir, video_name, f"{i:05d}.png"), pred_rgb)
    try:
        input_frame = cv2.putText(input_frame, f"Frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # gt = cv2.putText(gt, f"Pseudo GT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        pred_rgb = cv2.putText(pred_rgb, f"Pred", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # slt = cv2.putText(slt, f"SLT Net", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # row1 = cv2.hconcat([input_frame, gt])
        # row2 = cv2.hconcat([pred_rgb, slt])
        # res = cv2.vconcat([row1, row2])
        # res_writter.write(res)
    except:
        pass
    
# res_writter.release()
print("F1: ", confusion.get_f1(), "Recall: ", confusion.get_recall(), "Precision: ", confusion.get_precision(), "IoU: ", confusion.get_iou())
# os.system(f"ffmpeg -y -i {video_name}.avi -c:v libx264 -crf 23 -preset medium -movflags +faststart {video_name}.mp4 > /dev/null 2>&1")

with open(log_path, "a") as f:
    f.write(f"{video_name},{confusion.get_f1()},{confusion.get_iou()},{confusion.get_recall()},{confusion.get_precision()},{moved}\n")


