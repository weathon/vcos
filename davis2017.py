# %%
from skimage import data, img_as_float
import sys
import os
import cv2
# if too far away from mask do not accpt
from PIL import Image
os.environ["DISPLAY"] = "localhost:10.0"
# %%
import requests
import argparse

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM 

import requests
from PIL import Image
import torch

from transformers import Owlv2Processor, Owlv2ForObjectDetection
from scipy.ndimage import center_of_mass

from camera_motion import predict_camera_motion

from openai import OpenAI
import base64
import io
from pydantic import BaseModel, Field
from typing import List
client = OpenAI()

class ObjectList(BaseModel):
    object_list: List[str] = Field(
        description="List of objects",
    )

def get_prompt(image):
    image = Image.fromarray(image)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    completion = client.beta.chat.completions.parse(
                model="gpt-4.5-preview",
                messages=[
                    {"role": "system", "content": f"Use less than 5 words to describe the main object(s) that can move in the image. all lower case, no punctuation. Each object should be sperated. Each object only need to be listed once if they appear mutiple times. Make each instance as a whole, not a part of it. For example, only give car instead of car and windows. List all instances you can see. "},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/jpeg;base64," + image_base64},
                            },
                        ],
                    }                    
                ],
                response_format=ObjectList,
                temperature=0
            )
    print(completion.choices[0].message.content)
    return completion.choices[0].message.content
    



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
parser.add_argument("--no_negative_prompt", action="store_true", help="Do not use negative prompt for VLM")
parser.add_argument("--box_only", action="store_true", help="Only use box as prompt for SAM2")
args = parser.parse_args()

video_name = args.video_name
log_path = args.log_path

processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to("cuda")

img_dir = "/home/wg25r/DAVIS/JPEGImages/480p/"
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
import json
import torchvision

input_images = sorted(os.listdir(img_dir+video_name))
frame0 = cv2.imread(os.path.join(img_dir+video_name, input_images[0]))
frame0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
positive_prompt = [args.positive_prompt.replace("NAME", i) for i in json.loads(get_prompt(frame0))["object_list"]]
frame0 = Image.fromarray(frame0)

inputs = processor(text=[positive_prompt], images=frame0, return_tensors="pt").to("cuda")
with torch.no_grad():
    outputs = model(**inputs)

target_sizes = torch.tensor([(frame0.height, frame0.width)])
results = processor.post_process_object_detection(
    outputs=outputs, target_sizes=target_sizes, threshold=args.threshold
)
result = results[0]
boxes, scores, labels = result["boxes"], result["scores"], result["labels"]

filtered_boxes = []

for class_id in range(len(positive_prompt)):
    class_boxes = boxes[labels == class_id]
    inx = torchvision.ops.nms(class_boxes, scores[labels == class_id], iou_threshold=0.5)
    filtered_boxes.extend(class_boxes[inx].tolist())
    
frame0 = np.array(frame0)
for box in filtered_boxes:
    box = box
    frame0 = cv2.rectangle(frame0, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)

Image.fromarray(frame0).save("output.jpg")
inference_state = predictor.init_state(video_path=img_dir+video_name)
predictor.reset_state(inference_state)


for i, box in enumerate(filtered_boxes):
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=i,
        box=box,
    )
            
video_segments = {} 
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }


with open("color.txt", "r") as f:
    colors = [line.split(" ") for line in f.read().split("\n") if line]

video_writer = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"XVID"), 5, (frame0.shape[1] * 2, frame0.shape[0]))
import os
os.makedirs("davis2017.2/"+video_name, exist_ok=True)
for i, segment in enumerate(video_segments):
    canvas = np.zeros_like(frame0)
    gray_canvas = np.zeros_like(frame0)[:,:,0]
    for object_id in video_segments[segment].keys():
        color = colors[object_id + 1]
        canvas[video_segments[segment][object_id][0]] = color
        gray_canvas[video_segments[segment][object_id][0]] = object_id + 1 
    current_frame = cv2.imread(os.path.join(img_dir+video_name, input_images[segment]))
    visualized = cv2.hconcat([current_frame, canvas])
    visualized = cv2.cvtColor(visualized, cv2.COLOR_RGB2BGR)
    video_writer.write(visualized)
    cv2.imwrite(os.path.join("davis2017.2/"+video_name, f"%05d.png" % i), gray_canvas)
    
video_writer.release()

# convert to h264
os.system("ffmpeg -i output.avi -c:v libx264 -crf 23 -preset medium -y output.mp4")
os.makedirs("davis2/"+video_name, exist_ok=True)
os.rename("output.mp4", os.path.join("davis2/"+video_name, "output.mp4"))