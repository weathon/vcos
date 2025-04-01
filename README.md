# ZS-VCOS: Zero-Shot Outperforms Supervised Video Camouflaged Object Segmentation
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/zs-vcos-zero-shot-outperforms-supervised/camouflaged-object-segmentation-on-moca-mask)](https://paperswithcode.com/sota/camouflaged-object-segmentation-on-moca-mask?p=zs-vcos-zero-shot-outperforms-supervised)
Camouflaged object segmentation presents unique challenges compared to traditional segmentation tasks, primarily due to the high similarity in patterns and colors between camouflaged objects and their backgrounds. Effective solutions to this problem have significant implications in critical areas such as pest control, defect detection, and lesion segmentation in medical imaging. Prior research has predominantly emphasized supervised or unsupervised pre-training methods, leaving zero-shot approaches significantly underdeveloped. Existing zero-shot techniques commonly utilize the Segment Anything Model (SAM) in automatic mode or rely on vision-language models to generate cues for segmentation; however, their performances remain unsatisfactory. Optical flow, commonly utilized for detecting moving objects, has demonstrated effectiveness even with camouflaged entities. Our method integrates optical flow, a vision-language model, and SAM 2 into a sequential pipeline, where the output of one component provides cues for the next. Evaluated on the MoCA-Mask dataset, our approach achieves outstanding performance improvements, significantly outperforming existing zero-shot methods by raising the mean Intersection-over-Union (mIoU) from 0.273 to 0.561. Remarkably, this simple yet effective approach also surpasses supervised methods, increasing mIoU from 0.422 to 0.561. Additionally, evaluation on the MoCA-Filter dataset demonstrates an increase in the success rate from 0.628 to 0.697 when compared with FlowSAM, a supervised transfer method. A thorough ablation study further validates the individual contributions of each component.
## Leaderboard
![leaderboard](leaderboard.png)

## Method Overview
![flowchart](flow.png)


## Performance comparison on the MoCA-Mask dataset

"SV Tr" denotes supervised training, and "SV Te" denotes supervised testing, where one frame from the video was provided to the model along with prompts. "ZS" indicates zero-shot learning, while ZS w/ PK means zero-shot with prior knowledge (since the model already knows it is looking for animals). Our method significantly outperforms all zero-shot and even supervised methods.

| Method                               | Pub.      | Setting      | $S_{\alpha}$ | $F_{\beta}^{w}$ | MAE   |
|-------------------------------------|-----------|--------------|--------------|-----------------|--------|
| SLT-Net                              | CVPR 22   | SV Tr        | 0.656        | 0.357           | 0.021 |
| ZoomNeXt                             | TPAMI 24  | SV Tr        | **0.734**    | **0.476**       | 0.010 |
| TSP-SAM(M+B)                         | CVPR 24   | SV Tr        | 0.689        | 0.444           | **0.008** |
| Gao *et al.*                         | arXiv 25  | SV Tr        | 0.709        | 0.451           | 0.008 |
|-------------------------------------|-----------|--------------|--------------|-----------------|--------|
| SAM2 Tracking                        | arXiv 24  | SV Te*       | 0.804        | 0.691           | 0.004 |
|-------------------------------------|-----------|--------------|--------------|-----------------|--------|
| SAM-PM                               | CVPRW 24  | SV Tr+Te*    | 0.728        | 0.567           | 0.009 |
| Finetuned SAM2-T + Prompts          | arXiv 24  | SV Tr+Te*    | **0.832**    | **0.726**       | **0.005** |
|-------------------------------------|-----------|--------------|--------------|-----------------|--------|
| CVP                                  | ACM MM 24 | ZS           | 0.569        | 0.196           | 0.031 |
| SAM-2-L Auto                         | arXiv 24  | ZS           | 0.447        | 0.198           | 0.250 |
| LLaVA + SAM2-L                       | arXiv 24  | ZS w/ PK     | 0.622        | 0.296           | 0.047 |
| Shikra + SAM2-L                      | arXiv 24  | ZS w/ PK     | 0.495        | 0.132           | 0.107 |
| Ours                                 | -         | ZS w/ PK     | **0.776**    | **0.628**       | **0.008** |

## Setup Instructions

### Step 1: Download MoCA-Mask with Precomputed Optical Flow
```bash
wget https://zs-vcos.weasoft.com/FMOCA.zip
```
If the server is down, download from Google Drive.
https://drive.google.com/file/d/10D-K2jXZ96BeznXuYcHwom90g6cp_L6Q/view?usp=sharing

Verify file integrity with SHA-512:
```
eda88bd52daf0b44e20d5c1c545c3f3759e5368c6101a594396f4b1acf3034f812ee7aa19b3eca9203232aa0af922a2d252feec79914b125ccb2d52cf94829cf
```

### Step 2: Download and Install SAM-2
```bash
git clone https://github.com/facebookresearch/sam2.git
mv sam2 .sam2
cd .sam2
pip3 install -e .
```

If installation fails, run:
```bash
echo -e '[build-system]\nrequires = [\n    "setuptools>=62.3.0,<75.9",\n    "torch>=2.5.1",\n    ]\nbuild-backend = "setuptools.build_meta"' > pyproject.toml
```
(See https://github.com/facebookresearch/sam2/issues/611 for more)
Then run:
```bash
pip3 install -e .
```

Download the checkpoints:
```bash
cd checkpoints
bash download_ckpts.sh
```

More details: https://github.com/facebookresearch/sam2

### Step 3: Configure and Run

Modify `run.py` to include the following runtime arguments:

- `--video_name`: name of the input video (required)
- `--log_path`: log file output path (default: `output.log`)
- `--use_motion_detection`: enable motion detection support
- `--output_dir`: output directory for processed video (default: `output`)
- `--positive_prompt`: prompt to guide object detection (default: "an animal or insect being highlighted in blue")
- `--threshold`: object detection confidence threshold (default: `0.12`)
- `--use_bgs`: enable background subtraction
- `--no_back_tracking`: enable forward-only tracking
- `--momentum`: set optical flow momentum (default: `0`)
- `--no_mean_sub`: disable mean subtraction in optical flow
- `--no_negative_prompt`: disable negative prompts in VLM
- `--box_only`: use only box prompts for SAM2

### Step 4: Evaluation

Open `eval/main_MoCa.m`, update the file paths to match your local setup, and run the script using MATLAB.

For questions, contact: wg25r@student.ubc.ca

## Testing Visualizations

### Arctic Fox – mIoU: 0.842  
![Arctic Fox](webp/arctic_fox.webp)

### Arctic Fox 3 – mIoU: 0.787  
![Arctic Fox 3](webp/arctic_fox_3.webp)

### Black Cat 1 – mIoU: 0.479  
![Black Cat 1](webp/black_cat_1.webp)

### Copperhead Snake – mIoU: 0.575  
![Copperhead Snake](webp/copperhead_snake.webp)

### Flower Crab Spider 0 – mIoU: 0.761  
![Flower Crab Spider 0](webp/flower_crab_spider_0.webp)

### Flower Crab Spider 1 – mIoU: 0.783  
![Flower Crab Spider 1](webp/flower_crab_spider_1.webp)

### Flower Crab Spider 2 – mIoU: 0.758  
![Flower Crab Spider 2](webp/flower_crab_spider_2.webp)

### Hedgehog 3 – mIoU: 0.502  
![Hedgehog 3](webp/hedgehog_3.webp)

### Ibex – mIoU: 0.615  
![Ibex](webp/ibex.webp)

### Mongoose – mIoU: 0.388  
![Mongoose](webp/mongoose.webp)

### Moth – mIoU: 0.774  
![Moth](webp/moth.webp)

### Pygmy Seahorse 0 – mIoU: 0.000  
![Pygmy Seahorse 0](webp/pygmy_seahorse_0.webp)

### Rusty Spotted Cat 0 – mIoU: 0.217  
![Rusty Spotted Cat 0](webp/rusty_spotted_cat_0.webp)

### Sand Cat 0 – mIoU: 0.613  
![Sand Cat 0](webp/sand_cat_0.webp)

### Snow Leopard 10 – mIoU: 0.468  
![Snow Leopard 10](webp/snow_leopard_10.webp)

### Stick Insect 1 – mIoU: 0.246  
![Stick Insect 1](webp/stick_insect_1.webp)
