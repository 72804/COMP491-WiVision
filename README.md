# WiFi→DensePose UV Extraction Toolkit

This mini‐repository contains three key scripts for extracting and inspecting DensePose UV maps from images:

- **`extract_uv.py`**  
  Run inference on a single image and dump per-pixel `(y, x, U, V)` CSVs.

- **`extract_uv_dataset.py`**  
  Walks a folder of images, runs DensePose on each, and saves compressed `.npz` bundles containing:  
  - `u`    : per-part U maps  
  - `v`    : per-part V maps  
  - `mask` : fine-segmentation mask (binary)  
  - `segm` : coarse part-ID map (0 = background, 1–24 = parts)

- **`inspect_uv_sample.py`**  
  Loads one of the `.npz` bundles and shows:  
  1. Basic stats (shapes, min/max, unique values)  
  2. A 3-panel plot of the collapsed U heatmap, V heatmap, and true foreground mask.

---

## 📋 Prerequisites

1. **WSL2 with Ubuntu 22.04+** (or any Linux shell)  
2. **Python 3.8** (we recommend a clean Conda environment)  
3. NVIDIA GPU **_optional_** (all scripts can run on CPU)

---

## ⚙️ Environment Setup

1. **Create & activate a Conda env**  
   ```bash
   conda create -n densepose_env python=3.8 -y
   conda activate densepose_env



2.Install PyTorch (CPU‐only)
pip install torch==2.0.1+cpu torchvision==0.15.2+cpu \
  --index-url https://download.pytorch.org/whl/cpu


3.Install DensePose & dependencies
# Detectron2 w/ CPU wheels
pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch2.0/index.html

# Other libs
pip install opencv-python-headless numpy matplotlib tqdm


4.Download the DensePose model weights
mkdir -p weights && cd weights
curl -L -o densepose_rcnn_R_50_FPN_s1x.pkl \
  https://dl.fbaipublicfiles.com/detectron2/projects/DensePose/densepose_rcnn_R_50_FPN_s1x.pkl
cd ..

or manually download from [DensePose](https://github.com/facebookresearch/detectron2/tree/main/projects/DensePose/configs)


🏃‍♂️ Script Usage
1) Single‐image UV → CSV
python extract_uv.py \
  configs/densepose_rcnn_R_50_FPN_s1x.yaml \
  weights/densepose_rcnn_R_50_FPN_s1x.pkl \
  /path/to/image.jpg \
  dp_contour,bbox \
  --output image_uv.csv \
  --opts MODEL.DEVICE cpu
Output: one CSV per detected person with columns y,x,u,v.

2) Batch extract .npz UV bundles
python extract_uv_dataset.py \
  /mnt/c/Users/user/Desktop/images \
  uv_dataset
Input: folder of .jpg/.png images.
Output: uv_dataset/<image>_person<idx>.npz containing arrays u, v, mask, segm.

3) Inspect one UV bundle
python inspect_uv_sample.py uv_dataset/img001_person0.npz
Opens a 3-panel plot of:
    U map (collapsed across parts)
    V map
    Foreground mask (from segm > 0)