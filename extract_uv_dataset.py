#!/usr/bin/env python3
import os
import glob
import cv2
import numpy as np
from tqdm import tqdm

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from densepose import add_densepose_config

def make_predictor():
    cfg = get_cfg()
    add_densepose_config(cfg)
    cfg.merge_from_file("configs/densepose_rcnn_R_50_FPN_s1x.yaml")
    cfg.MODEL.WEIGHTS = "weights/densepose_rcnn_R_50_FPN_s1x.pkl"
    cfg.MODEL.DEVICE  = "cpu"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
    return DefaultPredictor(cfg)

def process_image(predictor, img_path, out_dir):
    img = cv2.imread(img_path)
    outputs = predictor(img)
    instances = outputs["instances"]
    dp_outputs = instances.pred_densepose  # list of DensePoseChartPredictorOutputWithConfidences

    base = os.path.splitext(os.path.basename(img_path))[0]
    for idx, dp in enumerate(dp_outputs):
        # 1) U/V maps
        u_map = dp.u.cpu().numpy()            # (1,25,H,W) or (25,H,W)
        v_map = dp.v.cpu().numpy()

        # 2) Fine segmentation mask (binary per-part)
        fine_mask = (dp.fine_segm.cpu().numpy() > 0).astype(np.uint8)

        # 3) Coarse segmentation part IDs
        # dp.coarse_segm is a one-hot mask (25,H,W) or (1,25,H,W)
        cs = dp.coarse_segm.cpu().numpy()
        if cs.ndim == 4 and cs.shape[0] == 1:
            cs = cs.squeeze(0)               # → (25,H,W)
        # Argmax over part channels: 0=background, 1..24 = part IDs
        segm = np.argmax(cs, axis=0).astype(np.uint8)  # (H,W)

        # 4) Save everything
        out_path = os.path.join(out_dir, f"{base}_person{idx}.npz")
        np.savez_compressed(
            out_path,
            u=u_map,
            v=v_map,
            mask=fine_mask,
            segm=segm,
        )

def main(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    predictor = make_predictor()

    images = sorted(
        glob.glob(os.path.join(input_dir, "*.jpg")) +
        glob.glob(os.path.join(input_dir, "*.png"))
    )
    for img_path in tqdm(images, desc="Extracting UV maps"):
        process_image(predictor, img_path, output_dir)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(
        description="Extract per-person UV & masks with DensePose"
    )
    p.add_argument("input_dir",  help="Folder of input images (.jpg/.png)")
    p.add_argument("output_dir", help="Folder to save .npz bundles")
    args = p.parse_args()
    main(args.input_dir, args.output_dir)
