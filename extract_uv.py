#!/usr/bin/env python3
import cv2
import numpy as np
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from densepose import add_densepose_config

def main():
    # 1) Build config & predictor
    cfg = get_cfg()
    add_densepose_config(cfg)
    cfg.merge_from_file("configs/densepose_rcnn_R_50_FPN_s1x.yaml")
    cfg.MODEL.WEIGHTS = "weights/densepose_rcnn_R_50_FPN_s1x.pkl"
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
    predictor = DefaultPredictor(cfg)

    # 2) Read image
    img = cv2.imread("/mnt/c/Users/user/Desktop/test.jpg")
    outputs = predictor(img)

    # 3) Grab DensePose results
    instances = outputs["instances"]
    dp_outputs = instances.pred_densepose  # list of DensePoseOutput

    # 4) For each detected person, extract (x,y,u,v) and write CSV
    for idx, dp in enumerate(dp_outputs):
        u_map = dp.u.cpu().numpy()
        v_map = dp.v.cpu().numpy()
        mask = dp.fine_segm.cpu().numpy()
        ys, xs = np.where(mask > 0)
        u_vals = u_map[ys, xs]
        v_vals = v_map[ys, xs]
        data = np.stack((ys, xs, u_vals, v_vals), axis=1)
        fn = f"uv_points_{idx}.csv"
        header = "y,x,u,v"
        np.savetxt(fn, data, delimiter=",", header=header, comments="")
        print(f"Instance {idx}: wrote {data.shape[0]} points → {fn}")

if __name__=="__main__":
    main()
