#!/usr/bin/env python3
import sys
import numpy as np
import matplotlib.pyplot as plt

def main(npz_path):
    # Load the bundle
    data   = np.load(npz_path)
    u_map4 = data["u"]    # (1,25,H,W) or (25,H,W)
    v_map4 = data["v"]
    fine4  = data["mask"] # (1,25,H,W) or (25,H,W)
    segm   = data["segm"] # (H,W), 0=background, 1..24 parts

    # Print shapes & ranges
    print(f"Loaded: {npz_path}")
    print("u_map4 shape:", u_map4.shape, "min/max:", u_map4.min(), u_map4.max())
    print("v_map4 shape:", v_map4.shape, "min/max:", v_map4.min(), v_map4.max())
    print("fine4 shape:", fine4.shape, "unique:", np.unique(fine4))
    print("segm   shape:", segm.shape,   "unique:", np.unique(segm))

    # Build a true binary mask from segm
    mask2d = (segm > 0)
    print("Binary mask unique:", np.unique(mask2d))
    ys, xs = np.where(mask2d)
    print(f"Found {len(ys)} foreground pixels")

    # Sample values at first point
    if len(ys)>0:
        y0, x0 = ys[0], xs[0]
        u_vals = u_map4.squeeze(0)[:, y0, x0]
        v_vals = v_map4.squeeze(0)[:, y0, x0]
        print(f"Sample at ({y0},{x0}):")
        print(" U for parts:", u_vals)
        print(" V for parts:", v_vals)

    # Collapse for visualization
    u_vis = u_map4.squeeze(0).max(axis=0)
    v_vis = v_map4.squeeze(0).max(axis=0)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(u_vis, cmap="jet")
    axes[0].set_title("U Map"); axes[0].axis("off")
    axes[1].imshow(v_vis, cmap="jet")
    axes[1].set_title("V Map"); axes[1].axis("off")
    axes[2].imshow(mask2d, cmap="gray")
    axes[2].set_title("Coarse Mask"); axes[2].axis("off")
    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    if len(sys.argv)!=2:
        print("Usage: python inspect_uv_sample.py path/to/file.npz")
        sys.exit(1)
    main(sys.argv[1])
