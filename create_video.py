#!/usr/bin/env python3
"""
Create side-by-side (RGB | depth) MP4.
Both halves are resized to the smaller of the two native resolutions.

Requires: opencv-python, numpy
"""

import argparse
import os
import glob
import cv2
import numpy as np
from tqdm import tqdm


# --------------------------- utility -----------------------------------------
def load_jpgs(folder):
    return sorted(
        f
        for f in glob.glob(os.path.join(folder, "*"))
        if (f.endswith(".jpg") or f.endswith(".png"))
        and not f.endswith("_pose.jpg")
        and not f.endswith("_pose.png")
    )


def depth_to_vis(depth_img, cmap_name=None):
    """uint16/float depth -> grayscale/colour BGR"""
    if depth_img.ndim == 3:
        depth_img = cv2.cvtColor(depth_img, cv2.COLOR_BGR2GRAY)
    depth_norm = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX)
    depth_u8 = depth_norm.astype("uint8")
    if cmap_name:
        cmap = getattr(cv2, f"COLORMAP_{cmap_name.upper()}")
        return cv2.applyColorMap(depth_u8, cmap)
    return cv2.cvtColor(depth_u8, cv2.COLOR_GRAY2BGR)


# ------------------------- main routine --------------------------------------
def make_video(rgb_dir, depth_dir, out_path, fps=30, cmap=None):
    rgb_files, depth_files = load_jpgs(rgb_dir), load_jpgs(depth_dir)

    if len(rgb_files) != len(depth_files):
        raise RuntimeError(f"#RGB={len(rgb_files)} vs #Depth={len(depth_files)}")

    # Match names – abort if any mismatch -------------------------------------
    def basename(p):
        return os.path.basename(p)

    rgb_map = {basename(p): p for p in rgb_files}
    dep_map = {basename(p): p for p in depth_files}
    common = sorted(set(rgb_map) & set(dep_map))
    if len(common) != len(rgb_files):
        diff = sorted(set(rgb_map) ^ set(dep_map))
        raise RuntimeError("Mismatched names:\n  " + "\n  ".join(diff))

    # Decide the reference (smaller) resolution --------------------------------
    rgb0 = cv2.imread(rgb_map[common[0]])
    dep0 = cv2.imread(dep_map[common[0]], cv2.IMREAD_UNCHANGED)
    h_rgb, w_rgb = rgb0.shape[:2]
    h_dep, w_dep = dep0.shape[:2]

    if w_rgb * h_rgb <= w_dep * h_dep:
        ref_w, ref_h = w_rgb, h_rgb  # RGB stream is smaller
    else:
        ref_w, ref_h = w_dep, h_dep  # Depth stream is smaller

    frame_size = (ref_w * 2, ref_h)  # width twice (RGB|DEP), same height
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_path, fourcc, fps, frame_size)

    # Process frames -----------------------------------------------------------
    for name in tqdm(common, desc="Processing frames"):
        rgb = cv2.imread(rgb_map[name])
        dep = cv2.imread(dep_map[name], cv2.IMREAD_UNCHANGED)

        rgb = cv2.resize(rgb, (ref_w, ref_h), interpolation=cv2.INTER_AREA)
        dep = cv2.resize(
            depth_to_vis(dep, cmap), (ref_w, ref_h), interpolation=cv2.INTER_NEAREST
        )

        vw.write(np.hstack((rgb, dep)))

    vw.release()
    print(f"✓ {len(common)} frames written → {out_path}")


# --------------------------- CLI wrapper -------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb_dir", default="log/s_h/forest1/test/rgb_maps")
    ap.add_argument("--depth_dir", default="log/s_h/forest1/test/depth_maps")
    ap.add_argument("--out", default="forest1-original-with-reg.mp4")
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument(
        "--colormap", default=None, help="OpenCV colour map, e.g. jet, turbo (optional)"
    )
    args = ap.parse_args()
    make_video(args.rgb_dir, args.depth_dir, args.out, args.fps, args.colormap)
