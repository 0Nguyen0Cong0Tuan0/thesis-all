#!/usr/bin/env python3
"""
Generate monocular surface-normal priors for a COLMAP/Mip-NeRF360 dataset
(root cause B fix — DN-Splatter style normal supervision, arXiv:2403.17822).

Run ONCE per scene (offline preprocessing). For every image in <source>/<images>
it estimates a per-pixel surface normal map and writes it to <source>/<out>/<stem>.npy
as float16 [H, W, 3] in [-1, 1], expressed in the **OpenCV / COLMAP camera frame**
(x → right, y → down, z → into the scene; a surface facing the camera therefore has
negative z). train.py transforms the rendered world-space Gaussian normals into this
exact frame before the cosine loss, so the conventions match by construction.

A small RGB preview (<stem>.png, normal*0.5+0.5) is written next to each .npy for a
quick visual sanity check.

Estimators (--model):
  marigold  (default) — diffusers MarigoldNormalsPipeline, pip-only, easiest on Kaggle:
                pip install -q diffusers transformers accelerate
             Marigold's native frame is (x right, y UP, z toward viewer); this script
             converts it to OpenCV (flip y and z) so the saved maps are always OpenCV.
  dsine     — DSINE (https://github.com/baegwangbin/DSINE), native OpenCV frame, no
              conversion. Requires the DSINE repo + checkpoint on PYTHONPATH; see --help.

Usage (Kaggle):
  python tools/gen_normal_priors.py -s ./datasets/mipnerf360/counter -i images --model marigold
Then train with:  --normal_prior_weight 0.05 --normal_prior_dir normals
"""
import os
import argparse
import glob
import numpy as np
from PIL import Image


def list_images(img_dir):
    exts = ("*.jpg", "*.JPG", "*.jpeg", "*.png", "*.PNG")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(img_dir, e)))
    return sorted(files)


def save_normal(out_dir, stem, n_cv):
    """n_cv: [H,W,3] float in [-1,1], OpenCV camera frame. Saves .npy + .png preview."""
    os.makedirs(out_dir, exist_ok=True)
    # re-normalize to unit length (guard against tiny drift), keep zeros as zeros
    norm = np.linalg.norm(n_cv, axis=2, keepdims=True)
    n_unit = np.where(norm > 1e-6, n_cv / np.maximum(norm, 1e-6), 0.0)
    np.save(os.path.join(out_dir, stem + ".npy"), n_unit.astype(np.float16))
    prev = ((n_unit * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(prev).save(os.path.join(out_dir, stem + ".png"))


def run_marigold(images, out_dir, device="cuda"):
    import torch
    from diffusers import MarigoldNormalsPipeline

    pipe = MarigoldNormalsPipeline.from_pretrained(
        "prs-eth/marigold-normals-v1-1",
        torch_dtype=torch.float16,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    for k, path in enumerate(images):
        stem = os.path.splitext(os.path.basename(path))[0]
        img = Image.open(path).convert("RGB")
        out = pipe(img)
        # diffusers returns prediction as np [1,H,W,3] in [-1,1], Marigold frame
        # (x right, y UP, z toward viewer). Convert to OpenCV (y down, z into scene).
        n = np.asarray(out.prediction)[0].astype(np.float32)  # [H,W,3]
        n_cv = np.stack([n[..., 0], -n[..., 1], -n[..., 2]], axis=2)
        save_normal(out_dir, stem, n_cv)
        if (k + 1) % 10 == 0 or k == 0:
            print(f"  [{k+1}/{len(images)}] {stem}")


def run_dsine(images, out_dir, device="cuda"):
    # DSINE outputs OpenCV-frame normals directly (no conversion).
    import torch
    try:
        from dsine import DSINE  # depends on how the repo is installed
    except Exception as e:
        raise SystemExit(
            "DSINE not importable. Clone https://github.com/baegwangbin/DSINE, add it to "
            "PYTHONPATH, and place the checkpoint per its README. Original error: " + repr(e)
        )
    model = DSINE().to(device).eval()
    for k, path in enumerate(images):
        stem = os.path.splitext(os.path.basename(path))[0]
        img = np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
        t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            n = model(t)[0].permute(1, 2, 0).cpu().numpy()  # [H,W,3], OpenCV
        save_normal(out_dir, stem, n)
        if (k + 1) % 10 == 0 or k == 0:
            print(f"  [{k+1}/{len(images)}] {stem}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-s", "--source", required=True, help="dataset root (contains the images dir)")
    ap.add_argument("-i", "--images", default="images", help="images subfolder name")
    ap.add_argument("-o", "--out", default="normals", help="output subfolder (under source)")
    ap.add_argument("--model", default="marigold", choices=["marigold", "dsine"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    img_dir = os.path.join(args.source, args.images)
    out_dir = os.path.join(args.source, args.out)
    images = list_images(img_dir)
    if not images:
        raise SystemExit(f"No images found in {img_dir}")
    print(f"[normal-priors] {len(images)} images | model={args.model} | -> {out_dir}")

    if args.model == "marigold":
        run_marigold(images, out_dir, args.device)
    else:
        run_dsine(images, out_dir, args.device)
    print("[normal-priors] done.")


if __name__ == "__main__":
    main()
