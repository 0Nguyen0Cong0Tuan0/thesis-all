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
    files = set()  # set() dedupes: on case-insensitive filesystems (Windows), "*.png"
                   # and "*.PNG" both match and return the SAME path string, doubling
                   # every image if not deduped (harmless on Kaggle's case-sensitive
                   # Linux fs, but silently doubles the sweep's runtime locally).
    for e in exts:
        files.update(glob.glob(os.path.join(img_dir, e)))
    return sorted(files)


def resize_to_max(img, max_size):
    """PIL resize so the longest side <= max_size (no-op if already smaller)."""
    if max_size <= 0:
        return img
    w, h = img.size
    m = max(w, h)
    if m <= max_size:
        return img
    s = max_size / float(m)
    return img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)


def save_normal(out_dir, stem, n_cv, save_preview=False):
    """n_cv: [H,W,3] float in [-1,1], OpenCV camera frame. Saves .npy (+ optional .png).

    The .npy (float16) is what train.py consumes; train.py resizes it to the render
    resolution, so we deliberately store a DOWNSCALED map (see --max_size) to keep disk
    usage sane — full-res maps are ~40 MB each and exhausted Kaggle's disk.
    """
    os.makedirs(out_dir, exist_ok=True)
    norm = np.linalg.norm(n_cv, axis=2, keepdims=True)
    n_unit = np.where(norm > 1e-6, n_cv / np.maximum(norm, 1e-6), 0.0)
    np.save(os.path.join(out_dir, stem + ".npy"), n_unit.astype(np.float16))
    if save_preview:
        try:
            prev = ((n_unit * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)
            Image.fromarray(prev).save(os.path.join(out_dir, stem + ".png"))
        except Exception as e:  # preview is non-essential; never abort the run for it
            print(f"  [warn] preview save failed for {stem}: {e!r}")


def run_marigold(images, out_dir, device="cuda", max_size=1024, preview=3):
    import torch
    from diffusers import MarigoldNormalsPipeline

    pipe = MarigoldNormalsPipeline.from_pretrained(
        "prs-eth/marigold-normals-v1-1",
        torch_dtype=torch.float16,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    for k, path in enumerate(images):
        stem = os.path.splitext(os.path.basename(path))[0]
        img = resize_to_max(Image.open(path).convert("RGB"), max_size)
        out = pipe(img)
        # diffusers returns prediction as np [1,H,W,3] in [-1,1], Marigold frame
        # (x right, y UP, z toward viewer). Convert to OpenCV (y down, z into scene).
        n = np.asarray(out.prediction)[0].astype(np.float32)  # [H,W,3]
        n_cv = np.stack([n[..., 0], -n[..., 1], -n[..., 2]], axis=2)
        save_normal(out_dir, stem, n_cv, save_preview=(k < preview))
        if (k + 1) % 10 == 0 or k == 0:
            print(f"  [{k+1}/{len(images)}] {stem} -> {n_cv.shape[1]}x{n_cv.shape[0]}")


def run_dsine(images, out_dir, device="cuda", max_size=1024, preview=3):
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
        pil = resize_to_max(Image.open(path).convert("RGB"), max_size)
        img = np.asarray(pil).astype(np.float32) / 255.0
        t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            n = model(t)[0].permute(1, 2, 0).cpu().numpy()  # [H,W,3], OpenCV
        save_normal(out_dir, stem, n, save_preview=(k < preview))
        if (k + 1) % 10 == 0 or k == 0:
            print(f"  [{k+1}/{len(images)}] {stem} -> {n.shape[1]}x{n.shape[0]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-s", "--source", required=True, help="dataset root (contains the images dir)")
    ap.add_argument("-i", "--images", default="images", help="images subfolder name")
    ap.add_argument("-o", "--out", default="normals", help="output subfolder (under source)")
    ap.add_argument("--model", default="marigold", choices=["marigold", "dsine"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_size", type=int, default=1024,
                    help="resize so the longest image side <= this before estimation; "
                         "keeps .npy small (train.py resizes to render res anyway). "
                         "0 = full resolution (large!).")
    ap.add_argument("--preview", type=int, default=3,
                    help="save a PNG preview for the first N images only (disk-friendly)")
    args = ap.parse_args()

    img_dir = os.path.join(args.source, args.images)
    out_dir = os.path.join(args.source, args.out)
    images = list_images(img_dir)
    if not images:
        raise SystemExit(f"No images found in {img_dir}")
    print(f"[normal-priors] {len(images)} images | model={args.model} | "
          f"max_size={args.max_size} | -> {out_dir}")

    if args.model == "marigold":
        run_marigold(images, out_dir, args.device, args.max_size, args.preview)
    else:
        run_dsine(images, out_dir, args.device, args.max_size, args.preview)
    print("[normal-priors] done.")


if __name__ == "__main__":
    main()
