#!/usr/bin/env python3
"""
Offline preprocessing sweep: precompute a Shafer dichromatic-model specular-candidate
score for every training image in a scene, BEFORE training starts.

Unlike the monocular normal prior (tools/gen_normal_priors.py, needs a GPU diffusion
model) or the Reflection Score (tools/extract_reflection_score.py, needs a TRAINED model
+ rendered depth), this sweep is pure classical CV (Shafer 1985 dichromatic model +
morphological white top-hat, see tools/classical_specular_mask.py) — no GPU, no trained
model, no external dependency beyond numpy/scipy/PIL. It can run locally on a laptop in
seconds per image, or on Kaggle before/without spending any GPU time.

For every image in <source>/<images>, writes a continuous specular-candidate score
(uint8 PNG, single channel, already gated by the raw dichromatic condition — see
classical_specular_mask.specular_score) to <source>/<out>/<image_name>.png. A small RGB
preview (red overlay on the thresholded mask) is written for the first N images only, to
keep disk light while still giving a fast visual sanity check.

Usage:
  python tools/gen_shafer_priors.py -s ./datasets/mipnerf360/counter -i images
Then train with:
  --spec_loss_mode shafer --shafer_prior_dir shafer_priors
  --spec_densify --spec_densify_locator shafer
"""
import os
import argparse
import glob
import numpy as np
from PIL import Image

from classical_specular_mask import specular_score


def list_images(img_dir):
    exts = ("*.jpg", "*.JPG", "*.jpeg", "*.png", "*.PNG")
    files = set()  # set() dedupes: on case-insensitive filesystems (Windows), "*.png"
                   # and "*.PNG" both match and return the SAME path string, doubling
                   # every image if not deduped (harmless on Kaggle's case-sensitive
                   # Linux fs, but silently doubles the sweep's runtime locally).
    for e in exts:
        files.update(glob.glob(os.path.join(img_dir, e)))
    return sorted(files)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-s", "--source", required=True, help="dataset root (contains the images dir)")
    ap.add_argument("-i", "--images", default="images", help="images subfolder name")
    ap.add_argument("-o", "--out", default="shafer_priors", help="output subfolder (under source)")
    ap.add_argument("--sat", type=float, default=0.25)
    ap.add_argument("--val", type=float, default=0.75)
    ap.add_argument("--tophat_radius", type=int, default=12)
    ap.add_argument("--tophat_thresh", type=float, default=0.08)
    ap.add_argument("--max_size", type=int, default=1600,
                    help="resize so the longest side <= this before scoring; keeps disk/"
                         "compute light on full-res real photos (0 = full resolution)")
    ap.add_argument("--preview", type=int, default=5,
                    help="save a red-overlay PNG preview for the first N images only")
    args = ap.parse_args()

    img_dir = os.path.join(args.source, args.images)
    out_dir = os.path.join(args.source, args.out)
    images = list_images(img_dir)
    if not images:
        raise SystemExit(f"No images found in {img_dir}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[shafer-priors] {len(images)} images -> {out_dir}")

    flagged_pcts = []
    for k, path in enumerate(images):
        stem = os.path.splitext(os.path.basename(path))[0]
        img = Image.open(path).convert("RGB")
        if args.max_size > 0:
            w, h = img.size
            m = max(w, h)
            if m > args.max_size:
                s = args.max_size / m
                img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
        arr01 = np.array(img).astype(np.float32) / 255.0
        mask, score = specular_score(arr01, args.sat, args.val, args.tophat_radius, args.tophat_thresh)
        flagged_pcts.append(100.0 * mask.mean())

        score_u8 = (score * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(score_u8).save(os.path.join(out_dir, stem + ".png"))

        if k < args.preview:
            overlay = (arr01 * 255).astype(np.uint8).copy()
            overlay[mask] = [255, 0, 0]
            Image.fromarray(overlay).save(os.path.join(out_dir, stem + "_preview.png"))

        if (k + 1) % 20 == 0 or k == 0:
            print(f"  [{k+1}/{len(images)}] {stem}  flagged={flagged_pcts[-1]:.2f}%")

    flagged_pcts = np.array(flagged_pcts)
    print(f"[shafer-priors] done. flagged%% mean={flagged_pcts.mean():.2f} "
          f"min={flagged_pcts.min():.2f} max={flagged_pcts.max():.2f}")
    if flagged_pcts.max() > 20.0:
        n_bad = int((flagged_pcts > 20.0).sum())
        print(f"[shafer-priors] WARNING: {n_bad} image(s) flagged >20% of pixels — "
              f"inspect their *_preview.png (if within the first {args.preview}) or "
              f"re-run with a smaller --preview cutoff covering them; a healthy true-"
              f"highlight image usually flags ~1-5%.")


if __name__ == "__main__":
    main()
