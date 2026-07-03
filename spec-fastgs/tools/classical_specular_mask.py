#!/usr/bin/env python3
"""
Classical (zero-ML, zero-disk, zero-domain-gap) specular-highlight candidate mask,
based on Shafer's dichromatic reflection model (1985) / Klinker et al.: a specular
highlight is a pixel that is BRIGHT (high V) and DESATURATED (low S) relative to its
surroundings, because the specular component adds the (usually near-white) illuminant
color on top of the diffuse (colored) component, diluting saturation.

STATUS (2026-07-01, validated on our own images — see
results/MIPNERF_BOTTLENECK_PLAN_2026-07-01.md "CNN/classical specular detection"
section): this threshold correctly flags TRUE specular highlights on real scenes
(metal bowl glints, reflective jar lid, baking-tray sheen on counter's GT images) but
ALSO false-positives on bright, desaturated DIFFUSE materials — most starkly, on
synthetic scenes with a white background (teapot: 78.6% of the image flagged, almost
entirely the background) and on bright white diffuse plastic on real scenes (counter:
a white plastic tube top flagged solid red). This is the SAME confound already
diagnosed for the v2.5-R1 luminance mask (results/analysis_out) and is a structural
limitation of ANY appearance-only detector (classical OR a learned CNN trained on
curated highlight datasets) — it cannot be fixed by better thresholds alone, because
"bright + desaturated" is necessarily satisfied by both true highlights and white
diffuse surfaces from a single image.

USE: as a free, instant cross-check / ensemble signal alongside the geometry-gated
Reflection Score (tools/extract_reflection_score.py), NOT as a standalone locator for
real scenes. RS is immune to this confound because it tests the actual physical
definition of specularity (view-DEPENDENT color) via multi-view consistency, rather
than single-image appearance statistics.

Usage:
  python tools/classical_specular_mask.py <image.png> <out_mask.png> [--sat 0.25] [--val 0.75]
"""
import argparse
import numpy as np
from PIL import Image


def classical_specular_mask(img_path, sat_thresh=0.25, val_thresh=0.75):
    """Returns (mask [H,W] bool, frac flagged, rgb_overlay [H,W,3] uint8)."""
    img = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
    maxc = img.max(axis=-1)
    minc = img.min(axis=-1)
    V = maxc
    S = np.where(maxc > 1e-6, (maxc - minc) / (maxc + 1e-6), 0.0)
    mask = (V > val_thresh) & (S < sat_thresh)
    overlay = (img * 255).astype(np.uint8).copy()
    overlay[mask] = [255, 0, 0]
    return mask, float(mask.mean()), overlay


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("out_mask")
    ap.add_argument("--sat", type=float, default=0.25)
    ap.add_argument("--val", type=float, default=0.75)
    args = ap.parse_args()
    mask, frac, overlay = classical_specular_mask(args.image, args.sat, args.val)
    Image.fromarray(overlay).save(args.out_mask)
    print(f"flagged {frac*100:.2f}% of pixels -> {args.out_mask}")
