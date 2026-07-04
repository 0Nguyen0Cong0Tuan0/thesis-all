#!/usr/bin/env python3
"""
Classical (zero-ML, zero-disk, zero-domain-gap) specular-highlight detector, based on
Shafer's dichromatic reflection model (1985) / Klinker et al.: a specular highlight is a
pixel that is BRIGHT (high V) and DESATURATED (low S) relative to its surroundings,
because the specular component adds the (usually near-white) illuminant color on top of
the diffuse (colored) component, diluting saturation.

HISTORY / WHY THIS FILE HAS A SECOND GATE (validated 2026-07-04, see
results/MIPNERF_BOTTLENECK_PLAN_2026-07-01.md and test_shafer_dichromatic_specular.ipynb):
- v1 (bright+desaturated only) correctly flags TRUE highlights (metal glints, reflective
  lids) but ALSO false-positives on bright desaturated DIFFUSE materials — most starkly a
  white background (teapot: 78.6% of the image flagged, almost entirely background) and
  bright white diffuse plastic on real scenes. Same confound as the v2.5-R1 luminance mask.
- A naive fix (Gaussian-blur high-pass "local contrast") removes the flat-background
  false positive (78.6% -> ~7%) but introduces a NEW artifact: a false ring along object
  SILHOUETTE EDGES, because a linear high-pass responds to any strong edge, not just
  compact bright blobs — a boundary between a bright background and a darker object is
  itself high local-contrast.
- FIX: a specular highlight is specifically a compact BRIGHT BLOB (a local bright region
  narrower than some radius, surrounded on most sides by darker/differently-colored
  pixels), which is exactly what morphological WHITE TOP-HAT filtering
  (image - grey_opening(image, disk(radius))) extracts: it suppresses both large flat
  regions (background: opening leaves it unchanged, top-hat ~ 0) AND straight object
  edges (an opening with a disk needs radius-clearance in ALL directions, so it does not
  produce the same halo a linear high-pass does), while still finding the highlight blobs.
  Empirically verified: teapot 78.6% -> 1.97% (no ring), counter 3.48% -> 1.72% (tighter,
  same true positives). Implemented with scipy.ndimage.grey_opening (no new dependency —
  equivalent to skimage.morphology.white_tophat, cross-checked to produce identical output).

STATUS: still an APPEARANCE-ONLY detector — it can only ever approximate specularity from
a single image's brightness/saturation/local-shape statistics, unlike the geometry-gated
Reflection Score (tools/extract_reflection_score.py), which tests the actual physical
definition of specularity (view-DEPENDENT color) via multi-view consistency. Use this as
a fast, disk-light, zero-GPU offline prior — good enough on its own now that the top-hat
gate is in place, but still worth cross-checking against RS where both are available.

KNOWN RESIDUAL LIMITATION (2026-07-04, spot-checked on production sweep output): a small,
uniformly bright, convex object whose size is comparable to or smaller than tophat_radius
(e.g. a white candle/cylinder on the counter scene) can still be flagged solid — top-hat
cannot distinguish "a highlight blob on a larger surface" from "a whole small bright
object," since both are local bright regions narrower than the structuring element. This
is far rarer/smaller-impact than the original flat-background problem (confirmed via a
full-dataset sweep: counter 240 images flagged 1.44% mean / 4.22% max, teapot 600 images
1.55% mean / 2.68% max — no image anywhere near the old 78.6% failure), but it means the
prior is not perfectly clean and should still be treated as a heuristic signal, not ground
truth.

Usage:
  python tools/classical_specular_mask.py <image.png> <out_mask.png> [--sat 0.25] [--val 0.75]
"""
import argparse
import numpy as np
from PIL import Image
from scipy.ndimage import grey_opening


def _disk_footprint(radius):
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return (x ** 2 + y ** 2 <= radius ** 2)


def specular_score(img01, sat_thresh=0.25, val_thresh=0.75, tophat_radius=12, tophat_thresh=0.08):
    """img01: [H,W,3] float in [0,1]. Returns (mask [H,W] bool, score [H,W] float in [0,1]).

    score is the top-hat response gated by the raw dichromatic condition, normalized to
    [0,1] per-image — a continuous signal suitable for quantile-thresholding at use time
    (same pattern as the residual specular-loss mask and the Reflection Score), rather than
    baking a single binary threshold into the saved prior.
    """
    maxc = img01.max(axis=-1)
    minc = img01.min(axis=-1)
    V = maxc
    S = np.where(maxc > 1e-6, (maxc - minc) / (maxc + 1e-6), 0.0)
    raw_mask = (V > val_thresh) & (S < sat_thresh)

    opened = grey_opening(V, footprint=_disk_footprint(tophat_radius))
    tophat = np.clip(V - opened, 0.0, None)  # white top-hat: bright blobs only

    mask = raw_mask & (tophat > tophat_thresh)
    score = tophat * raw_mask
    if score.max() > 0:
        score = score / score.max()
    return mask, score


def classical_specular_mask(img_path, sat_thresh=0.25, val_thresh=0.75,
                             tophat_radius=12, tophat_thresh=0.08):
    """Returns (mask [H,W] bool, frac flagged, rgb_overlay [H,W,3] uint8)."""
    img01 = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
    mask, _ = specular_score(img01, sat_thresh, val_thresh, tophat_radius, tophat_thresh)
    overlay = (img01 * 255).astype(np.uint8).copy()
    overlay[mask] = [255, 0, 0]
    return mask, float(mask.mean()), overlay


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("out_mask")
    ap.add_argument("--sat", type=float, default=0.25)
    ap.add_argument("--val", type=float, default=0.75)
    ap.add_argument("--tophat_radius", type=int, default=12)
    ap.add_argument("--tophat_thresh", type=float, default=0.08)
    args = ap.parse_args()
    mask, frac, overlay = classical_specular_mask(
        args.image, args.sat, args.val, args.tophat_radius, args.tophat_thresh)
    Image.fromarray(overlay).save(args.out_mask)
    print(f"flagged {frac*100:.2f}% of pixels -> {args.out_mask}")
