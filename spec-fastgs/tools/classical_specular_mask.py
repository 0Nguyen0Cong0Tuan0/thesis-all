#!/usr/bin/env python3
"""
Classical (zero-ML, zero-disk, zero-domain-gap) specular-highlight detector, based on
Tan & Ikeuchi's specular-free image (PAMI 2005): a specular highlight is a pixel whose
per-pixel MINIMUM channel Imin(x) is elevated. Under the dichromatic model with a white/
near-neutral illuminant, the specular component adds ~equally to all 3 channels, so
Imin(x) approximates the specular "pedestal" at that pixel — a saturated diffuse color
(e.g. pure red) has Imin~0 by definition (no specular contribution needed to explain it),
while a true highlight (all channels lifted toward white) has a large Imin.

HISTORY (2026-07-04): this file has gone through two prior production algorithms —
Shafer's dichromatic model (1985, v3.1), then Tan-Ikeuchi's Imin score gated by a plain
morphological top-hat (v3.2-draft, from test_specular_algorithms_comparison.ipynb
Algorithm 2). This is v3.2's actual FINAL algorithm: Algorithm 2b from that notebook,
which fixes a false-negative Algorithm 2 (and, by the identical mechanism, Shafer) both
have — see the two gates below.

WHY THIS FILE HAS TWO GATES (validated in test_specular_algorithms_comparison.ipynb):

GATE 1 — morphological white top-hat (fixes the FALSE-POSITIVE direction):
  Raw Imin-thresholding (plus a "bright_floor" requiring the pixel to also be bright in
  absolute terms) correctly flags TRUE highlights but ALSO false-positives on large flat
  bright/achromatic DIFFUSE surfaces — flooding windows, painted walls, a ceramic plate
  (same confound as the v2.5-R1 luminance mask and Shafer's own pre-tophat v1). FIX: a
  specular highlight is specifically a compact BRIGHT BLOB, which is exactly what
  morphological WHITE TOP-HAT filtering (Imin - grey_opening(Imin, disk(radius)))
  extracts: it suppresses large flat regions (opening leaves them unchanged, top-hat ~ 0)
  while still finding the highlight blobs.

GATE 2 — near-saturation OR-condition (fixes the FALSE-NEGATIVE direction, found later
on the kitchen scene): top-hat's rule is purely about SIZE, not cause — a genuine but
spatially BROAD highlight (camera bloom/glare off a glossy surface, measured ~105x46px
against a 24px-diameter structuring disk) gets suppressed by gate 1 exactly like a flat
wall, since grey_opening reconstructs any blob at least as wide as the disk back to
nearly its original value. Two more aggressive fixes were tried and REJECTED: multi-scale
top-hat (max response across radii 12/40/80) exploded flagged% to 35-65% (a bigger disk
also reacts to ordinary texture on large surfaces); connected-component area filtering on
the raw (pre-tophat) candidate mask failed because the missed blob is already merged into
the same giant connected component as the surrounding wall. What works: OR gate 1 with
`Imin > near_white_thresh` (default 0.85) — near the sensor's actual clipping point, not
just "bright." Ordinary room-lit materials rarely push Imin this close to 1.0 uniformly
over an area; only direct light sources or strong reflections typically do.

VALIDATION (full 240-image production-resolution sweeps, counter scene, 2026-07-04):
  Shafer (v3.1, prior production): mean=1.44% max=4.22%
  Tan-Ikeuchi + gate 1 only:        mean=7.47% max=17.96%  (~5x Shafer's false-positive rate)
  Tan-Ikeuchi + gate 1 + gate 2:    mean=7.89% max=17.96%  (gate 2 adds only +0.42pp mean,
                                                             recovers wide-highlight misses)
KNOWN TRADEOFF: this detector flags substantially more of the image as "specular" than
the prior Shafer implementation did (per-pixel Imin lacks Shafer's saturation condition,
which independently helps reject colored-but-bright diffuse pixels). Switched to
Tan-Ikeuchi anyway per project decision after reviewing
test_specular_algorithms_comparison.ipynb — monitor real training runs (Gaussian count,
PSNR/SSIM) for the same kind of regression the v2.5-R1 luminance mask caused, since a
looser locator risks diluting the specular-loss/densification signal with non-specular
pixels.

STATUS: still an APPEARANCE-ONLY detector — it can only ever approximate specularity from
a single image's brightness/local-shape statistics, unlike the geometry-gated Reflection
Score (tools/extract_reflection_score.py), which tests the actual physical definition of
specularity (view-DEPENDENT color) via multi-view consistency.

Usage:
  python tools/classical_specular_mask.py <image.png> <out_mask.png> [--bright_floor 0.6]
"""
import argparse
import numpy as np
from PIL import Image
from scipy.ndimage import grey_opening


def _disk_footprint(radius):
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return (x ** 2 + y ** 2 <= radius ** 2)


def specular_score(img01, bright_floor=0.6, tophat_radius=12, tophat_thresh=0.08,
                    near_white_thresh=0.85):
    """img01: [H,W,3] float in [0,1]. Returns (mask [H,W] bool, score [H,W] float in [0,1]).

    Tan & Ikeuchi (PAMI 2005) specular-pedestal score (Imin, the per-pixel minimum
    channel), gated by (1) a morphological white top-hat blob filter (rejects flat
    bright/achromatic backgrounds) OR (2) near-saturation (Imin close to the sensor's
    clipping point, recovers genuine highlights too spatially broad for gate 1), both
    additionally required to be bright in absolute terms (bright_floor on Imax). score is
    normalized to [0,1] per-image — a continuous signal suitable for quantile/threshold-
    based use at load time, rather than baking a single binary threshold into the saved
    prior.
    """
    Imin = img01.min(axis=-1)
    Imax = img01.max(axis=-1)

    opened = grey_opening(Imin, footprint=_disk_footprint(tophat_radius))
    tophat = np.clip(Imin - opened, 0.0, None)  # white top-hat: bright blobs only

    mask = ((tophat > tophat_thresh) | (Imin > near_white_thresh)) & (Imax > bright_floor)
    score = np.maximum(tophat, np.clip(Imin - near_white_thresh, 0.0, None) * (Imin > near_white_thresh))
    if score.max() > 0:
        score = score / score.max()
    return mask, score


def classical_specular_mask(img_path, bright_floor=0.6, tophat_radius=12, tophat_thresh=0.08,
                             near_white_thresh=0.85):
    """Returns (mask [H,W] bool, frac flagged, rgb_overlay [H,W,3] uint8)."""
    img01 = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
    mask, _ = specular_score(img01, bright_floor, tophat_radius, tophat_thresh, near_white_thresh)
    overlay = (img01 * 255).astype(np.uint8).copy()
    overlay[mask] = [255, 0, 0]
    return mask, float(mask.mean()), overlay


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("out_mask")
    ap.add_argument("--bright_floor", type=float, default=0.6)
    ap.add_argument("--tophat_radius", type=int, default=12)
    ap.add_argument("--tophat_thresh", type=float, default=0.08)
    ap.add_argument("--near_white_thresh", type=float, default=0.85)
    args = ap.parse_args()
    mask, frac, overlay = classical_specular_mask(
        args.image, args.bright_floor, args.tophat_radius, args.tophat_thresh, args.near_white_thresh)
    Image.fromarray(overlay).save(args.out_mask)
    print(f"flagged {frac*100:.2f}% of pixels -> {args.out_mask}")
