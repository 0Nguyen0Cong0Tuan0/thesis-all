#!/usr/bin/env python3
"""
Classical (zero-ML, zero-disk, zero-domain-gap) specular-highlight detector, based on
Tan & Ikeuchi's specular-free image (PAMI 2005): a specular highlight is a pixel whose
per-pixel MINIMUM channel Imin(x) is elevated. Under the dichromatic model with a white/
near-neutral illuminant, the specular component adds ~equally to all 3 channels, so
Imin(x) approximates the specular "pedestal" at that pixel — a saturated diffuse color
(e.g. pure red) has Imin~0 by definition (no specular contribution needed to explain it),
while a true highlight (all channels lifted toward white) has a large Imin.

HISTORY (2026-07-04): this file has gone through three production algorithms —
Shafer's dichromatic model (1985, v3.1), then Tan-Ikeuchi's Imin score gated by a plain
morphological top-hat (v3.2-draft, from test_specular_algorithms_comparison.ipynb
Algorithm 2, "Algorithm 2b" with the near-saturation fix below). This is v3.2's CURRENT
algorithm: Algorithm 2b PLUS an explicit desaturation gate (added after a code review
found the omission), described as GATE 3 below.

WHY THIS FILE HAS THREE GATES (validated in test_specular_algorithms_comparison.ipynb
and via full-sweep re-validation after each gate was added):

GATE 1 — morphological white top-hat (fixes the FALSE-POSITIVE-ON-FLAT-SURFACES
direction):
  Raw Imin-thresholding (plus a "bright_floor" requiring the pixel to also be bright in
  absolute terms) correctly flags TRUE highlights but ALSO false-positives on large flat
  bright/achromatic DIFFUSE surfaces — flooding windows, painted walls, a ceramic plate
  (same confound as the v2.5-R1 luminance mask and Shafer's own pre-tophat v1). FIX: a
  specular highlight is specifically a compact BRIGHT BLOB, which is exactly what
  morphological WHITE TOP-HAT filtering (Imin - grey_opening(Imin, disk(radius)))
  extracts: it suppresses large flat regions (opening leaves them unchanged, top-hat ~ 0)
  while still finding the highlight blobs.

GATE 2 — near-saturation OR-condition (fixes the FALSE-NEGATIVE-ON-WIDE-HIGHLIGHTS
direction, found later on the kitchen scene): top-hat's rule is purely about SIZE, not
cause — a genuine but spatially BROAD highlight (camera bloom/glare off a glossy
surface, measured ~105x46px against a 24px-diameter structuring disk) gets suppressed by
gate 1 exactly like a flat wall, since grey_opening reconstructs any blob at least as
wide as the disk back to nearly its original value. Two more aggressive fixes were tried
and REJECTED: multi-scale top-hat (max response across radii 12/40/80) exploded
flagged% to 35-65% (a bigger disk also reacts to ordinary texture on large surfaces);
connected-component area filtering on the raw (pre-tophat) candidate mask failed because
the missed blob is already merged into the same giant connected component as the
surrounding wall. What works: OR gate 1 with `Imin > near_white_thresh` (default 0.85) —
near the sensor's actual clipping point, not just "bright." Ordinary room-lit materials
rarely push Imin this close to 1.0 uniformly over an area; only direct light sources or
strong reflections typically do.

GATE 3 — explicit desaturation, `S < sat_thresh` (fixes a FALSE-POSITIVE-ON-COLORED-
BUT-LOCALLY-BRIGHTER-PATCHES gap found in a code review, 2026-07-04): gates 1+2 only
constrain Imin/Imax — neither checks the pixel's actual color purity (saturation). This
was a real, confirmed omission relative to the PRIOR (Shafer) production locator, which
required BOTH high V AND low S (`S = (V-min)/V`) as hard conditions. Without gate 3, any
patch that is merely LOCALLY brighter than its neighborhood (top-hat gate 1) can pass
regardless of how colored it still is — a moderately-saturated fabric highlight under
uneven lighting, for instance, satisfies gates 1+2 without being anywhere near
achromatic/white. This asymmetry (Shafer enforced desaturation; Tan-Ikeuchi as first
implemented did not) is the most likely MECHANISTIC explanation for gates-1+2's
measured "~5x more flagged than Shafer" full-sweep result below — not just an
unavoidable property of switching detector families, but a specific, fixable formula gap.
Adding `S < sat_thresh` (same formula and default threshold as Shafer's own gate,
`sat_thresh=0.25`) as a hard AND-condition on top of gates 1+2 closes this gap while
keeping gates 1+2's fixes for both false-positive/false-negative spatial failure modes.

VALIDATION (full 240-image production-resolution sweeps, counter scene):
  Shafer (v3.1, prior production):        mean=1.44% max=4.22%          (2026-07-04)
  Tan-Ikeuchi + gate 1 only:               mean=7.47% max=17.96%         (2026-07-04,
                                                                          ~5x Shafer's rate)
  Tan-Ikeuchi + gates 1+2:                 mean=7.89% max=17.96%         (2026-07-04,
                                            gate 2: +0.42pp mean, recovers wide-highlight misses)
  Tan-Ikeuchi + gates 1+2+3 (current):     mean=3.08% max=6.84%          (2026-07-05,
                                            min=0.31%) — gate 3 confirmed: cut mean
                                            flagged% by ~2.6x (7.89%->3.08%) and max by
                                            ~2.6x (17.96%->6.84%) versus gates 1+2,
                                            bringing this locator to within ~2.1x of
                                            Shafer's own mean (was ~5.5x) and ~1.6x of
                                            Shafer's max (was ~4.3x). The desaturation
                                            gap was indeed the dominant mechanistic cause
                                            of the earlier over-flagging, as hypothesized.

STATUS: still an APPEARANCE-ONLY detector — it can only ever approximate specularity from
a single image's brightness/saturation/local-shape statistics, unlike the geometry-gated
Reflection Score (tools/extract_reflection_score.py), which tests the actual physical
definition of specularity (view-DEPENDENT color) via multi-view consistency.

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
                    near_white_thresh=0.85, sat_thresh=0.25):
    """img01: [H,W,3] float in [0,1]. Returns (mask [H,W] bool, score [H,W] float in [0,1]).

    Tan & Ikeuchi (PAMI 2005) specular-pedestal score (Imin, the per-pixel minimum
    channel), gated by: (1) a morphological white top-hat blob filter (rejects flat
    bright/achromatic backgrounds) OR (2) near-saturation (Imin close to the sensor's
    clipping point, recovers genuine highlights too spatially broad for gate 1), AND
    (3) an explicit desaturation condition (S < sat_thresh, same formula/default as the
    prior Shafer locator) rejecting colored-but-locally-brighter patches that gates 1/2
    alone don't check for color purity, AND bright in absolute terms (bright_floor on
    Imax). score is normalized to [0,1] per-image — a continuous signal for
    visualization/comparison purposes (e.g. the algorithm-comparison notebook); the
    persisted on-disk prior (tools/gen_tanikeuchi_priors.py) saves the boolean `mask`
    directly rather than this per-image-relative score, so a fixed downstream threshold
    means the same thing on every image (see that file's docstring for why).
    """
    Imin = img01.min(axis=-1)
    Imax = img01.max(axis=-1)
    S = np.where(Imax > 1e-6, (Imax - Imin) / (Imax + 1e-6), 0.0)  # saturation (gate 3)

    opened = grey_opening(Imin, footprint=_disk_footprint(tophat_radius))
    tophat = np.clip(Imin - opened, 0.0, None)  # white top-hat: bright blobs only

    mask = ((tophat > tophat_thresh) | (Imin > near_white_thresh)) & (Imax > bright_floor) & (S < sat_thresh)
    score = np.maximum(tophat, np.clip(Imin - near_white_thresh, 0.0, None) * (Imin > near_white_thresh))
    if score.max() > 0:
        score = score / score.max()
    return mask, score


def classical_specular_mask(img_path, bright_floor=0.6, tophat_radius=12, tophat_thresh=0.08,
                             near_white_thresh=0.85, sat_thresh=0.25):
    """Returns (mask [H,W] bool, frac flagged, rgb_overlay [H,W,3] uint8)."""
    img01 = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
    mask, _ = specular_score(img01, bright_floor, tophat_radius, tophat_thresh, near_white_thresh, sat_thresh)
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
    ap.add_argument("--sat_thresh", type=float, default=0.25)
    args = ap.parse_args()
    mask, frac, overlay = classical_specular_mask(
        args.image, args.bright_floor, args.tophat_radius, args.tophat_thresh,
        args.near_white_thresh, args.sat_thresh)
    Image.fromarray(overlay).save(args.out_mask)
    print(f"flagged {frac*100:.2f}% of pixels -> {args.out_mask}")
