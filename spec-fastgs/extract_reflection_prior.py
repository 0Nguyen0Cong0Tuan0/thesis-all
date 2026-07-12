# ============================================================
# Extract Reflection Prior
# ============================================================

import os
import time
import imageio
import numpy as np
from scipy.ndimage import grey_opening
from tqdm import tqdm
from argparse import ArgumentParser

from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from arguments import ModelParams

def tan_ikeuchi_score(img01, thresh=0.35, bright_floor=0.6):
    """
    Tan-Ikeuchi style specular prior. This is the older ref-score extractor
    used by early spec-fastgs runs.
    """
    Imin = img01.min(axis=-1)
    Imax = img01.max(axis=-1)

    score = Imin
    mask = (score > thresh) & (Imax > bright_floor)
    final_score = np.where(mask, score, 0.0)

    if final_score.max() > 0:
        final_score = final_score / final_score.max()

    return final_score

def shafer_klinker_score(img01, intensity_thresh=0.7, sat_thresh=0.2):
    """
    Shafer/Klinker Dichromatic Reflection Model approximation.
    Identifies pixels that are bright (high intensity) and white/gray (low saturation).
    """
    Imax = img01.max(axis=-1)
    Imin = img01.min(axis=-1)
    
    # Saturation (avoid division by zero)
    saturation = 1.0 - (Imin / (Imax + 1e-6))
    
    # Mask: Bright AND Low Saturation
    mask = (Imax > intensity_thresh) & (saturation < sat_thresh)
    
    # Score: How close it is to perfect white specular
    # Higher Imax and lower saturation -> higher score
    score = Imax * (1.0 - saturation)
    
    final_score = np.where(mask, score, 0.0)

    if final_score.max() > 0:
        final_score = final_score / final_score.max()

    return final_score

def _disk_footprint(radius):
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return (x ** 2 + y ** 2 <= radius ** 2)

def shafer_tophat_score(img01, val_thresh=0.75, sat_thresh=0.25, tophat_radius=12, tophat_thresh=0.08):
    """
    Shafer/Klinker dichromatic threshold (bright + desaturated) gated by a
    morphological white top-hat filter on value V. The plain threshold in
    shafer_klinker_score()/tan_ikeuchi_score() has no spatial-compactness
    constraint, so it false-positives hard on large flat bright-diffuse
    regions (white backgrounds, glossy-but-non-specular plastic). Top-hat
    (V minus its grey-opening by a disk of `tophat_radius`) measures how much
    a pixel stands out from its local neighborhood, which true highlights do
    and flat bright surfaces do not.

    Ported from test_specular_algorithms_comparison.ipynb's validated inline
    fallback for tools/classical_specular_mask.py (that canonical module does
    not exist in this repo). Measured there on counter/teapot: raw
    Shafer/Klinker flags ~15% mean / ~25% max pixels; this top-hat-gated
    version flags ~1.4-1.5% mean, ~4.2% max, with no cherry-picking.
    """
    maxc = img01.max(axis=-1)
    minc = img01.min(axis=-1)
    V = maxc
    S = np.where(maxc > 1e-6, (maxc - minc) / (maxc + 1e-6), 0.0)

    raw_mask = (V > val_thresh) & (S < sat_thresh)

    opened = grey_opening(V, footprint=_disk_footprint(int(tophat_radius)))
    tophat = np.clip(V - opened, 0.0, None)

    mask = raw_mask & (tophat > tophat_thresh)
    score = tophat * raw_mask

    final_score = np.where(mask, score, 0.0)
    if final_score.max() > 0:
        final_score = final_score / final_score.max()

    return final_score

def extract_priors(dataset, args):
    start_time = time.time()

    # Load dataset cameras
    # We only need SH degree 0 just to initialize the scene loader
    gaussians = GaussianModel(0)
    scene = Scene(dataset, gaussians)
    
    train_cameras = scene.getTrainCameras()
    print(f"Loaded {len(train_cameras)} training cameras.")
    
    # Create reflection_prior folder
    save_dir = os.path.join(dataset.source_path, "reflection_prior")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Output directory: {save_dir}")

    progress_bar = tqdm(train_cameras, desc=f"Extracting Priors ({args.ref_prior_method})")

    for cam in progress_bar:
        ref_image_name = cam.image_name
        
        # original_image is [3, H, W] in [0, 1]
        img_tensor = cam.original_image.permute(1, 2, 0)
        img01 = img_tensor.cpu().numpy()
        
        if args.ref_prior_method == "tan":
            final_score = tan_ikeuchi_score(
                img01,
                thresh=args.ti_thresh,
                bright_floor=args.ti_bright
            )
        elif args.ref_prior_method == "shafer":
            final_score = shafer_klinker_score(
                img01,
                intensity_thresh=args.sk_intensity,
                sat_thresh=args.sk_saturation
            )
        elif args.ref_prior_method == "shafer_tophat":
            final_score = shafer_tophat_score(
                img01,
                val_thresh=args.tophat_val_thresh,
                sat_thresh=args.tophat_sat_thresh,
                tophat_radius=args.tophat_radius,
                tophat_thresh=args.tophat_thresh
            )
        else:
            raise ValueError(f"Unknown ref_prior_method: {args.ref_prior_method}")
        
        # Convert to 8-bit image
        score_img = (final_score * 255).astype(np.uint8)
        
        # Save to disk
        save_path = os.path.join(save_dir, f"{ref_image_name}_ref_score.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        imageio.imwrite(save_path, score_img)

    end_time = time.time()
    print(f"Prior extraction complete in {end_time - start_time:.2f} seconds!")

if __name__ == "__main__":
    parser = ArgumentParser(description="Extract Reflection Prior")
    
    lp = ModelParams(parser)
    
    parser.add_argument(
        "--ref_prior_method",
        type=str,
        default="tan",
        choices=["tan", "shafer", "shafer_tophat"],
        help=(
            "Reflection prior extractor: tan reproduces the older Tan-Ikeuchi-style prior; "
            "shafer uses plain Shafer/Klinker; shafer_tophat additionally gates by a "
            "morphological top-hat filter to reject flat bright-diffuse false positives "
            "(opt-in, does not change tan/shafer behavior)."
        )
    )
    parser.add_argument("--ti_thresh", type=float, default=0.35, help="Tan-Ikeuchi intensity threshold")
    parser.add_argument("--ti_bright", type=float, default=0.6, help="Tan-Ikeuchi bright floor threshold")
    parser.add_argument("--sk_intensity", type=float, default=0.7, help="Shafer/Klinker intensity threshold")
    parser.add_argument("--sk_saturation", type=float, default=0.2, help="Shafer/Klinker saturation threshold")
    parser.add_argument("--tophat_val_thresh", type=float, default=0.75, help="shafer_tophat: value (brightness) threshold")
    parser.add_argument("--tophat_sat_thresh", type=float, default=0.25, help="shafer_tophat: saturation threshold")
    parser.add_argument("--tophat_radius", type=int, default=12, help="shafer_tophat: morphological opening disk radius (px)")
    parser.add_argument("--tophat_thresh", type=float, default=0.08, help="shafer_tophat: minimum top-hat response to flag a pixel")

    args = parser.parse_args()

    safe_state(False)

    extract_priors(lp.extract(args), args)
