# ============================================================
# Extract Reflection Prior
# ============================================================

import os
import time
import imageio
import numpy as np
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

def _normalize_score(score):
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    score = np.clip(score, 0.0, None)
    score_max = score.max()
    if score_max > 0:
        score = score / score_max
    return score

def _soft_step(x, center, temperature=0.06):
    return 1.0 / (1.0 + np.exp(-(x - center) / max(temperature, 1e-6)))

def _box_blur_2d(x, radius):
    radius = int(radius)
    if radius <= 0:
        return x

    padded = np.pad(x, ((radius, radius), (radius, radius)), mode="reflect")
    acc = np.zeros_like(x, dtype=np.float32)
    count = 0
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            acc += padded[dy:dy + x.shape[0], dx:dx + x.shape[1]]
            count += 1
    return acc / max(count, 1)

def postprocess_score(score, gamma=1.0, quantile=0.0, smooth_radius=0):
    """
    Convert a raw specular cue into a conservative confidence map.
    This is intentionally soft: densification can use broad priors, while
    supervision/masking should trust only high-confidence regions.
    """
    score = _normalize_score(score)
    if smooth_radius > 0:
        score = _box_blur_2d(score.astype(np.float32), smooth_radius)
        score = _normalize_score(score)

    quantile = float(quantile)
    if 0.0 < quantile < 1.0 and score.max() > 0:
        pivot = np.quantile(score.reshape(-1), quantile)
        score_max = score.max()
        if score_max > pivot + 1e-6:
            score = np.clip((score - pivot) / (score_max - pivot + 1e-6), 0.0, 1.0)
        else:
            score = (score >= pivot).astype(np.float32)

    gamma = float(gamma)
    if gamma > 0 and gamma != 1.0:
        score = np.power(score, gamma)

    return _normalize_score(score)

def hybrid_confidence_score(
    img01,
    ti_thresh=0.35,
    ti_bright=0.6,
    sk_intensity=0.65,
    sk_saturation=0.3,
):
    """
    A softer confidence prior combining Tan-Ikeuchi, Shafer/Klinker, and
    local highlight contrast. It is still a prior, not ground truth.
    """
    Imax = img01.max(axis=-1)
    Imin = img01.min(axis=-1)
    saturation = 1.0 - (Imin / (Imax + 1e-6))

    tan_soft = _soft_step(Imin, ti_thresh) * _soft_step(Imax, ti_bright)
    shafer_soft = _soft_step(Imax, sk_intensity) * _soft_step(sk_saturation - saturation, 0.0)

    local_mean = _box_blur_2d(Imax.astype(np.float32), radius=3)
    local_highlight = _normalize_score(np.clip(Imax - local_mean, 0.0, None))

    gray_bright = _normalize_score(Imax * (1.0 - saturation))
    score = 0.35 * tan_soft + 0.35 * shafer_soft + 0.20 * gray_bright + 0.10 * local_highlight
    return _normalize_score(score)

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
        elif args.ref_prior_method == "hybrid":
            final_score = hybrid_confidence_score(
                img01,
                ti_thresh=args.ti_thresh,
                ti_bright=args.ti_bright,
                sk_intensity=args.sk_intensity,
                sk_saturation=args.sk_saturation,
            )
        else:
            raise ValueError(f"Unknown ref_prior_method: {args.ref_prior_method}")

        ref_score = _normalize_score(final_score)
        ref_conf = postprocess_score(
            final_score,
            gamma=args.ref_conf_gamma,
            quantile=args.ref_conf_quantile,
            smooth_radius=args.ref_conf_smooth_radius,
        )
        
        # Convert to 8-bit image
        score_img = (ref_score * 255).astype(np.uint8)
        conf_img = (ref_conf * 255).astype(np.uint8)
        
        # Save to disk
        save_path = os.path.join(save_dir, f"{ref_image_name}_ref_score.png")
        conf_path = os.path.join(save_dir, f"{ref_image_name}_ref_conf.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        imageio.imwrite(save_path, score_img)
        imageio.imwrite(conf_path, conf_img)

    end_time = time.time()
    print(f"Prior extraction complete in {end_time - start_time:.2f} seconds!")

if __name__ == "__main__":
    parser = ArgumentParser(description="Extract Reflection Prior")
    
    lp = ModelParams(parser)
    
    parser.add_argument(
        "--ref_prior_method",
        type=str,
        default="tan",
        choices=["tan", "shafer", "hybrid"],
        help="Reflection prior extractor: tan reproduces the older prior; shafer uses Shafer/Klinker; hybrid builds a softer confidence prior."
    )
    parser.add_argument("--ti_thresh", type=float, default=0.35, help="Tan-Ikeuchi intensity threshold")
    parser.add_argument("--ti_bright", type=float, default=0.6, help="Tan-Ikeuchi bright floor threshold")
    parser.add_argument("--sk_intensity", type=float, default=0.7, help="Shafer/Klinker intensity threshold")
    parser.add_argument("--sk_saturation", type=float, default=0.2, help="Shafer/Klinker saturation threshold")
    parser.add_argument("--ref_conf_gamma", type=float, default=1.0, help="Post-process gamma for extracted reflection confidence")
    parser.add_argument("--ref_conf_quantile", type=float, default=0.0, help="Optional quantile cutoff for conservative confidence maps")
    parser.add_argument("--ref_conf_smooth_radius", type=int, default=0, help="Optional box smoothing radius for extracted confidence maps")

    args = parser.parse_args()

    safe_state(False)

    extract_priors(lp.extract(args), args)
