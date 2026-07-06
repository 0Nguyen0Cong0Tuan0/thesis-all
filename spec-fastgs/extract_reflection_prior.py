# ============================================================
# Extract Reflection Prior (Tan-Ikeuchi Algorithm)
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

    progress_bar = tqdm(train_cameras, desc="Extracting Priors")

    for cam in progress_bar:
        ref_image_name = cam.image_name
        
        # original_image is [3, H, W] in [0, 1]
        img_tensor = cam.original_image.permute(1, 2, 0)
        img01 = img_tensor.cpu().numpy()
        
        # Calculate Ref Score using Shafer/Klinker
        final_score = shafer_klinker_score(img01, intensity_thresh=args.sk_intensity, sat_thresh=args.sk_saturation)
        
        # Convert to 8-bit image
        score_img = (final_score * 255).astype(np.uint8)
        
        # Save to disk
        save_path = os.path.join(save_dir, f"{ref_image_name}_ref_score.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        imageio.imwrite(save_path, score_img)

    end_time = time.time()
    print(f"Prior extraction complete in {end_time - start_time:.2f} seconds!")

if __name__ == "__main__":
    parser = ArgumentParser(description="Extract Reflection Prior using Shafer/Klinker")
    
    lp = ModelParams(parser)
    
    # Add Shafer/Klinker specific parameters
    parser.add_argument("--sk_intensity", type=float, default=0.7, help="Shafer/Klinker intensity threshold")
    parser.add_argument("--sk_saturation", type=float, default=0.2, help="Shafer/Klinker saturation threshold")

    args = parser.parse_args()

    safe_state(False)

    extract_priors(lp.extract(args), args)
