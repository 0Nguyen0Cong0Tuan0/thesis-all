import os
import json
import torch
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scene.gaussian_model import GaussianModel
import math

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def main():
    model_path = "output/counter"
    source_path = "datasets/mipnerf360/counter"
    target_img_name = "DSCF5857" # Assuming camera 00000 corresponds to this

    # 1. Load Camera info
    with open(os.path.join(model_path, "cameras.json"), 'r') as f:
        cameras = json.load(f)
    
    cam_info = next((c for c in cameras if target_img_name in c['img_name']), cameras[0])
    print(f"Using camera: {cam_info['img_name']}")
    
    W = cam_info['width']
    H = cam_info['height']
    
    # Actually, the 3DGS training uses downscaled images (images_8)
    # The ref_score is also downscaled or we need to match resolution.
    # Let's load the ref_score image to get the actual training resolution
    ref_score_path = os.path.join(source_path, "reflection_prior", f"{cam_info['img_name']}_ref_score.png")
    ref_score = imageio.imread(ref_score_path)
    # If it has multiple channels, take the first one
    if len(ref_score.shape) == 3:
        ref_score = ref_score[:, :, 0]
    
    train_H, train_W = ref_score.shape
    scale_w = train_W / W
    scale_h = train_H / H
    print(f"Training resolution: {train_W}x{train_H}, scale: {scale_w}")
    
    fx = cam_info['fx'] * scale_w
    fy = cam_info['fy'] * scale_h
    fovX = focal2fov(fx, train_W)
    fovY = focal2fov(fy, train_H)
    
    # Compute projection matrix
    znear = 0.01
    zfar = 100.0
    projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, fovX=fovX, fovY=fovY).transpose(0, 1).cuda()
    
    # Compute view matrix
    position = np.array(cam_info['position'])
    rotation = np.array(cam_info['rotation'])
    # In cameras.json, W2C is stored as rotation and position
    # Actually, wait, cameras.json stores:
    # W2C[:3, :3] = camera.R.transpose() -> no wait, in camera_to_JSON, W2C is np.linalg.inv(Rt)
    # So rotation is W2C rotation, position is W2C translation
    R = torch.tensor(rotation, dtype=torch.float32)
    T = torch.tensor(position, dtype=torch.float32)
    
    world_view_transform = torch.zeros(4, 4)
    world_view_transform[:3, :3] = R
    world_view_transform[3, :3] = T
    world_view_transform[3, 3] = 1.0
    world_view_transform = world_view_transform.cuda()
    
    full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
    
    # 2. Load Gaussians
    ply_path = os.path.join(model_path, "point_cloud", "iteration_30000", "point_cloud.ply")
    gaussians = GaussianModel(sh_degree=3, asg_degree=24)
    gaussians.load_ply(ply_path)
    
    xyz = gaussians.get_xyz.cuda()
    opacity = gaussians.get_opacity.cuda()
    
    # Filter by opacity (only keep somewhat visible ones)
    valid_mask = (opacity > 0.05).squeeze()
    xyz = xyz[valid_mask]
    opacity = opacity[valid_mask]
    
    print(f"Total valid gaussians: {xyz.shape[0]}")
    
    # 3. Project to 2D
    xyz_homo = torch.cat([xyz, torch.ones_like(xyz[..., :1])], dim=-1)
    clip_space = xyz_homo @ full_proj_transform
    
    # Filter points behind camera
    in_front = clip_space[..., 3] > 0
    clip_space = clip_space[in_front]
    
    ndc_space = clip_space[..., :2] / (clip_space[..., 3:4] + 1e-6)
    ndc_space[..., 1] *= -1 # Flip Y
    
    # Convert NDC to pixel coordinates
    # NDC: [-1, 1] x [-1, 1] -> [0, W] x [0, H]
    # x = (ndc_x + 1) * W / 2
    # y = (ndc_y + 1) * H / 2
    pixel_x = ((ndc_space[..., 0] + 1.0) * train_W / 2.0).detach().cpu().numpy()
    pixel_y = ((ndc_space[..., 1] + 1.0) * train_H / 2.0).detach().cpu().numpy()
    
    # 4. Plot
    plt.figure(figsize=(20, 15))
    plt.imshow(ref_score, cmap='gray')
    plt.scatter(pixel_x, pixel_y, s=1, c='red', alpha=0.5)
    plt.xlim(0, train_W)
    plt.ylim(train_H, 0)
    plt.title("Gaussian Centers (Red) overlaid on Ref Score Mask")
    plt.axis('off')
    
    save_path = "missing_specular_analysis.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Saved visualization to {save_path}")

if __name__ == "__main__":
    main()
