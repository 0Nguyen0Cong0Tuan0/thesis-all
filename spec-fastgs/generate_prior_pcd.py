# ============================================================
# Prior-guided Point Cloud Initialization via Space Carving
# ============================================================

import os
import json
import torch
import numpy as np
import imageio
from PIL import Image
from argparse import ArgumentParser
from plyfile import PlyData, PlyElement
from tqdm import tqdm
from pathlib import Path

def focal2fov(focal, pixels):
    return 2*np.arctan(pixels/(2*focal))

def get_projection_matrix(znear, zfar, fovX, fovY):
    tanHalfFovY = np.tan((fovY / 2))
    tanHalfFovX = np.tan((fovX / 2))

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

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'red', 'green', 'blue']
    ele64 = np.concatenate((xyz, normals, rgb), axis=1)
    for i, attr in enumerate(attributes):
        elements[attr] = ele64[:, i]
        
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def generate_prior_pcd(dataset_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Cameras from transforms_train.json
    transforms_path = os.path.join(dataset_path, "transforms_train.json")
    if not os.path.exists(transforms_path):
        print("Real dataset detected (no transforms_train.json). Space Carving requires perfect alpha masks.")
        print("Skipping Prior-guided Point Cloud Initialization. Using default COLMAP points3D.ply.")
        return

    with open(transforms_path, 'r') as f:
        transforms = json.load(f)
    
    fovx = transforms["camera_angle_x"]
    frames = transforms["frames"]
    
    print(f"Loaded {len(frames)} training frames.")

    # 2. Create Initial 3D Grid (Bounding Box [-1.5, 1.5])
    # Resolution 200x200x200 = 8,000,000 points
    grid_size = 200
    x = torch.linspace(-1.5, 1.5, grid_size, device=device)
    y = torch.linspace(-1.5, 1.5, grid_size, device=device)
    z = torch.linspace(-1.5, 1.5, grid_size, device=device)
    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
    points3D = torch.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], dim=-1)
    
    # We add a homogeneous coordinate to points3D
    points3D_hom = torch.cat([points3D, torch.ones_like(points3D[:, :1])], dim=-1) # [N, 4]
    
    print(f"Initial grid: {len(points3D)} points.")

    # 3. Space Carving (Filter out points that project to Alpha=0 in any camera)
    mask = torch.ones(len(points3D), dtype=torch.bool, device=device)
    seen_count = torch.zeros(len(points3D), dtype=torch.int32, device=device)
    
    # We also keep track of Ref Score for surviving points
    accumulated_ref_score = torch.zeros(len(points3D), dtype=torch.float32, device=device)
    
    for frame in tqdm(frames, desc="Space Carving & Ref Score Projection"):
        # Image paths
        img_rel_path = frame["file_path"] + ".png"
        img_path = os.path.join(dataset_path, img_rel_path)
        
        # Load Image Alpha
        img = Image.open(img_path)
        width, height = img.size
        fovy = focal2fov(width / (2 * np.tan(fovx / 2)), height)
        
        img_rgba = np.array(img.convert("RGBA"))
        alpha = torch.from_numpy(img_rgba[:, :, 3]).float().to(device) / 255.0 # [H, W]
        
        # Load Ref Score
        stem = Path(img_rel_path).stem
        ref_score_path = os.path.join(dataset_path, "reflection_prior", f"train_{stem}_ref_score.png")
        if os.path.exists(ref_score_path):
            ref_score_img = imageio.imread(ref_score_path)
            ref_score = torch.from_numpy(ref_score_img).float().to(device) / 255.0
        else:
            ref_score = torch.zeros_like(alpha)
            
        # Camera Matrices
        c2w = np.array(frame["transform_matrix"])
        c2w[:3, 1:3] *= -1 # Blender to COLMAP
        w2c = np.linalg.inv(c2w)
        
        R = w2c[:3, :3]
        T = w2c[:3, 3]
        
        world_view_transform = torch.eye(4, device=device)
        world_view_transform[:3, :3] = torch.from_numpy(R.T).float().to(device)
        world_view_transform[3, :3] = torch.from_numpy(T).float().to(device)
        
        projection_matrix = get_projection_matrix(znear=0.01, zfar=100.0, fovX=fovx, fovY=fovy).to(device)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        
        # Project Points
        p_hom = torch.matmul(points3D_hom, full_proj_transform) # [N, 4]
        
        # Perspective divide
        p_w = 1.0 / (p_hom[:, 3] + 1e-6)
        p_proj = p_hom[:, :3] * p_w.unsqueeze(-1)
        
        # NDC to Pixel coordinates
        px = ((p_proj[:, 0] + 1.0) * width - 1.0) * 0.5
        py = ((p_proj[:, 1] + 1.0) * height - 1.0) * 0.5
        
        # Check if points are valid (in front of camera and inside image)
        valid_z = p_hom[:, 2] > 0
        valid_x = (px >= 0) & (px < width - 1)
        valid_y = (py >= 0) & (py < height - 1)
        valid_mask = valid_z & valid_x & valid_y
        
        # Sample Alpha and Ref Score for valid points
        px_int = px[valid_mask].long()
        py_int = py[valid_mask].long()
        
        sampled_alpha = alpha[py_int, px_int]
        sampled_ref = ref_score[py_int, px_int]
        
        # 0. Mark point as seen
        seen_count[valid_mask] += 1

        # 1. Carving rule: If a point is valid in camera BUT falls on background (alpha < 0.5), it must be carved!
        carve_mask = valid_mask.clone()
        carve_mask[valid_mask] = (sampled_alpha < 0.5)
        
        mask[carve_mask] = False
        
        # 2. Ref Score Accumulation: If a point is valid and falls on foreground, accumulate its Ref Score
        acc_mask = valid_mask & (~carve_mask)
        # Update valid_mask to only those that survived
        valid_survivors = acc_mask
        
        if valid_survivors.any():
            px_surv = px[valid_survivors].long()
            py_surv = py[valid_survivors].long()
            accumulated_ref_score[valid_survivors] += ref_score[py_surv, px_surv]
            
    # Apply Carving Mask AND Discard unseen points
    mask = mask & (seen_count > 0)
    points3D = points3D[mask]
    accumulated_ref_score = accumulated_ref_score[mask]
    
    print(f"Space Carving completed. Surviving points: {len(points3D)}")
    
    # 4. Distribution Strategy (50/50 Base and Specular)
    total_budget = 100_000
    spec_budget = total_budget // 2
    base_budget = total_budget - spec_budget
    
    # Specular Points (Top N by Ref Score)
    _, top_indices = torch.topk(accumulated_ref_score, min(spec_budget, len(points3D)))
    specular_points = points3D[top_indices]
    
    # Base Points (Randomly sampled from the rest)
    # Actually, to make base points truly uniform over the surface, we sample from all surviving points
    rand_indices = torch.randperm(len(points3D))[:base_budget]
    base_points = points3D[rand_indices]
    
    final_points = torch.cat([base_points, specular_points], dim=0)
    
    # Convert to Numpy
    xyz = final_points.cpu().numpy()
    
    # Initialize Random Colors (Standard 3DGS init)
    rgb = np.random.random((len(xyz), 3)) * 255.0
    rgb = rgb.astype(np.uint8)
    
    print(f"Final Point Cloud generated: {len(xyz)} points ({len(base_points)} Base, {len(specular_points)} Specular).")
    
    # 5. Save to File
    out_path = os.path.join(dataset_path, "points3d_prior.ply")
    storePly(out_path, xyz, rgb)
    print(f"Saved prior point cloud to {out_path}")

if __name__ == "__main__":
    parser = ArgumentParser(description="Generate Prior Point Cloud via Space Carving")
    parser.add_argument("-s", "--source_path", type=str, required=True, help="Path to the dataset (e.g., data/nerf_synthetic/toaster)")
    args = parser.parse_args()
    
    generate_prior_pcd(args.source_path)
