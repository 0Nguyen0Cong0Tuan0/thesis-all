import os
import sys
import torch
import imageio
import numpy as np

from tqdm import tqdm
from argparse import ArgumentParser

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gaussian_renderer import render_fastgs
from scene import Scene, GaussianModel
from utils.graphics_utils import patch_offsets, patch_warp
from arguments import ModelParams, PipelineParams
from extract_reflection_prior import get_multi_view_neighbor, compute_depth_edge_mask
from collections import defaultdict

def save_heatmap(tensor, path, cmap='gray'):
    # Normalize tensor to 0-255
    if tensor.max() > 0:
        tensor = tensor / tensor.max()
    tensor = (tensor * 255).byte().cpu().numpy()
    imageio.imwrite(path, tensor)

def debug_calc_ref_score(scene: Scene, gaussians: GaussianModel, pipe, bg, target_cam_name):
    patch_size = 4
    total_patch_size = (2 * patch_size + 1) ** 2
    
    viewpoint_stack = scene.getTrainCameras().copy()
    
    # 1. Find target camera
    target_cam = None
    for cam in viewpoint_stack:
        if target_cam_name in cam.image_name:
            target_cam = cam
            break
    
    if target_cam is None:
        print(f"Target camera {target_cam_name} not found!")
        return
        
    ref_image_name = target_cam.image_name
    print(f"--- Debugging Camera: {ref_image_name} ---")
    
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_output", ref_image_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}")
    
    # 2. Get Neighbors
    nearest_neighbor_dicts = get_multi_view_neighbor(scene)
    nearest_cam_info = nearest_neighbor_dicts[ref_image_name]
    nearest_cam_lst = [viewpoint_stack[idx] for (idx, _) in nearest_cam_info]
    print(f"Found {len(nearest_cam_lst)} multi-view neighbors.")
    
    # 3. Compute Depth Maps (Caching)
    print("Computing depth map for target camera and neighbors...")
    depth_map_cached = defaultdict()
    cams_to_render = [target_cam] + nearest_cam_lst
    
    zero_bg = torch.zeros_like(bg)
    xyz = gaussians.get_xyz
    
    for cam in cams_to_render:
        view_xyz = xyz @ cam.world_view_transform[:3, :3] + cam.world_view_transform[3, :3]
        depths = view_xyz[:, 2].unsqueeze(1).repeat(1, 3)
        render_pkg = render_fastgs(cam, gaussians, pipe, zero_bg, 1.0, override_color=depths)
        depth_map_cached[cam.image_name] = render_pkg["render"][0:1].detach().cpu()
        
    ref_depth = depth_map_cached[ref_image_name].cuda()
    H, W = ref_depth.squeeze().shape
    
    # Save Target Depth Map
    save_heatmap(ref_depth.squeeze(), os.path.join(out_dir, "1_depth_map.png"))
    
    # Save Target RGB
    imageio.imwrite(os.path.join(out_dir, "0_target_rgb.png"), (target_cam.original_image.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8))
    
    # 4. Projection & Variance
    ix, iy = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy')
    pixels = torch.stack([ix, iy], dim=-1).float().to(ref_depth.device)
    pts = gaussians.get_points_from_depth(target_cam, ref_depth)
    
    sampled_rgb_lst = []
    anchored_rgb = None
    
    print("Projecting and warping neighbors...")
    
    accumulated_d_mask = torch.ones((H*W), dtype=torch.bool, device="cuda")
    
    for i, nearest_cam in enumerate(nearest_cam_lst):
        pts_in_nearest_cam = pts.clone() @ nearest_cam.world_view_transform[:3,:3] + nearest_cam.world_view_transform[3,:3]
        map_z, d_mask = gaussians.get_points_depth_in_depth_map(nearest_cam, depth_map_cached[nearest_cam.image_name].cuda(), pts_in_nearest_cam)
        
        projected_z = pts_in_nearest_cam[..., 2]
        sampled_z = map_z.squeeze()
        depth_threshold = 0.05
        occlusion_mask = torch.abs(projected_z - sampled_z) < depth_threshold
        d_mask = d_mask & occlusion_mask
        
        pts_in_nearest_cam = pts_in_nearest_cam / (pts_in_nearest_cam[:,2:3] + 1e-6)
        pts_in_nearest_cam = pts_in_nearest_cam * map_z.squeeze()[...,None]
        
        if isinstance(nearest_cam.R, torch.Tensor):
            R = nearest_cam.R.float().cuda()
            T = nearest_cam.T.float().cuda()
        else:
            R = torch.tensor(nearest_cam.R).float().cuda()
            T = torch.tensor(nearest_cam.T).float().cuda()
        
        pts_ = (pts_in_nearest_cam-T)@R.transpose(-1,-2)
        pts_in_view_cam = pts_ @ target_cam.world_view_transform[:3,:3] + target_cam.world_view_transform[3,:3]
        pts_projections = torch.stack([
            pts_in_view_cam[:,0] * target_cam.Fx / (pts_in_view_cam[:,2] + 1e-6) + target_cam.Cx,
            pts_in_view_cam[:,1] * target_cam.Fy / (pts_in_view_cam[:,2] + 1e-6) + target_cam.Cy
        ], -1).float()
        
        pixel_noise = torch.norm(pts_projections - pixels.reshape(*pts_projections.shape), dim=-1)
        d_mask = d_mask & (pixel_noise < 2.0)
        d_mask = d_mask.reshape(-1)
        
        # Accumulate occlusion mask for visualization
        accumulated_d_mask = accumulated_d_mask & d_mask
        
        valid_indices = torch.arange(d_mask.shape[0], device=d_mask.device)[d_mask] 
        pixels_this_neighbor = pixels.clone().reshape(-1,2)[valid_indices]
        offsets = patch_offsets(patch_size, pixels_this_neighbor.device)
        ori_pixels_patch = pixels_this_neighbor.reshape(-1, 1, 2) + offsets.float()

        if anchored_rgb is None:
            ref_pixels_patch = pixels.clone().reshape(-1,1,2) + offsets.float()
            ref_pixels_patch[:, :, 0] = 2 * ref_pixels_patch[:, :, 0] / (W - 1) - 1.0
            ref_pixels_patch[:, :, 1] = 2 * ref_pixels_patch[:, :, 1] / (H - 1) - 1.0
            ref_image_rgb = target_cam.original_image.cuda()
            anchored_rgb = torch.nn.functional.grid_sample(ref_image_rgb.unsqueeze(0), ref_pixels_patch.reshape(1,-1,1,2), align_corners=True)
            anchored_rgb = anchored_rgb.reshape(3,-1,total_patch_size)

        ref_to_neareast_r = nearest_cam.world_view_transform[:3,:3].transpose(-1,-2) @ target_cam.world_view_transform[:3,:3]
        ref_to_neareast_t = -ref_to_neareast_r @ target_cam.world_view_transform[3,:3] + nearest_cam.world_view_transform[3,:3]

        ref_local_n = torch.tensor([0,0,1], dtype=torch.float32, device="cuda").expand(pts.shape[0], 3)[valid_indices]
        ref_local_d = (pts[:,2])[valid_indices]

        H_ref_to_neareast = ref_to_neareast_r[None] - torch.matmul(ref_to_neareast_t[None,:,None].expand(ref_local_d.shape[0],3,1), ref_local_n[:,:,None].expand(ref_local_d.shape[0],3,1).permute(0, 2, 1)) / (ref_local_d[...,None,None]+1e-6)
        
        K_nearest = torch.tensor([[nearest_cam.Fx, 0, nearest_cam.Cx], [0, nearest_cam.Fy, nearest_cam.Cy], [0, 0, 1]], dtype=torch.float32, device="cuda")
        K_inv_view = torch.tensor([[1/target_cam.Fx, 0, -target_cam.Cx/target_cam.Fx], [0, 1/target_cam.Fy, -target_cam.Cy/target_cam.Fy], [0, 0, 1]], dtype=torch.float32, device="cuda")
        
        H_ref_to_neareast = torch.matmul(K_nearest[None].expand(ref_local_d.shape[0], 3, 3), H_ref_to_neareast)
        H_ref_to_neareast = H_ref_to_neareast @ K_inv_view

        grid = patch_warp(H_ref_to_neareast.reshape(-1,3,3), ori_pixels_patch)
        grid[:, :, 0] = 2 * grid[:, :, 0] / (W - 1) - 1.0
        grid[:, :, 1] = 2 * grid[:, :, 1] / (H - 1) - 1.0
        image_rgb = nearest_cam.original_image.cuda()
        
        sampled_rgb = torch.nn.functional.grid_sample(image_rgb.unsqueeze(0), grid.reshape(1,-1,1,2), align_corners=True)
        sampled_rgb = sampled_rgb.reshape(3,-1,total_patch_size)
        
        sampled_rgb_map = torch.zeros((3,H,W,total_patch_size), dtype=torch.float16, device=sampled_rgb.device)
        sampled_rgb_map[:,pixels_this_neighbor[:,1].long(),pixels_this_neighbor[:,0].long(),:] = sampled_rgb.half()
        
        # Save warped image (center pixel of the patch)
        warped_vis = sampled_rgb_map[:,:,:,total_patch_size//2].float().permute(1,2,0).cpu().numpy()
        imageio.imwrite(os.path.join(out_dir, f"2_warped_neighbor_{i}.png"), (warped_vis * 255).astype(np.uint8))
        
        sampled_rgb_lst += [sampled_rgb_map.reshape((3,-1,total_patch_size)).cpu()]
    
    # Save combined occlusion mask
    save_heatmap(accumulated_d_mask.reshape(H,W).float(), os.path.join(out_dir, "3_occlusion_mask.png"), cmap='gray')
    
    all_sampled_rgb = torch.stack(sampled_rgb_lst).permute((0,2,3,1)).contiguous() # CPU, float16
    anchored_rgb = anchored_rgb.unsqueeze(0).permute((0,2,3,1)).contiguous().cpu().half()
    
    tot_sampled_rgb = torch.concat([all_sampled_rgb,anchored_rgb],dim=0)
    
    print("Calculating variance...")
    mean_color = tot_sampled_rgb.mean(0,keepdim=True)
    std_color = (((tot_sampled_rgb-mean_color)**2)).mean(0)**0.5
    RS_temp = std_color.float().sum(-1).mean(-1)
    
    # Save Raw Variance Map (before depth/edge filtering)
    save_heatmap(RS_temp.reshape(H,W), os.path.join(out_dir, "4_variance_raw.png"))
    
    valid_depth_mask = (ref_depth.squeeze() > 0.1).reshape(-1)
    RS_temp[~valid_depth_mask] = 0.0
    
    edge_mask = compute_depth_edge_mask(ref_depth.unsqueeze(0), threshold=0.1).squeeze().reshape(-1)
    RS_temp[~edge_mask] = 0.0
    
    # Normalize and save final Ref Score
    RS_temp_vis = RS_temp.reshape((H,W))
    if RS_temp_vis.max() > 0:
        v_max = torch.quantile(RS_temp_vis.float(), 0.95)
        RS_temp_vis = torch.clamp(RS_temp_vis / (v_max + 1e-6), 0.0, 1.0)
        
    save_heatmap(RS_temp_vis, os.path.join(out_dir, "5_final_ref_score.png"), cmap='gray')
    
    print(f"Debug finished for {ref_image_name}! Check {out_dir}")

if __name__ == "__main__":
    parser = ArgumentParser(description="Debug Ref Score")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=5000, type=int)
    parser.add_argument("--target_cam", default="train_r_0", type=str)
    args = parser.parse_args()
    
    dataset = model.extract(args)
    pipe = pipeline.extract(args)
    
    print(f"Loading Gaussians from iteration {args.iteration}")
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    debug_calc_ref_score(scene, gaussians, pipe, background, args.target_cam)
