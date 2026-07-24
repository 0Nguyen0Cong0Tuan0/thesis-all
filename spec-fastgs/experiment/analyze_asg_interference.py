import os
import sys
import torch
import imageio
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gaussian_renderer import render_fastgs
from scene import Scene, GaussianModel, SpecularModel
from arguments import ModelParams, PipelineParams, get_combined_args
from argparse import ArgumentParser
import torch.nn.functional as F_nn

def save_heatmap(tensor, path, cmap='jet'):
    import matplotlib.pyplot as plt
    tensor_np = tensor.detach().cpu().numpy()
    if tensor_np.max() > 0:
        tensor_np = tensor_np / tensor_np.max()
    
    plt.imsave(path, tensor_np, cmap=cmap)

def analyze(scene: Scene, gaussians: GaussianModel, pipe, bg, specular_mlp, target_cam_name):
    viewpoint_stack = scene.getTrainCameras().copy() + scene.getTestCameras().copy()
    
    target_cam = None
    for cam in viewpoint_stack:
        if target_cam_name in cam.image_name:
            target_cam = cam
            break
            
    if target_cam is None:
        print(f"Target camera {target_cam_name} not found! Falling back to the first test camera...")
        if len(scene.getTestCameras()) > 0:
            target_cam = scene.getTestCameras()[0]
        else:
            target_cam = scene.getTrainCameras()[0]
        
    print(f"--- Analyzing Camera: {target_cam.image_name} ---")
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis", target_cam.image_name)
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Prepare inputs
    xyz = gaussians.get_xyz
    cam_center = target_cam.camera_center.to("cuda")  
    viewdir = xyz - cam_center
    viewdir = viewdir / (viewdir.norm(dim=1, keepdim=True) + 1e-6)
    normal = gaussians.get_normal_axis(viewdir).to("cuda")
    
    # 2. Base MLP Output
    spec_sparse = specular_mlp.step(
        gaussians.get_asg_features.to("cuda"),
        viewdir,
        normal
    )
    
    # 3. Apply Ref Score Gating (Match train.py)
    sampled_ref_score = None
    if hasattr(target_cam, 'ref_score'):
        xyz_homo = torch.cat([xyz, torch.ones_like(xyz[..., :1])], dim=-1)
        clip_space = xyz_homo @ target_cam.full_proj_transform
        ndc_space = clip_space[..., :2] / (clip_space[..., 3:4] + 1e-6)
        ndc_space[..., 1] *= -1 
        ref_score_tensor = target_cam.ref_score.unsqueeze(0).unsqueeze(0).cuda()
        ndc_grid = ndc_space.unsqueeze(0).unsqueeze(0)
        sampled_ref_score = F_nn.grid_sample(
            ref_score_tensor, ndc_grid, align_corners=False, padding_mode='border'
        ).view(-1)
        
        spec_sparse = spec_sparse * sampled_ref_score.unsqueeze(-1)
        
        # Save Ref Score Mask
        save_heatmap(target_cam.ref_score, os.path.join(out_dir, "0_ref_score_mask.png"), cmap='gray')

    # 4. Render Base Only (SH only)
    sh_pkg = render_fastgs(
        target_cam, gaussians, pipe, bg, 1.0, mlp_color=None
    )
    sh_image = sh_pkg["render"]
    
    # 5. Render Full (SH + ASG)
    full_pkg = render_fastgs(
        target_cam, gaussians, pipe, bg, 1.0, mlp_color=spec_sparse
    )
    full_image = full_pkg["render"]
    
    # ASG Only
    asg_image = full_image - sh_image
    
    gt = target_cam.original_image.cuda()[0:3, :, :]
    
    # 6. Calculate Errors
    error_full = (full_image - gt).abs().mean(dim=0)
    error_base = (sh_image - gt).abs().mean(dim=0)
    
    # Error Diff: Positive (Red) means ASG made it WORSE. Negative (Green) means ASG made it BETTER.
    error_diff = error_full - error_base 
    
    # 7. Save outputs
    import torchvision
    torchvision.utils.save_image(gt, os.path.join(out_dir, "1_gt.png"))
    torchvision.utils.save_image(sh_image.clamp(0.0, 1.0), os.path.join(out_dir, "2_base_only.png"))
    torchvision.utils.save_image(full_image.clamp(0.0, 1.0), os.path.join(out_dir, "3_full_render.png"))
    torchvision.utils.save_image(asg_image.clamp(0.0, 1.0), os.path.join(out_dir, "4_asg_only.png"))
    
    save_heatmap(error_base, os.path.join(out_dir, "5_error_base.png"), cmap='hot')
    save_heatmap(error_full, os.path.join(out_dir, "6_error_full.png"), cmap='hot')
    
    # For diff, use bwr (Blue-White-Red). Red = ASG worse, Blue = ASG better
    import matplotlib.pyplot as plt
    diff_np = error_diff.detach().cpu().numpy()
    vmax = np.abs(diff_np).max()
    plt.imsave(os.path.join(out_dir, "7_error_diff_red_worse_blue_better.png"), diff_np, cmap='bwr', vmin=-vmax, vmax=vmax)

    print(f"Analysis saved to {out_dir}")
    print(f"Mean Error Base: {error_base.mean().item():.5f}")
    print(f"Mean Error Full: {error_full.mean().item():.5f}")

if __name__ == "__main__":
    parser = ArgumentParser(description="Analyze ASG Interference")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--target_cam", default="train_r_0", type=str)
    args = get_combined_args(parser)
    
    dataset = model.extract(args)
    pipe = pipeline.extract(args)
    
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    
    # Load Ref Prior logic
    import imageio
    ref_prior_dir = os.path.join(dataset.source_path, "reflection_prior")
    if os.path.exists(ref_prior_dir):
        print("Loading Reflection Priors...")
        for cam in scene.getTrainCameras() + scene.getTestCameras():
            npath = os.path.join(ref_prior_dir, f"{cam.image_name}_ref_score.png")
            if os.path.exists(npath):
                try:
                    ref_tensor = torch.tensor(imageio.imread(npath) / 255.0, dtype=torch.float32)
                    if ref_tensor.shape[0] != cam.image_height or ref_tensor.shape[1] != cam.image_width:
                        ref_tensor = F_nn.interpolate(
                            ref_tensor.unsqueeze(0).unsqueeze(0), size=(cam.image_height, cam.image_width),
                            mode='bilinear', align_corners=False
                        ).squeeze()
                    cam.ref_score = ref_tensor
                except Exception:
                    pass

    asg_path = os.path.join(dataset.model_path, f"point_cloud/iteration_{scene.loaded_iter}/asg.pt")
    gaussians._features_asg = torch.load(asg_path).cuda()
    
    specular_mlp = SpecularModel(dataset.is_real, dataset.is_indoor)
    specular_mlp.load_weights(dataset.model_path, iteration=scene.loaded_iter)
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    analyze(scene, gaussians, pipe, background, specular_mlp, args.target_cam)
