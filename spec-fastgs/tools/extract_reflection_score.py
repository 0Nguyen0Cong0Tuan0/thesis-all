#!/usr/bin/env python3
"""
Model-FREE, geometry-gated Reflection Score (RS) — a specular LOCATOR for real scenes.

WHY: on Mip-NeRF (real) scenes spec_densify is neutral because its model-residual locator
(spec_frac) can't separate true specular from diffuse texture/geometry clutter. RS instead
flags specular from the PHYSICS: a Lambertian point looks identical across views (low
multi-view color variance); a specular point changes with view (high variance).

KEY ROBUSTIFICATION vs the original draft (adapted from PGSR, arXiv:2406.06521):
  * exact POINT reprojection using the RENDERED depth (no fronto-parallel [0,0,1] homography
    approximation — that breaks on slanted real geometry), via the CPU-verified helpers in
    utils.graphics_utils (round-trip err 3e-4 px on real cameras);
  * a DEPTH-CONSISTENCY gate: a neighbor view only contributes where the reprojected point
    is actually VISIBLE there (|z - sampled neighbor depth| small) — this rejects occlusion
    / geometry-error false positives, the dominant real-scene confound.
RS = std of colors across geometrically-valid views. Saved as 1-channel PNG (disk-light) to
<source>/ref_priors/<image_name>_ref_score.png.

This is a STANDALONE locator step: it loads an already-trained model (-m), so we can EYEBALL
the RS maps on counter (do they light up the espresso machine / metal, not just edges?)
BEFORE wiring RS into spec_densify. Validate first; integrate second.

Usage:
  python tools/extract_reflection_score.py -s <scene_path> -m <trained_model_dir> [--rs_neighbors 8]
"""
import os
import torch
import numpy as np
import imageio
from argparse import ArgumentParser
from tqdm import tqdm

from scene import Scene, GaussianModel
from gaussian_renderer import render_fastgs
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from utils.general_utils import safe_state
from utils.graphics_utils import (cam_intrinsics, unproject_depth_to_world,
                                   project_world_to_camera)


@torch.no_grad()
def render_depth(cam, gaussians, pipe, bg, mult):
    """Render the view-space z as an image (override_color trick). Returns [H,W]."""
    xyz = gaussians.get_xyz
    wvt = cam.world_view_transform
    view_xyz = xyz @ wvt[:3, :3] + wvt[3, :3]          # row-vec world->camera
    z = view_xyz[:, 2:3].repeat(1, 3).contiguous()      # [N,3], depth as color
    img = render_fastgs(cam, gaussians, pipe, bg, mult, override_color=z)["render"]
    return img[0]                                        # [H,W]


def nearest_neighbors(cams, k):
    """k nearest cameras by center distance (excluding self & near-duplicates)."""
    centers = torch.stack([c.camera_center for c in cams])           # [M,3]
    d = torch.cdist(centers, centers)                                # [M,M]
    nn = {}
    for i in range(len(cams)):
        order = torch.argsort(d[i])
        picks = [j.item() for j in order if j != i and d[i, j] > 1e-4][:k]
        nn[i] = picks
    return nn


@torch.no_grad()
def compute_rs(scene, gaussians, pipe, bg, source_path, mult, k=8, depth_tol=0.01):
    cams = scene.getTrainCameras().copy()
    save_dir = os.path.join(source_path, "ref_priors")
    os.makedirs(save_dir, exist_ok=True)

    print("Caching depths...")
    depths = [render_depth(c, gaussians, pipe, bg, mult) for c in tqdm(cams)]
    nn = nearest_neighbors(cams, k)

    for i, ref in enumerate(tqdm(cams, desc="Reflection Score")):
        H, W = depths[i].shape
        ref_d = depths[i]
        Fx, Fy, Cx, Cy = cam_intrinsics(ref.FoVx, ref.FoVy, W, H)
        pw = unproject_depth_to_world(ref_d, Fx, Fy, Cx, Cy, ref.world_view_transform)  # [HW,3]

        ref_rgb = ref.original_image.cuda()                          # [3,H,W]
        colors = [ref_rgb.reshape(3, -1).transpose(0, 1)]           # [[HW,3]]
        valid = [torch.ones(H * W, dtype=torch.bool, device="cuda")]

        for j in nn[i]:
            nb = cams[j]
            nFx, nFy, nCx, nCy = cam_intrinsics(nb.FoVx, nb.FoVy, W, H)
            uv, z = project_world_to_camera(pw, nFx, nFy, nCx, nCy, nb.world_view_transform)
            gx = 2 * uv[:, 0] / (W - 1) - 1.0
            gy = 2 * uv[:, 1] / (H - 1) - 1.0
            grid = torch.stack([gx, gy], dim=-1).reshape(1, -1, 1, 2)
            in_frame = (gx.abs() <= 1) & (gy.abs() <= 1) & (z > 0)
            # depth-consistency gate: is this point actually visible in the neighbor?
            z_map = torch.nn.functional.grid_sample(
                depths[j].reshape(1, 1, H, W), grid, align_corners=True).reshape(-1)
            geo_ok = in_frame & ((z - z_map).abs() < depth_tol * z.clamp(min=1e-3) + 1e-3)
            nb_rgb = torch.nn.functional.grid_sample(
                nb.original_image.cuda().unsqueeze(0), grid, align_corners=True
            ).reshape(3, -1).transpose(0, 1)                        # [HW,3]
            colors.append(nb_rgb)
            valid.append(geo_ok)

        C = torch.stack(colors)                                     # [V,HW,3]
        Vm = torch.stack(valid).float().unsqueeze(-1)               # [V,HW,1]
        cnt = Vm.sum(0)                                             # [HW,1]
        mean = (C * Vm).sum(0) / cnt.clamp(min=1)                   # [HW,3]
        var = (((C - mean) ** 2) * Vm).sum(0) / cnt.clamp(min=1)   # [HW,3]
        rs = (var.sqrt().sum(-1))                                   # [HW]
        rs[cnt.squeeze(-1) < 2] = 0.0                               # need >=2 views
        rs = rs.reshape(H, W)
        if rs.max() > 0:
            rs = rs / rs.max()
        out = (rs * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
        imageio.imwrite(os.path.join(save_dir, f"{ref.image_name}_ref_score.png"), out)

    print(f"RS maps written to {save_dir}")


if __name__ == "__main__":
    parser = ArgumentParser()
    lp = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--rs_neighbors", type=int, default=8)
    parser.add_argument("--rs_depth_tol", type=float, default=0.01)
    args = get_combined_args(parser)
    safe_state(False)

    dataset = lp.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=-1, shuffle=False)
    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                      dtype=torch.float32, device="cuda")
    compute_rs(scene, gaussians, pp.extract(args), bg, dataset.source_path,
               op.extract(args).mult, k=args.rs_neighbors, depth_tol=args.rs_depth_tol)
    print("Done.")
