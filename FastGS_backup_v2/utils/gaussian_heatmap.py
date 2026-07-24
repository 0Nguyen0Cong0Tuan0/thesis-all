import json
import os

import torch
import torch.nn.functional as F
import torchvision
from tqdm import tqdm


def _apply_heat_colormap(values):
    zeros = torch.zeros_like(values)
    ones = torch.ones_like(values)
    red = torch.clamp(2.2 * values - 0.2, 0.0, 1.0)
    green = torch.clamp(2.0 * values - 0.75, 0.0, 1.0)
    blue = torch.clamp(1.5 - 2.0 * values, 0.0, 1.0) * (values > 0).float()
    colored = torch.stack((red, green, blue), dim=0)
    return torch.where(values.unsqueeze(0) > 0, colored, zeros + 0.02 * ones)


def _project_gaussian_centers(view, xyz):
    height = int(view.image_height)
    width = int(view.image_width)
    ones = torch.ones((xyz.shape[0], 1), dtype=xyz.dtype, device=xyz.device)
    xyz_h = torch.cat((xyz, ones), dim=1)
    projected = xyz_h @ view.full_proj_transform
    ndc = projected[:, :3] / projected[:, 3:4].clamp_min(1e-7)

    px = ((ndc[:, 0] + 1.0) * width - 1.0) * 0.5
    py = ((ndc[:, 1] + 1.0) * height - 1.0) * 0.5
    in_frame = (
        (projected[:, 3] > 0)
        & (px >= 0)
        & (px <= width - 1)
        & (py >= 0)
        & (py <= height - 1)
    )
    return px, py, in_frame


def _build_view_heatmap(view, xyz, radii, blur_kernel=9):
    height = int(view.image_height)
    width = int(view.image_width)
    heat = torch.zeros((height, width), dtype=torch.float32, device=xyz.device)

    px, py, in_frame = _project_gaussian_centers(view, xyz)
    visible = (radii > 0) & in_frame
    if visible.any():
        xs = px[visible].round().long().clamp(0, width - 1)
        ys = py[visible].round().long().clamp(0, height - 1)
        weights = torch.log1p(radii[visible].float()).clamp_min(1e-4)
        heat.index_put_((ys, xs), weights, accumulate=True)

    if blur_kernel and blur_kernel > 1:
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        heat = F.avg_pool2d(
            heat[None, None],
            kernel_size=blur_kernel,
            stride=1,
            padding=blur_kernel // 2,
        )[0, 0]

    if heat.max() > 0:
        heat = torch.log1p(heat)
        heat = heat / heat.max().clamp_min(1e-6)
    return heat, int(visible.sum().item())


def save_gaussian_view_heatmaps(
    views,
    gaussians,
    model_path,
    iteration,
    render_func,
    render_args,
    split="test",
    blur_kernel=9,
):
    heatmap_path = os.path.join(model_path, split, f"ours_{iteration}", "heatmap")
    os.makedirs(heatmap_path, exist_ok=True)

    xyz = gaussians.get_xyz.detach()
    stats = {
        "num_total_gaussians": int(xyz.shape[0]),
        "num_views": len(views),
        "blur_kernel": int(blur_kernel),
        "weighting": "log1p(screen_radius)",
        "description": "Per-render visible Gaussian center density heatmaps.",
        "views": [],
    }

    with torch.no_grad():
        for idx, view in enumerate(tqdm(views, desc=f"{split} Gaussian heatmaps")):
            render_pkg = render_func(view, gaussians, *render_args)
            heat, visible_count = _build_view_heatmap(
                view,
                xyz,
                render_pkg["radii"].detach(),
                blur_kernel=blur_kernel,
            )
            filename = f"{idx:05d}.png"
            torchvision.utils.save_image(
                _apply_heat_colormap(heat.cpu()),
                os.path.join(heatmap_path, filename),
            )
            stats["views"].append(
                {
                    "index": idx,
                    "image_name": getattr(view, "image_name", f"{idx:05d}"),
                    "file": filename,
                    "visible_gaussians": visible_count,
                }
            )

    with open(os.path.join(heatmap_path, "heatmap_stats.json"), "w") as f:
        json.dump(stats, f, indent=4)

    return heatmap_path
