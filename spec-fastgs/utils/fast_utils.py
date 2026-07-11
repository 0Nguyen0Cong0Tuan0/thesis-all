import torch
from PIL import ImageFilter
from gaussian_renderer import render_fastgs
from .loss_utils import l1_loss
from fused_ssim import fused_ssim as fast_ssim
import torchvision.transforms as transforms
import random


def sampling_cameras(my_viewpoint_stack, num_cams=10):
    ''' Randomly sample a given number of cameras from the viewpoint stack'''

    camlist = []
    num_cams = min(num_cams, len(my_viewpoint_stack))
    for _ in range(num_cams):
        loc = random.randint(0, len(my_viewpoint_stack) - 1)
        camlist.append(my_viewpoint_stack.pop(loc))
    
    return camlist

def get_loss(reconstructed_image, original_image):
    l1_loss = torch.mean(torch.abs(reconstructed_image - original_image), 0).detach()
    # Epsilon guard: a degenerate uniform-error view would otherwise divide by
    # zero and poison the metric map (and every score downstream) with NaNs.
    l1_loss_norm = (l1_loss - torch.min(l1_loss)) / (torch.max(l1_loss) - torch.min(l1_loss) + 1e-8)

    return l1_loss_norm

def compute_photometric_loss(viewpoint_cam, image, lambda_dssim=0.2):
    gt_image = viewpoint_cam.original_image.cuda()
    Ll1 = l1_loss(image, gt_image)
    loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0)))
    return loss

def normalize(config_value, value_tensor):
    multiplier = config_value
    value_tensor[value_tensor.isnan()] = 0

    valid_indices = (value_tensor > 0)
    valid_value = value_tensor[valid_indices].to(torch.float32)

    ret_value = torch.zeros_like(value_tensor, dtype=torch.float32)
    ret_value[valid_indices] = multiplier * (valid_value / torch.median(valid_value))

    return ret_value

def compute_gaussian_score_fastgs(camlist, gaussians, pipe, bg, args, DENSIFY, iteration=None, specular_mlp=None):
    """Compute multi-view consistency scores for Gaussians to guide densification.

    For each camera in `camlist` the function renders the scene and computes a
    photometric loss and a binary metric map of high-error pixels. It accumulates
    per-Gaussian counts of views that flagged the Gaussian and a weighted
    photometric score across views.

    Args:
        camlist (list): list of viewpoint camera objects to render from.
        gaussians: current Gaussian representation (model/state) used for rendering.
        pipe: rendering pipeline/context required by `render`.
        bg: background used for rendering.
        args: runtime config containing thresholds (e.g. `loss_thresh`).
        DENSIFY (bool): whether to compute and return the importance score
            used for densification. If False, only the pruning score is computed.
        specular_mlp: optional SpecularModel. When provided, the scoring renders
            include the specular (ASG) contribution so genuinely view-dependent
            Gaussians are not misread as multi-view inconsistent (the SH-only
            vote is Lambertian-biased against specular surfaces). None keeps the
            original SH-only behavior. Callers must wrap this function in
            torch.no_grad() (both existing call sites already do).

    Returns:
        importance_score (Tensor): per-Gaussian integer counts of how many views
            marked the Gaussian as high-error (floor-averaged across views).
            This output is only returned if `DENSIFY` is True.
        pruning_score (Tensor): normalized (0..1) per-Gaussian score used to
            prioritize densification (higher means worse reconstruction consistency).
    """

    full_metric_counts = None
    full_metric_score = None

    for view in range(len(camlist)):
        my_viewpoint_cam = camlist[view]

        mlp_color = None
        if specular_mlp is not None:
            xyz = gaussians.get_xyz
            viewdir = xyz - my_viewpoint_cam.camera_center
            viewdir = viewdir / (viewdir.norm(dim=1, keepdim=True) + 1e-6)
            normal = gaussians.get_normal_axis(viewdir)
            mlp_color = specular_mlp.step(gaussians.get_asg_features, viewdir, normal)

        render_image = render_fastgs(my_viewpoint_cam, gaussians, pipe, bg, args.mult, mlp_color=mlp_color)["render"]
        photometric_loss = compute_photometric_loss(my_viewpoint_cam, render_image, getattr(args, 'lambda_dssim', 0.2))

        gt_image = my_viewpoint_cam.original_image.cuda()
        get_flag = True
        l1_loss_norm = get_loss(render_image, gt_image)
            
        metric_map = (l1_loss_norm > args.loss_thresh).int()

        # Ref-score guides FastGS ADC; it does not spawn Gaussians by itself.
        use_ref_score = False
        ref_score_threshold = getattr(args, 'refscore_threshold_min', 0.5)
        if (getattr(args, 'use_ref_score', False) and hasattr(my_viewpoint_cam, 'ref_score')
                and iteration is not None and not getattr(args, 'disable_ref_score', False)):
            if iteration % args.densification_refscore_interval == 0:
                n_budget = getattr(args, 'max_refscore_gaussians', 0)
                n_current = gaussians.get_xyz.shape[0]
                if n_budget > 0 and n_current < n_budget:
                    ratio = min(max(n_current / n_budget, 0.0), 1.0)
                    decay_power = getattr(args, 'refscore_decay_power', 1.0)
                    min_strength = getattr(args, 'refscore_min_strength', 0.15)
                    strength = max((1.0 - ratio) ** decay_power, min_strength)
                    threshold_min = getattr(args, 'refscore_threshold_min', 0.5)
                    threshold_max = getattr(args, 'refscore_threshold_max', 0.9)
                    ref_score_threshold = threshold_min + (1.0 - strength) * (threshold_max - threshold_min)
                    use_ref_score = True

        if use_ref_score:
            ref_mask = (my_viewpoint_cam.ref_score.cuda() > ref_score_threshold).int()
            metric_map = torch.max(metric_map, ref_mask)

        render_pkg = render_fastgs(my_viewpoint_cam, gaussians, pipe, bg, args.mult, mlp_color=mlp_color, get_flag = get_flag, metric_map = metric_map)

        accum_loss_counts = render_pkg["accum_metric_counts"]

        if DENSIFY:
            if full_metric_counts is None:
                full_metric_counts = accum_loss_counts.clone()
            else:
                full_metric_counts += accum_loss_counts

        if full_metric_score is None:
            full_metric_score = photometric_loss * accum_loss_counts.clone()
        else:
            full_metric_score += photometric_loss * accum_loss_counts

    pruning_score = (full_metric_score - torch.min(full_metric_score)) / (torch.max(full_metric_score) - torch.min(full_metric_score) + 1e-8)
    
    if DENSIFY:
        importance_score = torch.div(full_metric_counts, len(camlist), rounding_mode='floor')
    else:
        importance_score = None
    return importance_score, pruning_score
