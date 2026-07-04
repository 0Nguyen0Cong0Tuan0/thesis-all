import os
import torch
import numpy as np
from PIL import Image, ImageFilter
from gaussian_renderer import render_fastgs
from .loss_utils import l1_loss
from fused_ssim import fused_ssim as fast_ssim
import torchvision.transforms as transforms
import random

# v3.1/v3.2: tiny per-process cache for precomputed Tan-Ikeuchi specular priors used by
# compute_gaussian_score_fastgs's "tanikeuchi" locator (module-level so it persists across
# the many calls during training without threading extra state through the caller).
_tanikeuchi_prior_cache: dict = {}


def _load_tanikeuchi_prior_for_densify(tanikeuchi_dir, image_name, H, W):
    key = (tanikeuchi_dir, image_name)
    if key in _tanikeuchi_prior_cache:
        t = _tanikeuchi_prior_cache[key]
        return None if t is None else t.cuda()
    path = os.path.join(tanikeuchi_dir, image_name + ".png")
    if not os.path.exists(path):
        _tanikeuchi_prior_cache[key] = None
        return None
    arr = np.array(Image.open(path)).astype(np.float32) / 255.0
    t = torch.from_numpy(arr)
    if t.shape[0] != H or t.shape[1] != W:
        t = torch.nn.functional.interpolate(
            t.unsqueeze(0).unsqueeze(0), size=(H, W), mode="nearest"
        ).squeeze(0).squeeze(0)
    _tanikeuchi_prior_cache[key] = t.cpu()
    return t.cuda()


def sampling_cameras(my_viewpoint_stack):
    ''' Randomly sample a given number of cameras from the viewpoint stack'''

    num_cams = 10
    camlist = []
    for _ in range(num_cams):
        loc = random.randint(0, len(my_viewpoint_stack) - 1)
        camlist.append(my_viewpoint_stack.pop(loc))
    
    return camlist

def get_loss(reconstructed_image, original_image):
    l1_loss = torch.mean(torch.abs(reconstructed_image - original_image), 0).detach()
    l1_loss_norm = (l1_loss - torch.min(l1_loss)) / (torch.max(l1_loss) - torch.min(l1_loss))

    return l1_loss_norm

def compute_photometric_loss(viewpoint_cam, image):
    gt_image = viewpoint_cam.original_image.cuda()
    Ll1 = l1_loss(image, gt_image)
    loss = (1.0 - 0.2) * Ll1 + 0.2 * (1.0 - fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0)))
    return loss

def compute_metric_map(render_image, gt_image, loss_thresh, highlight_quantile=1.0):
    """Per-pixel high-error vote for the multi-view consistency score.

    `highlight_quantile` < 1.0 excludes the brightest GT pixels (above that
    luminance quantile) from the vote. The consistency test implicitly assumes
    Lambertian surfaces — specular highlights move with the camera, so they show
    permanent cross-view "error" that SH can never fix, and voting on them drives
    runaway clone/split at highlight regions (measured: 4x Gaussian bloat
    concentrated where bright-region RMSE peaks). Masking them keeps the vote
    meaningful on the diffuse majority of the image."""
    l1_loss_norm = get_loss(render_image, gt_image)
    metric_map = l1_loss_norm > loss_thresh

    if highlight_quantile < 1.0:
        lum = gt_image.mean(dim=0)  # [H, W]
        lum_thresh = torch.quantile(lum.flatten(), highlight_quantile)
        metric_map = metric_map & (lum <= lum_thresh)

    return metric_map.int()


def normalize(config_value, value_tensor):
    multiplier = config_value
    value_tensor[value_tensor.isnan()] = 0

    valid_indices = (value_tensor > 0)
    valid_value = value_tensor[valid_indices].to(torch.float32)

    ret_value = torch.zeros_like(value_tensor, dtype=torch.float32)
    ret_value[valid_indices] = multiplier * (valid_value / torch.median(valid_value))

    return ret_value

def compute_gaussian_score_fastgs(camlist, gaussians, pipe, bg, args, DENSIFY = False,
                                  specular_mlp = None, tanikeuchi_dir = None):
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
        specular_mlp: when the specular branch is active, pass the SpecularModel so
            the scoring renders include the specular residual. Otherwise the scorer
            renders SH-only while training renders SH+specular, and everything the
            MLP correctly explains is counted as cross-view "error" — biasing
            densification toward specular regions. Caller is expected to run this
            under torch.no_grad() (train.py does).
        tanikeuchi_dir: absolute path to the precomputed Tan-Ikeuchi specular-prior
            directory (tools/gen_tanikeuchi_priors.py), or None. Only used when
            args.spec_densify_locator == "tanikeuchi" (and spec_densify is active); falls
            back to the model-residual locator for any camera missing a prior file.

    Returns:
        importance_score (Tensor): per-Gaussian integer counts of how many views
            marked the Gaussian as high-error (floor-averaged across views).
            This output is only returned if `DENSIFY` is True.
        pruning_score (Tensor): normalized (0..1) per-Gaussian score used to
            prioritize densification (higher means worse reconstruction consistency).
    """

    full_metric_counts = None
    full_metric_score = None

    # v3.0: specular-aware densification is active only when the flag is set AND the
    # specular branch exists (we need a diffuse-vs-full decomposition).
    spec_densify = getattr(args, "spec_densify", False) and (specular_mlp is not None)
    lam = getattr(args, "spec_densify_weight", 0.5)
    frac_thresh = getattr(args, "spec_densify_explained_frac", 0.5)
    # v3.1/v3.2: which locator decides "is this pixel specular" for the densification vote.
    use_tanikeuchi_locator = (getattr(args, "spec_densify_locator", "model_residual") == "tanikeuchi"
                              and tanikeuchi_dir is not None)
    tanikeuchi_prior_thresh = getattr(args, "tanikeuchi_prior_thresh", 0.3)

    for view in range(len(camlist)):
        my_viewpoint_cam = camlist[view]

        # Specular-aware scoring: evaluate the MLP for THIS camera (full set,
        # no_grad) so scoring renders match what the training loss actually sees.
        mlp_color = None
        if specular_mlp is not None:
            viewdir = gaussians.get_xyz - my_viewpoint_cam.camera_center
            viewdir = viewdir / (viewdir.norm(dim=1, keepdim=True) + 1e-6)
            normal = gaussians.get_normal_axis(viewdir)
            mlp_color = specular_mlp.step(gaussians.get_asg_features, viewdir, normal)

        render_image = render_fastgs(my_viewpoint_cam, gaussians, pipe, bg, args.mult, mlp_color=mlp_color)["render"]
        photometric_loss = compute_photometric_loss(my_viewpoint_cam, render_image)

        gt_image = my_viewpoint_cam.original_image.cuda()
        get_flag = True

        if spec_densify:
            appearance_flag = None
            if use_tanikeuchi_locator:
                # v3.1/v3.2: MODEL-INDEPENDENT locator — precomputed classical prior.
                prior = _load_tanikeuchi_prior_for_densify(
                    tanikeuchi_dir, my_viewpoint_cam.image_name,
                    render_image.shape[1], render_image.shape[2])
                if prior is not None:
                    appearance_flag = prior >= tanikeuchi_prior_thresh

            # Always compute the on-the-fly diffuse-vs-full residual decomposition when
            # spec_densify is active: needed either AS the "model_residual" locator
            # itself, or (when the tanikeuchi locator is in use) as a CORROBORATION
            # check before letting an appearance-only flag exclude a pixel from
            # geometric densification credit — see the comment below. This reintroduces
            # the extra render pass the tanikeuchi locator was originally chosen to
            # avoid, but that is the necessary cost of the correctness fix (code
            # review 2026-07-04, findings #2 and #5).
            diffuse_image = render_fastgs(
                my_viewpoint_cam, gaussians, pipe, bg, args.mult, mlp_color=None
            )["render"]
            e_full = (render_image - gt_image).abs().mean(0)     # [H,W] error left after specular
            e_diff = (diffuse_image - gt_image).abs().mean(0)    # [H,W] error of diffuse base
            spec_explained = torch.relu(e_diff - e_full)         # error the ASG branch removed

            if appearance_flag is not None:
                # CORROBORATION fixes two related risks: (a) early in specular
                # training (right after specular_start_iter), the ASG branch hasn't
                # learned anything yet, so appearance-flagged pixels would otherwise be
                # excluded from geo_map immediately — starving them of geometric
                # densification during exactly the window they need it most; (b) a
                # genuinely diffuse object the classical locator misclassifies as
                # specular (its own documented "small bright convex object"
                # limitation) would otherwise be silently protected from correction
                # forever, since the ASG branch can never actually explain a truly
                # diffuse defect. In both cases spec_explained stays ~0 (the branch
                # isn't/can't reduce error there), so requiring it to be measurably
                # positive before trusting the appearance flag routes the pixel back
                # to ordinary geometric treatment instead of silently excluding it.
                model_corroborates = spec_explained > 1e-4
                spec_pixel = appearance_flag & model_corroborates
            else:
                # default / fallback: pure on-the-fly residual decomposition (used
                # when spec_densify_locator == "model_residual", or when the
                # tanikeuchi prior file is missing for this camera).
                spec_frac = spec_explained / (e_diff + 1e-6)         # scene-independent, in [0,1]
                spec_pixel = spec_frac > frac_thresh                 # genuinely specular pixel

            base_hi = get_loss(render_image, gt_image) > args.loss_thresh  # high UNexplained error
            # (1) geometric vote: high error NOT attributable to specular (keeps bright-diffuse)
            geo_map = (base_hi & (~spec_pixel)).int()
            # (2) specular-deficit vote: specular present but still under-resolved
            spec_map = (base_hi & spec_pixel).int()

            geo_counts = render_fastgs(my_viewpoint_cam, gaussians, pipe, bg, args.mult,
                                       get_flag=get_flag, metric_map=geo_map, mlp_color=mlp_color)["accum_metric_counts"]
            spec_counts = render_fastgs(my_viewpoint_cam, gaussians, pipe, bg, args.mult,
                                        get_flag=get_flag, metric_map=spec_map, mlp_color=mlp_color)["accum_metric_counts"]

            # importance = geometric under-fit + weighted specular-deficit allocation
            accum_loss_counts = geo_counts + lam * spec_counts
            # pruning is driven by GEOMETRIC error only — never prune a Gaussian merely
            # for carrying (correct) view-dependent specular variation.
            score_counts = geo_counts
        else:
            metric_map = compute_metric_map(
                render_image, gt_image, args.loss_thresh,
                highlight_quantile=getattr(args, "highlight_mask_quantile", 1.0)
            )
            render_pkg = render_fastgs(my_viewpoint_cam, gaussians, pipe, bg, args.mult, get_flag = get_flag, metric_map = metric_map, mlp_color=mlp_color)
            accum_loss_counts = render_pkg["accum_metric_counts"]
            score_counts = accum_loss_counts

        if DENSIFY:
            if full_metric_counts is None:
                full_metric_counts = accum_loss_counts.clone()
            else:
                full_metric_counts += accum_loss_counts

        if full_metric_score is None:
            full_metric_score = photometric_loss * score_counts.clone()
        else:
            full_metric_score += photometric_loss * score_counts

    pruning_score = (full_metric_score - torch.min(full_metric_score)) / (torch.max(full_metric_score) - torch.min(full_metric_score))
    
    if DENSIFY:
        importance_score = torch.div(full_metric_counts, len(camlist), rounding_mode='floor')
    else:
        importance_score = None
    return importance_score, pruning_score