# ============================================================
# Training (Spec-Gaussian + FastGS)
# ============================================================

import torch
import numpy as np
import os, time, sys, json
from random import randint
from tqdm import tqdm
import uuid

from fused_ssim import fused_ssim as fast_ssim
from utils.loss_utils import l1_loss
from utils.image_utils import psnr

from gaussian_renderer import render_fastgs
from scene import Scene, GaussianModel, SpecularModel

from utils.general_utils import safe_state, sh_lr_scale_cosine
from utils.fast_utils import compute_gaussian_score_fastgs, sampling_cameras

from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except:
    TENSORBOARD_FOUND = False


# ============================================================
# TRAINING LOOP
# ============================================================

def training(dataset, opt, pipe):

    start_time = time.time()
    tb_writer = prepare_output_and_logger(dataset)

    # ------------------------------------------------------------
    # INIT MODELS
    # ------------------------------------------------------------

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    initial_gaussians = gaussians.get_xyz.shape[0]
    gaussians.training_setup(opt)

    specular_mlp = SpecularModel(dataset.is_real, dataset.is_indoor)
    specular_mlp.train_setting(opt)

    # ------------------------------------------------------------
    # BG
    # ------------------------------------------------------------

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # ------------------------------------------------------------
    # TRAIN LOOP
    # ------------------------------------------------------------

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))

    progress_bar = tqdm(range(1, opt.iterations + 1), desc="Training")
    ema_loss = 0.0

    # ── Phase A: per-camera cached visibility mask for sparse MLP evaluation ──
    # Fix #2: cameras are sampled RANDOMLY each iteration, so a single previous-
    # iteration mask came from an unrelated view — the MLP was being run on the
    # wrong Gaussians, starving specular supervision (measured low NCC vs GT).
    # Instead we cache visibility PER CAMERA (keyed by cam.uid): the mask reused
    # for a given view is from the last time that SAME view was rendered, which
    # is a stable predictor since geometry drifts slowly.
    # After densification the Gaussian count changes → size mismatch is detected
    # and we fall back to evaluating the full set for that single step.
    vis_cache: dict = {}

    for iteration in progress_bar:

        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # --------------------------------------------------------
        # SAMPLE CAMERA
        # --------------------------------------------------------

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))

        idx = randint(0, len(viewpoint_indices) - 1)
        cam = viewpoint_stack.pop(idx)
        viewpoint_indices.pop(idx)

        # --------------------------------------------------------
        # SPECULAR (SG STYLE) — Phase A: sparse MLP via cached visibility
        # --------------------------------------------------------
        # Only Gaussians visible in the PREVIOUS step feed the ASG MLP.
        # Typically 10–30 % of Gaussians are on-screen per frame, so this
        # cuts MLP forward+backward cost by 3–10×.
        # After densification the count changes → we detect size mismatch
        # and fall back to all Gaussians for that one step.

        spec_sparse: torch.Tensor | None = None  # sparse MLP output [M, 3]
        vis_indices: torch.Tensor | None = None   # indices of evaluated Gaussians
        mlp_color: torch.Tensor | None = None     # full-scene buffer  [N, 3]
        ran_spec = False                          # did the MLP actually run this iter?

        # Tier-1 fix #3 (throttle). When the per-camera visibility cache is VALID
        # (stable Gaussian count → mostly the post-densification phase), the MLP runs
        # on the small visible subset, so we evaluate it EVERY iter — cheap and
        # view-correct. When the cache is INVALID (count just changed during
        # densification, or first sight of this camera) the only option is the full
        # set, which is expensive, so we throttle that to once every K iters. We do
        # NOT reuse a cached specular across iters: it is view-dependent, and views
        # are sampled randomly, so a stale buffer would be the wrong view.
        spec_full_interval = 4  # K: full-set MLP cadence while cache is invalid

        if iteration > opt.specular_start_iter:
            n_gs = gaussians.get_xyz.shape[0]

            cached_mask = vis_cache.get(cam.uid)
            cache_valid = cached_mask is not None and cached_mask.shape[0] == n_gs

            if cache_valid:
                # cached_mask lives on CPU (saves VRAM across many cameras);
                # move only the small index list to the GPU.
                vis_indices = cached_mask.nonzero(as_tuple=False).squeeze(1).to("cuda")  # [M]
                do_spec = True                                   # cheap subset → every iter
            else:
                # First time this camera is seen OR count changed after densification
                vis_indices = torch.arange(n_gs, device="cuda")  # full set
                do_spec = (iteration % spec_full_interval == 0)  # throttle the costly path

            if do_spec and vis_indices.numel() > 0:
                # viewdir/normal only for the evaluated subset: get_normal_axis
                # (argsort + build_rotation) over all N every iter was pure waste
                # on the 15k pre-specular iters and on throttled iters.
                xyz_vis = gaussians.get_xyz[vis_indices]                  # [M, 3]
                viewdir = xyz_vis - cam.camera_center
                viewdir = viewdir / (viewdir.norm(dim=1, keepdim=True) + 1e-6)
                normal = gaussians.get_normal_axis(viewdir, indices=vis_indices)

                spec_sparse = specular_mlp.step(
                    gaussians.get_asg_features[vis_indices],
                    viewdir,
                    normal,
                )  # [M, 3]

                # Scatter back into a full-scene buffer; index_put preserves grad
                mlp_color = torch.zeros(
                    (n_gs, 3), device="cuda"
                ).index_put((vis_indices,), spec_sparse)
                ran_spec = True

        # --------------------------------------------------------
        # RENDER  (single pass — Phase A removes redundant sh & spec-sharp passes)
        # --------------------------------------------------------

        render_pkg = render_fastgs(
            cam,
            gaussians,
            pipe,
            background,
            opt.mult,
            mlp_color=mlp_color
        )

        image                 = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter     = render_pkg["visibility_filter"]
        radii                 = render_pkg["radii"]

        # ── Update THIS camera's cached visibility mask ───────────────────
        # radii has shape [N]; (radii > 0) gives the boolean visibility mask.
        # Stored per-camera (on CPU to bound VRAM) so the next time this exact
        # view is sampled the MLP runs on the Gaussians actually visible from it.
        vis_cache[cam.uid] = (radii > 0).cpu()  # [N] bool, CPU

        # --------------------------------------------------------
        # LOSS
        # --------------------------------------------------------
        # Phase A design: photometric L1 + SSIM, unchanged from FastGS baseline.
        # Gradients flow back through the renderer into mlp_color, so the
        # specular MLP is already supervised by image reconstruction.
        # --------------------------------------------------------

        gt = cam.original_image.cuda()

        Ll1      = l1_loss(image, gt)
        ssim_val = fast_ssim(image.unsqueeze(0), gt.unsqueeze(0))

        loss = (
            (1.0 - opt.lambda_dssim) * Ll1
            + opt.lambda_dssim * (1.0 - ssim_val)
        )

        loss.backward()

        # --------------------------------------------------------
        # DENSIFICATION STATS  (Tier-1 fix #1: single render pass)
        # --------------------------------------------------------
        # Previously a SECOND, SH-only render_fastgs + torch.autograd.grad ran every
        # iter from specular_start→densify_until purely to source "decoupled" geometry
        # gradients — roughly doubling per-iter cost in the densification phase. We
        # drop it and reuse the main render's screen-space stats (standard 3DGS/FastGS
        # densification). The specular residual is small (analysis: mean ~2.5 / std ~12
        # on [0,255]), and with the throttled MLP most densify-phase iters already
        # render SH-only, so the densification signal stays geometry-driven.
        viewspace_point_tensor_final = viewspace_point_tensor
        visibility_filter_final = visibility_filter
        radii_final = radii

        # --------------------------------------------------------
        # Reduce SH competition: soft cosine decay of the f_rest LEARNING RATE
        # around specular activation (Sol 6). LR scaling (not grad.mul_) because
        # post-15k the optimizer accumulates grads over 32/64 iters: mutating the
        # accumulated buffer every iter compounded to 0.01^j, and the old block
        # also froze _features_dc (diffuse) unintentionally. f_dc learns normally.
        # --------------------------------------------------------
        gaussians.set_sh_lr_scale(
            sh_lr_scale_cosine(
                iteration,
                opt.specular_start_iter,
                decay_steps=opt.sh_decay_steps,
                scale_min=opt.sh_scale_min,
                scale_after=opt.sh_scale_after,
            )
        )

        # --------------------------------------------------------
        # OPTIMIZER STEP
        # --------------------------------------------------------

        gaussians.optimizer_step(iteration)

        # Update specular lr BEFORE stepping optimizer to ensure non-zero lr is used.
        # Only step when the MLP actually ran this iter (ran_spec): on throttled iters
        # the specular graph wasn't built, so its grads are None and a step would be a
        # no-op at best. update_learning_rate must precede optimizer_step.
        # The ASG latents step at the SAME cadence as the MLP — leaving them on the
        # main optimizer's 32/64-iter throttle starved the specular branch's
        # per-Gaussian capacity (~470 Adam steps over its whole training window).
        if iteration > opt.specular_start_iter and ran_spec:
            specular_mlp.update_learning_rate(iteration - opt.specular_start_iter)
            specular_mlp.optimizer_step()
            gaussians.asg_optimizer_step()

        # --------------------------------------------------------
        # LOG
        # --------------------------------------------------------

        ema_loss = 0.4 * loss.item() + 0.6 * ema_loss

        if iteration % 10 == 0:
            progress_bar.set_postfix({"loss": f"{ema_loss:.6f}"})

        # --------------------------------------------------------
        # DENSIFY (FASTGS)
        # --------------------------------------------------------

        if iteration < opt.densify_until_iter:

            gaussians.max_radii2D[visibility_filter_final] = torch.max(
                gaussians.max_radii2D[visibility_filter_final],
                radii_final[visibility_filter_final]
            )

            gaussians.add_densification_stats(
                viewspace_point_tensor_final,
                visibility_filter_final
            )

            if (
                iteration > opt.densify_from_iter and
                iteration % opt.densification_interval == 0
            ):
                size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                camlist = sampling_cameras(scene.getTrainCameras().copy())

                # Once specular is active, score with specular-aware renders so the
                # MLP's (view-dependent) contribution isn't counted as cross-view
                # error by the consistency vote.
                spec_for_score = specular_mlp if iteration > opt.specular_start_iter else None

                with torch.no_grad():
                    importance_score, pruning_score = compute_gaussian_score_fastgs(
                        camlist, gaussians, pipe, background, opt, DENSIFY=True,
                        specular_mlp=spec_for_score
                    )

                gaussians.densify_and_prune_fastgs(
                    max_screen_size=size_threshold,
                    min_opacity=0.005,
                    extent=scene.cameras_extent,
                    radii=radii_final,
                    args=opt,
                    importance_score=importance_score,
                    pruning_score=pruning_score
                )

        # --------------------------------------------------------
        # SAVE
        # --------------------------------------------------------

        if iteration in [17000, opt.iterations]:
            print(f"[ITER {iteration}] Saving...")
            scene.save(iteration)

            # ✅ QUAN TRỌNG NHẤT
            specular_mlp.save_weights(scene.model_path, iteration)

    # ------------------------------------------------------------
    # SAVE METADATA
    # ------------------------------------------------------------
    end_time = time.time()
    duration = end_time - start_time
    minutes = int(duration // 60)
    seconds = int(duration % 60)

    metadata = {
        "scene": dataset.source_path.split("/")[-1],
        "git_branch": get_git_branch(),
        "image_scale": dataset.images,
        "iterations": opt.iterations,
        "initial_gaussians": initial_gaussians,
        "final_gaussians": gaussians.get_xyz.shape[0],
        "training_time_seconds": round(duration, 2),
        "training_time_formatted": f"{minutes}m {seconds}s",
        "peak_vram_mib": round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2),
        "datetime_completed": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    info_path = os.path.join(dataset.model_path, "train_info.json")
    with open(info_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Training metadata saved to {info_path}")

# ============================================================
# UTILS
# ============================================================

def get_git_branch():
    try:
        import subprocess
        return subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode("utf-8").strip()
    except Exception:
        return "unknown"

def prepare_output_and_logger(args):

    if not args.model_path:
        unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str)
    else:
        # If the output directory already exists and contains files, back it up first
        if os.path.exists(args.model_path) and os.listdir(args.model_path):
            import datetime
            existing_branch = "unknown"
            info_file = os.path.join(args.model_path, "train_info.json")
            if os.path.exists(info_file):
                try:
                    with open(info_file, "r") as f:
                        old_info = json.load(f)
                        existing_branch = old_info.get("git_branch", "unknown")
                except Exception:
                    pass
            
            if existing_branch == "unknown":
                existing_branch = get_git_branch()
                
            # Use last modified time of the directory for the backup timestamp
            mtime = os.path.getmtime(args.model_path)
            timestamp = datetime.datetime.fromtimestamp(mtime).strftime("%Y%m%d_%H%M%S")

            parent_dir = os.path.dirname(args.model_path)
            folder_name = os.path.basename(args.model_path.rstrip('/'))
            backup_dir = os.path.join(parent_dir, "backups", folder_name)
            backup_path = os.path.join(backup_dir, f"{existing_branch}_{timestamp}")
            
            print(f"Output folder already exists and is not empty. Moving old run to: {backup_path}")
            try:
                os.makedirs(backup_dir, exist_ok=True)
                os.rename(args.model_path, backup_path)
            except Exception as e:
                print(f"Warning: Could not rename existing output folder: {e}")

    print("Output folder:", args.model_path)
    os.makedirs(args.model_path, exist_ok=True)

    with open(os.path.join(args.model_path, "cfg_args"), "w") as f:
        f.write(str(Namespace(**vars(args))))

    if TENSORBOARD_FOUND:
        return SummaryWriter(args.model_path)
    return None


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    parser = ArgumentParser()

    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    args = parser.parse_args()

    safe_state(False)

    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args)
    )

    print("Training complete.")

