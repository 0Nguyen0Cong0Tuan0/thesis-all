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

from utils.general_utils import safe_state
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

    gaussians = GaussianModel(dataset.sh_degree, dataset.asg_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    initial_gaussians = gaussians.get_xyz.shape[0]
    gaussians.training_setup(opt)

    specular_mlp = SpecularModel(dataset.asg_degree, dataset.is_real, dataset.is_indoor)
    specular_mlp.train_setting(opt)

    # ------------------------------------------------------------
    # BG
    # ------------------------------------------------------------

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # ------------------------------------------------------------
    # LOAD REFLECTION PRIORS (If available)
    # ------------------------------------------------------------
    import imageio
    ref_prior_dir = os.path.join(dataset.source_path, "reflection_prior")
    if opt.use_ref_score and os.path.exists(ref_prior_dir):
        print("Loading Reflection Priors...")
        loaded_ref_priors = 0
        missing_ref_priors = 0
        for cam in scene.getTrainCameras():
            npath = os.path.join(ref_prior_dir, f"{cam.image_name}_ref_score.png")
            if os.path.exists(npath):
                try:
                    ref_img = imageio.imread(npath)
                    if len(ref_img.shape) == 3:
                        ref_img = ref_img[..., 0]
                    ref_tensor = torch.tensor(ref_img / 255.0, dtype=torch.float32).cuda()
                    if ref_tensor.shape[0] != cam.image_height or ref_tensor.shape[1] != cam.image_width:
                        ref_tensor = torch.nn.functional.interpolate(
                            ref_tensor.unsqueeze(0).unsqueeze(0),
                            size=(cam.image_height, cam.image_width),
                            mode='bilinear',
                            align_corners=False
                        ).squeeze()
                    cam.ref_score = ref_tensor
                    loaded_ref_priors += 1
                except Exception:
                    pass
            else:
                missing_ref_priors += 1
        print(f"Loaded {loaded_ref_priors} reflection priors; missing {missing_ref_priors}.")
    elif opt.use_ref_score:
        print(f"Reflection prior enabled, but directory not found: {ref_prior_dir}")

    # ------------------------------------------------------------
    # TRAIN LOOP
    # ------------------------------------------------------------

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))

    progress_bar = tqdm(range(1, opt.iterations + 1), desc="Training")
    ema_loss = 0.0

    # Cached boolean visibility mask for sparse ASG evaluation. This follows
    # the original fast path: reuse the previous frame's mask and fall back to
    # full ASG only when Gaussian count changes or an explicit refresh is due.
    prev_vis_mask: torch.Tensor | None = None

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
        # COMPUTE VIEWDIR + NORMAL
        # --------------------------------------------------------

        xyz = gaussians.get_xyz
        cam_center = cam.camera_center

        viewdir = xyz - cam_center
        viewdir = viewdir / (viewdir.norm(dim=1, keepdim=True) + 1e-6)

        normal = gaussians.get_normal_axis(viewdir)

        # --------------------------------------------------------
        # SPECULAR (SG STYLE) — sparse MLP via previous-frame visibility
        # --------------------------------------------------------
        # Only Gaussians visible in the previous frame feed the ASG MLP.
        # Typically 10–30 % of Gaussians are on-screen per frame, so this
        # cuts MLP forward+backward cost by 3–10×.
        # After densification the count changes → we detect size mismatch
        # and fall back to all Gaussians for that one step.

        spec_sparse: torch.Tensor | None = None  # sparse MLP output [M, 3]
        vis_indices: torch.Tensor | None = None   # indices of evaluated Gaussians
        mlp_color: torch.Tensor | None = None     # full-scene buffer  [N, 3]

        if iteration > opt.specular_start_iter:
            n_gs = gaussians.get_xyz.shape[0]
            asg_feat = gaussians.get_asg_features  # [N, asg_degree]
            force_full_asg = (
                opt.full_asg_interval > 0 and
                iteration % opt.full_asg_interval == 0
            )

            # Determine which Gaussians to evaluate
            if not force_full_asg and prev_vis_mask is not None and prev_vis_mask.shape[0] == n_gs:
                vis_indices = prev_vis_mask.nonzero(as_tuple=False).squeeze(1)  # [M]
            else:
                # First specular step, full refresh, or count changed after densification.
                vis_indices = torch.arange(n_gs, device="cuda")

            if vis_indices.numel() > 0:
                spec_sparse = specular_mlp.step(
                    asg_feat[vis_indices],
                    viewdir[vis_indices],
                    normal[vis_indices].detach(),
                )  # [M, 3]



                # Scatter back into a full-scene buffer; index_put preserves grad
                mlp_color = torch.zeros(
                    (n_gs, 3), device="cuda"
                ).index_put((vis_indices,), spec_sparse)

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

        # ── Update cached visibility mask for the NEXT iteration ──────────
        # radii has shape [N]; (radii > 0) gives the boolean visibility mask.
        prev_vis_mask = (radii > 0)  # [N] bool

        # --------------------------------------------------------
        # LOSS
        # --------------------------------------------------------
        # Phase A design:
        #   • photometric_loss  — L1 + SSIM, unchanged from FastGS baseline.
        #     Gradients flow back through the renderer into mlp_color, so the
        #     specular MLP is already supervised by image reconstruction.
        #   • spec_reg          — lightweight L2 penalty on specular MLP outputs
        #     (Gaussian-space). Prevents the MLP from producing unbounded colors
        #     or collapsing to zero, without requiring a second render pass.
        # --------------------------------------------------------

        gt = cam.original_image.cuda()

        Ll1      = l1_loss(image, gt)
        ssim_val = fast_ssim(image.unsqueeze(0), gt.unsqueeze(0))
        photometric_loss = (
            (1.0 - opt.lambda_dssim) * Ll1
            + opt.lambda_dssim * (1.0 - ssim_val)
        )
        
        loss = photometric_loss

        loss.backward()

        # --- NO SH DEGREE RESTRICTION ---
        # Allow SH to train normally everywhere to form a smooth base color.
        # ASG will naturally handle the specular highlights due to gradient boosting.

        # --------------------------------------------------------
        # OPTIMIZER STEP
        # --------------------------------------------------------

        # Set skip_sh=False because we now use fine-grained gradient scaling per-Gaussian
        skip_sh = False
        gaussians.optimizer_step(iteration, skip_sh=skip_sh)

        # Update specular lr BEFORE stepping optimizer to ensure non-zero lr is used
        specular_mlp.update_learning_rate(iteration)
        specular_mlp.optimizer_step()

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

            gaussians.max_radii2D[visibility_filter] = torch.max(
                gaussians.max_radii2D[visibility_filter],
                radii[visibility_filter]
            )

            # --- GUIDED DENSIFICATION ---
            # Trả lại quyền kiểm soát sinh hạt cho cơ chế ADC của FastGS
            viewspace_grad = viewspace_point_tensor.grad.clone()
            
            class DummyTensor:
                def __init__(self, grad):
                    self.grad = grad
            
            dummy_viewspace = DummyTensor(viewspace_grad)

            gaussians.add_densification_stats(
                dummy_viewspace,
                visibility_filter
            )

            if (
                iteration > opt.densify_from_iter and
                iteration % opt.densification_interval == 0
            ):
                size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                camlist = sampling_cameras(scene.getTrainCameras().copy(), opt.num_score_cameras)

                with torch.no_grad():
                    importance_score, pruning_score = compute_gaussian_score_fastgs(
                        camlist, gaussians, pipe, background, opt, DENSIFY=True, iteration=iteration
                    )

                gaussians.densify_and_prune_fastgs(
                    max_screen_size=size_threshold,
                    min_opacity=0.005,
                    extent=scene.cameras_extent,
                    radii=radii,
                    args=opt,
                    importance_score=importance_score,
                    pruning_score=pruning_score
                )

            if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                gaussians.reset_opacity()

        # The multiview consistent pruning of fastgs. We do it every 3k iterations after 15k
        if iteration % 3000 == 0 and iteration > 15_000 and iteration < 30_000:
            camlist = sampling_cameras(scene.getTrainCameras().copy(), opt.num_score_cameras)
            with torch.no_grad():
                _, pruning_score = compute_gaussian_score_fastgs(
                    camlist, gaussians, pipe, background, opt, False, iteration=iteration
                )
            gaussians.final_prune_fastgs(min_opacity=0.1, pruning_score=pruning_score)

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
        "asg_degree": dataset.asg_degree,
        "specular_start_iter": opt.specular_start_iter,
        "full_asg_interval": opt.full_asg_interval,
        "num_score_cameras": opt.num_score_cameras,
        "use_ref_score": opt.use_ref_score,
        "ref_prior_method": opt.ref_prior_method,
        "ti_thresh": opt.ti_thresh,
        "ti_bright": opt.ti_bright,
        "sk_intensity": opt.sk_intensity,
        "sk_saturation": opt.sk_saturation,
        "f_rest_warmup_until": opt.f_rest_warmup_until,
        "f_rest_interval_early": opt.f_rest_interval_early,
        "f_rest_interval_mid": opt.f_rest_interval_mid,
        "f_rest_interval_late": opt.f_rest_interval_late,
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
