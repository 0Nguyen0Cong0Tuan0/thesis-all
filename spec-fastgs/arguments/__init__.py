#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.asg_degree = 24
        self.is_real = False
        self.is_indoor = False

        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.separate_sh = True
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.antialiasing = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025 
        self.shfeature_lr = 0.005 
        self.opacity_lr = 0.025 
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.001
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        
        # fastgs parameters
        self.loss_thresh = 0.1
        self.grad_abs_thresh = 0.0012  
        self.highfeature_lr = 0.005
        self.lowfeature_lr = 0.0025
        self.grad_thresh = 0.0002
        self.dense = 0.001
        self.mult = 0.5      # multiplier for the compact box to control the tile number of each splat

        self.random_background = False
        self.optimizer_type = "default"

        self.specular_lr_max_steps = 30000
        # Specular starts MID-densification (like Spec-Gaussian's iter-3000 start):
        # densifying without a view-dependent model recruits geometry to fake
        # highlights (Gaussian bloat + baked-in blur). The scorer is specular-aware
        # from this iter on (see compute_gaussian_score_fastgs).
        self.specular_start_iter = 7000

        # Exclude GT pixels above this luminance quantile from the multi-view
        # consistency vote (specular highlights violate its Lambertian assumption).
        # 1.0 disables the mask.
        # v2.4 (2026-06-12): DISABLED. The v2.3 run showed quantile=0.95 cut the
        # Gaussian count only ~10% (negligible speed/VRAM gain) while starving
        # highlight geometry — specular energyRatio fell 0.674→0.428, blur σ 3.15→4.9.
        # It is a net-negative trade, so the mask is off pending a better targeting.
        self.highlight_mask_quantile = 1.0

        # soft SH shielding around specular activation (cosine LR decay on f_rest).
        # v2.4 (2026-06-12): gentler handoff. The aggressive 1.0→0.1 decay turned
        # down SH high-freq faster than the (under-producing) specular MLP could
        # replace it, dimming highlights (gain a*=1.21, lum bias −7.6). Keep more
        # SH high-freq during the ramp: floor 0.3 during decay, 0.5 afterwards.
        self.sh_decay_steps = 2000
        self.sh_scale_min = 0.3
        self.sh_scale_after = 0.5

        # v2.5/v2.6: specular-targeted supervision (root cause A).
        # The global L1+DSSIM loss drowns the ~5% highlight pixels under the 95%
        # diffuse region, so the MLP underfits and the residual comes out dim+blurry
        # (gain a*>1, σ~4.9). This adds a weighted L1 on a highlight mask.
        # weight=0 disables it (default path unchanged).
        #
        # spec_loss_mode selects HOW the mask is built:
        #   "residual"  (v2.6, default): mask = top-(1-quantile) of |GT - diffuse|,
        #       where diffuse is a no-grad SH-only render. This is the SAME locator the
        #       diagnostic uses (results/analysis_out) — it targets energy NOT explained
        #       by the diffuse base, i.e. true specular, and ignores bright-but-flat
        #       diffuse surfaces.
        #   "luminance" (v2.5, deprecated): mask = top-(1-quantile) of GT luminance.
        #       FALSIFIED: it catches bright DIFFUSE (white counter/walls), drove +31%
        #       Gaussians and REGRESSED every metric (v2.5-R1). Kept for reproducibility.
        # Recommended for residual mode: weight 0.1-0.2, quantile ~0.95 (top 5%).
        # Ablation: results/MLP_LATENT_ABLATION_2026-06-13.md.
        self.spec_loss_weight = 0.0
        self.spec_loss_quantile = 0.97
        self.spec_loss_mode = "residual"

        # v2.7 (root cause B): monocular normal prior (DN-Splatter, arXiv:2403.17822).
        # v2.6-R3 proved the residual-mask loss fixes specular MAGNITUDE (energy+gain)
        # but NOT placement (NCC/σ flat) — the fingerprint of noisy min-axis normals.
        # This supervises the rendered per-Gaussian normal against a precomputed
        # monocular normal map (see tools/gen_normal_priors.py) via a cosine loss, so
        # the geometry is reshaped toward true surface orientation. weight=0 disables it
        # (default path unchanged). normal_prior_dir is a subfolder of the source path
        # holding <image_name>.npy maps (OpenCV camera frame). normal_prior_flip negates
        # the prior if the startup alignment diagnostic reports anti-correlation
        # (convention mismatch). Recommended test value: weight 0.02-0.05.
        self.normal_prior_weight = 0.0
        self.normal_prior_dir = "normals"
        self.normal_prior_flip = False
        # v2.8: iteration at which the normal prior starts. v2.7-R4 applied it only
        # after specular_start_iter (7000), which was too late to reshape converged
        # geometry (NCC moved only +0.008). Apply it EARLY (e.g. 500) so it has
        # authority during densification. -1 = fall back to specular_start_iter.
        self.normal_prior_start_iter = -1

        # v3.0 (CVPR contribution): SPECULAR-AWARE DENSIFICATION. FastGS's multi-view
        # consistency vote is Lambertian-biased — specular highlights are view-
        # inconsistent, so the vote either fakes them with diffuse geometry (Gaussian
        # bloat) or under-resolves them. Instead of the crude luminance exclusion
        # (highlight_mask_quantile), decompose the per-pixel error with a diffuse (SH-
        # only) vs full (SH+ASG) render: e_full=|gt-full|, spec_explained=|gt-diffuse|
        # -|gt-full|. (1) residual-gated vote: geometric densify on high e_full that is
        # NOT specular-explained (keeps bright-diffuse under-fit, drops specular the
        # branch handles); (2) specular-deficit allocation: also clone where e_full high
        # AND specular present-but-under-resolved, weighted by spec_densify_weight, to
        # anchor sharper highlights (fixes placement via ALLOCATION — the angle normal
        # supervision could not). Requires the specular branch active. Default off ->
        # exact FastGS vote. See results/CVPR_SPECULAR_DENSIFICATION_PLAN.md.
        self.spec_densify = False
        self.spec_densify_weight = 0.5          # lambda: weight of the specular-deficit clone vote
        self.spec_densify_explained_frac = 0.5  # a pixel is "specular" if spec removed > this frac of error

        # v2.9 (root cause B, DISK-FREE alternative to the monocular normal prior):
        # self-supervised learnable normal refinement. Adds a tiny BOUNDED MLP inside
        # the ASG renderer that predicts a per-Gaussian normal correction from the
        # existing ASG latent; the (working) multi-view specular loss identifies the
        # true normal (Ref-NeRF / 3DGS-DR principle) — no external normal maps, no disk.
        # Zero-initialised + 0.3*tanh bounded => starts as a no-op, can only gently
        # refine. False = default path unchanged. Pairs with spec_loss (root cause A).
        self.normal_refine = False

        # Free-text label for the experiment (e.g. "r5"). Logged to train_info.json and
        # printed at startup so an output folder SELF-IDENTIFIES which run produced it,
        # independent of how the folder was named/copied (caught a mislabeled v2.8 dir).
        self.run_tag = ""

        # v2.5: opt-in alternative specular architecture (utils/spec_arch.py).
        # JSON dict, e.g. '{"activation":"relu","latent_mode":"lowrank","rank":8}'.
        # Empty string keeps the original SpecularNetwork. Can also be set via the
        # SPEC_ARCH env var. The ablation found low-rank latent best (quality+compact);
        # SIREN/WIRE do NOT help — keep relu.
        self.spec_arch = ""

        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
