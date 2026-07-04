# CLAUDE.md

## Project: Spec-FastGS (3D Object Reconstruction via 3DGS)

Thesis project that unifies **FastGS** (multi-view-consistency densification, fast
training) with **Spec-Gaussian** (replaces SH high-frequency handling with a 24-dim
per-Gaussian ASG latent + a `SpecularNetwork` MLP producing a specular residual added
to `sh_color`).

Key behaviors (as of 2026-06-11):
- Specular branch activates after iter 7000 (`specular_start_iter` default) —
  mid-densification, mirroring Spec-Gaussian's iter-3000 start, so densification
  doesn't recruit geometry to fake highlights.
- FastGS's multi-view consistency vote is Lambertian-biased against specular; two
  guards are in place: the scorer renders specular-aware once the branch is active
  (`compute_gaussian_score_fastgs(..., specular_mlp=...)`), and GT pixels above the
  `highlight_mask_quantile=0.95` luminance quantile are excluded from the vote
  (`compute_metric_map` in `utils/fast_utils.py`).
- SH shielding is a soft cosine **LR** decay on `f_rest` only (1.0→0.1 over
  `sh_decay_steps=2000` after specular start, then 0.3) via `set_sh_lr_scale`;
  the old hard `grad.mul_(0.01)` freeze is gone.
- `_features_asg` has its own Adam optimizer stepped at MLP cadence (the main
  optimizer only steps every 32/64 iters post-15k).
- DashGaussian was deliberately dropped.
- ASG integration is already wired up and working (see repo `task.md`).
- Current state, known issues and next steps: `results/CODE_REVIEW_2026-06-11.md`.

## Current phase: pipeline optimization

Targeting two bottlenecks:

1. **Speed** — training inflated from ~4 min to ~30 min, caused by:
   - 3 rasterization passes/iter in `train.py`.
   - Running `SpecularNetwork` on ALL N Gaussians (including invisible ones) every iter.
2. **Quality** — SSIM/LPIPS regression caused by:
   - Adaptive top-5% L2 mask with `spec_weight=2.0`.
   - Hard SH gradient freeze.

## Prioritized fixes

1. Eliminate the 2 redundant render passes.
2. Visibility-gated MLP (run only on `radii>0`).
3. Laplacian-pyramid L1 loss + drop `spec_weight` 2.0 → 0.5.
4. Soft cosine SH gradient decay (instead of hard freeze).
5. Stale MLP cache, K=4.
6. LoRA low-rank ASG factorization.
7. Hash-grid appearance field.

**Quick-win batch** = fixes 1+2+3+4 (~50 lines), targeting 30 min → ~8–12 min plus
recovered SSIM/LPIPS. Fixes **6 and 7** are flagged as the strongest novel thesis
contributions.

## Specular MLP architecture options

Options for changing the specular MLP architecture, all drop-in swaps sharing the same
`ASGRender` interface (`forward(pts, viewdirs, features, normal) → [N,3]`) in
`spec-fastgs/utils/spec_utils.py`.

### Fix first (bugs/mismatches vs Specular-Gaussians baseline)
- **Output activation**: spec-fastgs applies `sigmoid` (clamps to [0,1]); baseline uses
  **none** (unbounded). Sigmoid kills bright HDR specular peaks — likely a large chunk
  of the quality gap on specular regions.
- **`SpecularNetworkReal.asg_feature`**: spec-fastgs uses 24-dim; baseline uses **12-dim**
  (spec-fastgs doubles per-Gaussian ASG storage in real mode).

### Ranked options
1. **Option 1 — Remove sigmoid** *(1 line, try first)* — `spec_utils.py:124`,
   `rgb = torch.sigmoid(self.mlp(mlp_input))` → `rgb = self.mlp(mlp_input)`.
2. **Option 2 — Residual connection in `ASGRender`** *(paper 1710.04773)* — replace
   `nn.Sequential` with explicit forward and a skip: `h2 = F.relu(self.fc2(h1)) + h1`.
3. **Option 3 — SIREN activation** *(best for high-frequency specular)* — replace `ReLU`
   with `sin(ω₀·x)` (SineLayer + SIREN init); needs weight re-init / fresh training.
   Specular highlights are high-frequency view-dependent → SIREN is designed for this.
4. **Option 4 — Hyper-Connection MLP** *(paper 2409.19606, ICLR 2025)* — use n=2 (not 4;
   the MLP is only 3 layers). Static variant (SHC) is sufficient: learnable A_r/A_m/B
   matrices mixing n copies of the hidden state.
5. **Option 5 — Deeper + wider MLP** *(more capacity)* — `80→128→128→3` →
   `80→256→256→256→3` with residual connections.

### Recommended experiment order
1. Remove sigmoid alone — see if it closes the gap with no architecture change.
2. Sigmoid removed + residual skip.
3. Sigmoid removed + SIREN — highest expected gain on specular.
4. Sigmoid removed + Hyper-Connections (n=2) — strongest theoretical backing.

**Status (from prior session):** Option 1 + Option 2 (residual skip) done.

**UPDATE 2026-06-13 — SIREN refuted; architecture is not the bottleneck.** A controlled
GPU-free ablation (`results/MLP_LATENT_ABLATION_2026-06-13.md`,
`spec-fastgs/bench_spec_arch.py`, drop-in `utils/spec_arch.py`) found: the baseline ReLU
MLP already fits sharp view-dependent lobes (energyRatio 0.87, a*≈1, NCC 0.81) given
clean inputs; **SIREN HURTS** (energyRatio 0.37 @ω₀=30, collapses when deep; the additive
near-zero specular residual is hostile to a sine prior), WIRE worsens placement,
deeper+wider not worth 7× params. The real levers are **(A) specular-weighted/HDR loss**
and **(B) normal priors** (noisy normals degrade every architecture equally); the one
worthwhile architecture change is **low-rank/LoRA ASG latent** (best NCC + fewer params).
So do NOT chase SIREN/Hyper-Connections next — keep ReLU.

**v2.5 (2026-06-13) implemented, both opt-in / default-off:** (1) specular-weighted L1 on
the brightest GT pixels — `--spec_loss_weight 0.5 --spec_loss_quantile 0.97` (train.py
loss; root cause A); (2) alternative architecture via `--spec_arch '{...}'` or `SPEC_ARCH`
env (wired in `SpecularModel`), e.g. `{"activation":"relu","latent_mode":"lowrank","rank":8}`.
Defaults leave the v2.4 path unchanged. CODE_VERSION bumped to v2.5; new params logged to
train_info.json.

**v2.5 results (2026-06-14, Kaggle counter) — BOTH NEGATIVE.** R1 (spec_loss_weight 0.5,
luminance mask): PSNR 30.41→29.72, SSIM 0.9232→0.918, energyRatio 0.388→0.363, NCC
0.529→0.475, Gaussians 587k→**768k (+31%)**. Root cause of failure: the mask is GT-LUMINANCE
top-3% → catches bright DIFFUSE (counter/walls), not specular → drives bad densification.
R2 (R1 + low-rank rank-8 latent): WORSE still (PSNR 29.59, energyRatio **0.322**, NCC
**0.437** — lowest of all). The synthetic ablation's "low-rank is best" did NOT transfer:
architecture can't compensate for a bad supervision signal + noisy normals. **v2.3 stays
strict Pareto-best.** Report tables (tab:sf_quant/sf_spec) + analysis bullets updated.

**v2.6 (2026-06-14) implemented — re-targeted specular loss (fixes R1's flaw).** New opt
`--spec_loss_mode residual` (default; "luminance" kept for repro): mask = top-(1−quantile)
of **|GT − diffuse|** (no-grad SH-only render), the same locator the diagnostic uses → true
specular, ignores bright-flat diffuse. Run via `run_spec-fastgs_big_r3.sh`
(`--spec_loss_weight 0.15 --spec_loss_quantile 0.95 --spec_loss_mode residual`, arch
unchanged to isolate). CODE_VERSION→v2.6; spec_loss_mode logged. Defaults still no-op
(weight 0).

**v2.6-R3 results (2026-06-15, Kaggle counter) — FIRST POSITIVE; root cause A CONFIRMED.**
energyRatio **0.388→0.461** (+19% rel, first rise since v2.2, beats v2.3's 0.428), gain
a* **1.033** (closest-to-1 of any version; R1/R2 ~1.07 = too dim), Gaussians **672k** (vs
R1/R2's 768k → inflation +31%→+14%), PSNR fully recovered 30.43≈v2.4, SSIM 0.9207, LPIPS
0.170. **But NCC flat (0.529→0.526), structural% 98.9%, σ 4.90→4.78** — the residual-mask
loss fixes MAGNITUDE (energy+brightness), NOT placement/sharpness. That residual is the
fingerprint of **root cause B (noisy min-axis normals)**: supervision sets how-bright,
normals set where. v2.6 empirically SEPARATES the two causes (A solved by supervision, B
open). R3 doesn't strictly Pareto-dominate v2.3 (trades +14% Gaussians/slower/slightly
lower SSIM-NCC for much higher specular energy). **NEXT = root cause B: normal priors
(DN-Splatter) / optimized per-Gaussian normals** to fix NCC/σ — the sole remaining limiter.

**v2.7 (2026-06-15) implemented — monocular normal prior (root cause B, DN-Splatter style).**
Opt-in `--normal_prior_weight` (default 0 → no-op). When active, train.py renders the
per-Gaussian normals (via `render_fastgs(..., override_color=get_normal_axis(all))`, bg=0),
transforms world→camera (`n_world @ Wv[:3,:3]`, OpenCV frame), and applies a cosine loss vs
a precomputed monocular normal map — gradient flows through `get_normal_axis` into
scaling/rotation, reshaping geometry toward true surface orientation to fix the specular
PLACEMENT (NCC/σ) the residual loss left flat. Priors generated offline by
`tools/gen_normal_priors.py` (default Marigold via diffusers, converted to OpenCV frame;
DSINE alt), saved `<source>/normals/<image_name>.npy`. A one-time startup diagnostic prints
mean cos(render,prior) (want >0; `--normal_prior_flip` if negative). Run via
`run_spec-fastgs_big_r4.sh` (R3 config + `--normal_prior_weight 0.05`, isolates the normal
prior). CODE_VERSION→v2.7; new params logged.

**v2.7-R4 results (2026-06-16) — normal prior WEAK-POSITIVE but underpowered.** Clean A/B
(r3 weight 0 vs r4 weight 0.05): NCC 0.524→0.532, σ 4.82→4.80, energyRatio 0.457→0.466,
structural% 98.74→98.54 — all four placement metrics nudged the RIGHT way (≈2× run-to-run
noise) so the mechanism is correct, but far too small to matter, and +26% train time
(93m→117m). Cause: weight 0.05 too low AND gated at iter 7000 (too late — geometry already
converged). **v2.8 (2026-06-16) implemented:** new `normal_prior_start_iter` (−1 = fall back
to specular_start_iter) lets the prior apply EARLY. `run_spec-fastgs_big_r5.sh` = R4 but
weight 0.05→0.15 + `--normal_prior_start_iter 500` (shapes geometry during densification).
CODE_VERSION→v2.8. Prediction: NCC clearly up / σ down if root cause B is fixable this way;
if still flat, the Marigold prior signal itself is the limiter (switch to DSINE). Pending run.

**v2.9 (2026-06-20) implemented — DISK-FREE root cause B (the normal prior OOM'd on Kaggle).**
The monocular-prior route (R4/R5) cost ~1GB of per-image normal maps → "No space left on
device", and was only weak-positive. Replaced with a self-contained fix: opt-in
`--normal_refine` adds a tiny BOUNDED MLP inside `ASGRender` (`utils/spec_utils.py`) that
predicts a per-Gaussian normal correction from the existing 24-d ASG latent —
`normal' = normalize(normal + 0.3*tanh(refine_mlp(pts)))` before `reflect(...)`. The working
v2.6 multi-view specular loss identifies the true normal (Ref-NeRF / 3DGS-DR principle), so
NO external normal maps, NO extra disk, NO densification surgery. Zero-init output → exact
no-op at start (low-risk). Checkpoint is self-describing (`scene/specular_model.py` load
detects `render_module.refine_mlp.*` keys → render/metrics need no change). Run via
`run_spec-fastgs_big_r6.sh` (= R3 residual specular loss + `--normal_refine`, isolates it;
stale-code guard requires the `normal_refine` token). Notebook R6 cell added (no prereq,
disk-free). CODE_VERSION→v2.9. Prediction: NCC up / σ down at zero disk cost. Pending run.

**v2.9-R6 result (2026-06-21) — NULL / slightly NEGATIVE.** Clean A/B vs R3 control: NCC
0.524→0.508 (−0.016, real small decline), σ/energyRatio flat. The disk-free self-supervised
refinement did NOT fix placement. Combined with R4 (weak+) and R5 (OOM), THREE normal
interventions all fail → root cause B (placement) is identifiability-limited on sparse-specular
indoor scenes. Written into the report as the method's characterized boundary.

**STRATEGIC PIVOT → CVPR contribution = SPECULAR-AWARE DENSIFICATION (v3.0, 2026-06-21).**
Stop incrementing on `counter` (diffuse, hides specular wins). Plan:
`results/CVPR_SPECULAR_DENSIFICATION_PLAN.md`. FastGS's multi-view consistency vote is
Lambertian-biased; make it specular-aware in `utils/fast_utils.compute_gaussian_score_fastgs`
via a diffuse(SH-only)-vs-full(SH+ASG) residual decomposition: e_full=|gt-full|,
spec_explained=|gt-diffuse|-|gt-full|, spec_frac=spec_explained/(e_diff+eps). (1) residual-gated
vote: geometric densify on `base_hi & ~(spec_frac>frac)` (keeps bright-diffuse, drops specular
the branch explains — replaces the luminance hack); (2) specular-deficit allocation: also clone
on `base_hi & (spec_frac>frac)`, weighted by `spec_densify_weight` (λ=0.5), to anchor sharper
highlights. Pruning driven by GEOMETRIC error only. This reframes placement (root cause B) as an
ALLOCATION problem — the angle normal supervision couldn't crack. Opt-in `--spec_densify`
(+`--spec_densify_weight`, `--spec_densify_explained_frac`), default off → exact FastGS vote.
Active window: densify iters 7000-15000 (specular branch live). CODE_VERSION→v3.0; logged to
banner+train_info. Run `run_spec-fastgs_big_r7.sh` (= R3 + `--spec_densify`, stale-code guard on
`spec_densify` token); notebook now runs ONLY R7. py_compile clean. NON-NEGOTIABLE NEXT: multi-
scene + specular benchmarks (Shiny Blender / Ref-NeRF real / glossy Mip360) with same-GPU
baselines (3DGS, FastGS, Spec-Gaussian) — counter alone cannot show the win. Pending Kaggle run.

**v3.0 spec_densify RESULTS (2026-07-01) + Mip-NeRF pivot.** Benchmark = Spec-Gaussian's own
suite (Anisotropic Synthetic + NSVF + Mip360); Blender/NSVF loaders added & verified
(`scene/dataset_readers.py`). Results: **works on Shiny Blender/synthetic specular**, but
**NEUTRAL on Mip-NeRF (counter A/B: PSNR 30.36→30.42, NCC 0.524→0.532, σ/energyRatio flat)**
and slightly-neg on teapot (no headroom, base NCC already 0.935). Root bottleneck = the
model-residual LOCATOR (`spec_frac`) can't isolate true specular from diffuse texture/geometry
clutter on real scenes. **FIX (in progress) = model-free, geometry-gated multi-view Reflection
Score (RS)** — colleague's idea (`v3_new_architecture/extract_reflection_prior.py`, PGSR-derived
arXiv:2406.06521, but that draft is non-runnable: missing helpers). Built our own: verified
PGSR geometry helpers in `utils/graphics_utils.py` (cam_intrinsics/unproject/project/patch_*,
CPU round-trip 3e-4px) + `tools/extract_reflection_score.py` (loads trained model → exact point
reprojection + DEPTH-CONSISTENCY gate → RS PNGs in `<source>/ref_priors/`; cleaner than the
fronto-parallel draft). GATE: run extractor on counter, EYEBALL RS maps (real highlights vs
edges?) BEFORE wiring RS into spec_densify + loss. Plans: results/MIPNERF_BOTTLENECK_PLAN_2026-07-01.md,
results/REFLECTION_SCORE_ASSESSMENT_2026-07-01.md.

**CNN/classical specular-detection survey (2026-07-01) — verdict: don't adopt a pretrained
CNN, DO use the classical Shafer detector as a cross-check/prior (with a fix).** Surveyed
released CNN highlight detectors (SHIQ/CVPR21, SHDNet, SpecSeg, M2-Net, StableDelight) —
same domain-gap risk profile as the Marigold normal prior. The classical dichromatic
(Shafer 1985) bright+desaturated threshold correctly flags true highlights but
false-positives hard on bright DIFFUSE surfaces: 78.6% of teapot flagged (its white
background). Tool: `spec-fastgs/tools/classical_specular_mask.py`.

**v3.1 (2026-07-04) — the classical detector FIXED (morphological top-hat) and wired into
training as an opt-in locator.** A linear-high-pass "local contrast" fix removed the
background false-positive but added a NEW artifact (a false ring along object silhouette
edges — any strong edge is also high local-contrast). Fixed with morphological WHITE
TOP-HAT (image − grey_opening(image, disk(radius)), pure scipy, no new dependency,
verified equivalent to skimage): teapot 78.6%→1.97% (no ring), counter 3.48%→1.72% (still
finds true glints). Full-dataset validation (not cherry-picked): counter 240 imgs
mean 1.44%/max 4.22%, teapot 600 imgs mean 1.55%/max 2.68% — no image near the old
78.6% failure. Known residual limitation: a small bright convex object comparable in size
to `tophat_radius` can still false-positive (e.g. a white candle) — rarer/lower-impact
than the background problem, but the prior is a heuristic, not ground truth.

Wired in as TWO independent opt-in locators (both default off, exact old behavior
otherwise): `tools/gen_shafer_priors.py` sweeps a scene offline (pure CPU, no GPU/model,
~0.6s/image) → `--spec_loss_mode shafer` (specular loss mask) and
`--spec_densify_locator shafer` (densification vote in `fast_utils.compute_gaussian_score_fastgs`
— actually CHEAPER than model_residual: skips the online diffuse render). Falls back to
model_residual per-camera if a prior file is missing. `run_spec-fastgs_big_r8.sh` = R7 +
both locators swapped to shafer, isolating the locator source as the single change vs R7
(which was neutral on counter — see the Mip-NeRF pivot entry above). CODE_VERSION→v3.1.
Also fixed a real bug found along the way: `gen_normal_priors.py`/`gen_shafer_priors.py`'s
image globbing didn't dedupe `*.png`/`*.PNG` matches, silently double-processing every
image on case-insensitive filesystems (Windows) — harmless on Kaggle's Linux fs, fixed in
both. Also closed a `.gitignore` gap: the local `dataset/` mirror (containing real
`mipnerf360/*.JPG` photos) was untracked-but-NOT-ignored — only the `Anisotropic-Synthetic-
Dataset`/`Synthetic_NSVF` subfolder NAMES happened to match existing rules; added `/dataset/`
explicitly. Status: implemented + CPU/data-validated, NOT yet GPU-tested — pending Kaggle R8
run. Plan updated: results/MIPNERF_BOTTLENECK_PLAN_2026-07-01.md §7.

**v3.2 (2026-07-04) — classical locator SWAPPED from Shafer to Tan-Ikeuchi + top-hat +
near-saturation recovery (Algorithm 2b).** Built `test_specular_algorithms_comparison.ipynb`
(4 classical algorithm families side by side: Shafer/Klinker, Tan-Ikeuchi, Shen
ColorLines, Kim/Guo DarkChannel). Initial finding: raw Tan-Ikeuchi (Imin thresholded
directly) floods bright/flat surfaces exactly like pre-tophat Shafer did; porting the
identical top-hat gate to Imin helps but a full 240-image production-resolution sweep
still gave mean=7.47%/max=17.96% vs Shafer's 1.44%/4.22% — traced to this notebook's
Tan-Ikeuchi being a simplified single-pass proxy of the paper's core insight, not the
real iterative algorithm. Separately found (via screenshots) that top-hat ALSO produces a
false negative — suppresses genuine highlights spatially WIDER than the structuring disk
(measured: a 105×46px glare blob vs a 24px-diameter disk) exactly like it suppresses flat
backgrounds, since top-hat's rule is purely about size. Two aggressive fixes failed
(multi-scale top-hat exploded flagged% to 35-65%; connected-component area filtering on
the raw candidate mask failed since the missed blob is already merged into the
surrounding wall's connected component). Fix that worked: OR the existing top-hat gate
with `Imin > 0.85` (near the sensor's clipping point) — full-sweep validated: mean
7.47%→7.89% (+0.42pp only), max unchanged at 17.96%. This is "Algorithm 2b" in the
comparison notebook. **Decision (after visual review): swapped production from Shafer to
Tan-Ikeuchi+top-hat+near-saturation (2b)**, despite it flagging ~5.5x more of the image
than Shafer did on the full-sweep numbers — Imin alone lacks Shafer's saturation
condition, which independently helped reject colored-but-bright diffuse pixels. Same
opt-in wiring pattern as v3.1, renamed throughout: `tools/gen_tanikeuchi_priors.py`
(replaces `gen_shafer_priors.py`), `--spec_loss_mode tanikeuchi` /
`--spec_densify_locator tanikeuchi`, `tanikeuchi_prior_dir`/`tanikeuchi_prior_thresh`
args, `run_spec-fastgs_big_r8.sh` updated. CODE_VERSION→v3.2. **RISK FLAGGED**: since this
locator is measurably looser than Shafer, watch the R8 run for a v2.5-R1-style
regression (Gaussian bloat / PSNR-SSIM drop) — the likely fix if it regresses is
raising `tanikeuchi_prior_thresh` above its current 0.3 default. Status: implemented +
CPU/data-validated, NOT yet GPU-tested — pending Kaggle R8 run.

Sources: Residual Connections (arXiv:1710.04773), Hyper-Connections (arXiv:2409.19606),
SpecNeRF (arXiv:2312.13102), 3DGS with Deferred Reflection (arXiv:2404.18454),
SIREN (arXiv:2006.09661), WIRE (arXiv:2301.05187), LoRA (arXiv:2106.09685),
Focal Frequency Loss (arXiv:2012.12821), DN-Splatter (arXiv:2403.17822).

## Reference artifacts

Antigravity research:
- `C:\Users\nguye\.gemini\antigravity\brain\3c91828b-1b1a-4b1e-9717-55469dd3c185\artifacts\pipeline_optimization_research.md`
- `C:\Users\nguye\.gemini\antigravity\brain\3c91828b-1b1a-4b1e-9717-55469dd3c185\artifacts\thesis_and_codebase_analysis.md`
