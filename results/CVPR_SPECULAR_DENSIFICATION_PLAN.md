# CVPR-track plan — Specular-Aware Densification for Fast 3DGS

**Date:** 2026-06-21
**Decision:** primary contribution = *specular-aware densification*; compute = Kaggle + occasional bigger GPU.

## 1. The problem (what we proved the hard way)
- FastGS densifies by a **multi-view photometric-consistency vote** (`compute_gaussian_score_fastgs`,
  `compute_metric_map` in `utils/fast_utils.py`): a per-pixel `|render−gt| > thresh` map, splatted to
  per-Gaussian counts across views.
- Specular highlights are **view-inconsistent by construction**, so the vote mis-allocates Gaussians on
  glossy surfaces: it either recruits diffuse geometry to *fake* highlights (the measured +31% Gaussian
  bloat) or under-resolves real highlights. The current mitigation (`highlight_mask_quantile`, drop bright
  pixels) is a crude band-aid — it also drops bright *diffuse* pixels and gives no positive allocation.
- Root cause A (specular **magnitude**) was fixed by the residual-mask specular loss (v2.6: energyRatio
  0.388→0.461, a*→1.03). Root cause B (specular **placement**, NCC/σ) resisted ALL normal supervision —
  monocular prior (R4, weak+), stronger/earlier prior (R5, OOM), self-supervised refine (R6, null/−0.016).
  Conclusion: placement is not fixable by supervising noisy normals on sparse-specular scenes.

## 2. The contribution
**Specular-aware densification via residual decomposition.** Per scoring view, render twice (no_grad):
`diffuse` (SH-only) and `full` (SH+ASG). Per pixel:
- `e_full = |gt − full|`         (unexplained error → genuine geometric under-fit)
- `spec_explained = |gt − diffuse| − |gt − full|`   (error the ASG branch removed)

Two changes to the vote:
1. **Residual-gated vote (don't fake it):** geometric densification driven by `e_full > t1`, replacing the
   luminance exclusion. Keeps bright-diffuse under-fit in the vote; drops specular the branch explains.
2. **Specular-deficit allocation (do resolve it):** an extra clone signal where `(e_full > t1) AND
   (spec_explained > t2)` — specular present but under-resolved → add Gaussians to anchor sharper ASG lobes.

**Why it's publishable:** novel (no specular-aware densification for 3DGS); it is the genuine
FastGS∩Spec-Gaussian fusion (allocation × appearance), not a bolt-on; it **reframes placement (root cause
B) as an allocation problem** — the angle the normal experiments could not crack; and every claim is
measurable with our existing specular diagnostic (energyRatio / NCC / σ / structural%).

## 3. Non-negotiable foundation
- **Benchmarks = Spec-Gaussian's own suite** (decided 2026-06-21 — use the baseline's exact datasets for
  direct comparability; we can drop our row into their published Tables 1-4). From the paper
  (`papers/3dgs_papers/_work/2402.15870v2_SpecGaussian.txt`):
  1. **Anisotropic Synthetic** — Spec-Gaussian's OWN dataset, purpose-built for specular/anisotropic
     (their Table 1). The specular stage `counter` never was. Released with their code.
  2. **NeRF Synthetic** (Blender 8: chair/drums/ficus/hotdog/materials/mic/ship/lego) — Table 3.
  3. **NSVF Synthetic** — Table 4.
  4. **Mip-NeRF 360** (real, indoor+outdoor) — Table 2. (We already have `counter`.)
  Spec-Gaussian did NOT use Shiny Blender / Ref-NeRF / T&T — dropped. Synthetic scenes are small
  (~800^2, ~100 imgs) → disk-light + fast on Kaggle. Prioritise Anisotropic Synthetic + NeRF Synthetic.
- **Code already supports both:** synthetic = `is_real=False` (SpecularNetwork); real = `--is_real
  --is_indoor` (SpecularNetworkReal). `--spec_densify` is scene-agnostic (operates on render error).
- **Baselines:** run 3DGS / FastGS / Spec-Gaussian / ours on the same GPU, OR cite Spec-Gaussian's
  published numbers directly (datasets match → fair, and cheaper).
- **Headline:** Pareto triangle — PSNR/LPIPS up, training time down, #Gaussians down, with specular-
  diagnostic breakdown showing *why*. On Anisotropic Synthetic the gain should be large (highlights are a
  big fraction of pixels, unlike counter's ~5%).

## 3a. P0 harness implication
Run scripts/notebook are hardcoded for `counter` (`--is_real --is_indoor`,
`./datasets/mipnerf360/counter`). P0 must generalise to a scene list with per-scene config (synthetic:
no `--is_real`; real: `--is_real --is_indoor`) and aggregate into the Table-1..4 layout. Gating item:
get **Anisotropic Synthetic** + **NeRF Synthetic** onto Kaggle (Spec-Gaussian released them).

## 4. Phased plan
- **P0 — harness:** per-scene runner + a results aggregator (table: PSNR/SSIM/LPIPS/time/#GS + diagnostic).
  Pick scene list under the Kaggle+occasional-A100 budget (target ~4 specular + ~3 Mip360).
- **P1 — mechanism:** implement residual-gated vote + specular-deficit allocation behind `--spec_densify`
  (default off). Isolated change in `fast_utils.py` + flag + wiring. CPU-sanity the per-pixel maps.
- **P2 — ablation (the paper's core):** (a) baseline vote, (b) +residual-gate only, (c) +deficit-alloc only,
  (d) both. Show each half's effect on #GS and NCC/σ. This is the ablation table reviewers will want.
- **P3 — scale-out:** run the benchmark sweep (ours vs baselines) on the chosen scenes; bigger GPU for the
  full table.
- **P4 — write-up:** method + diagnostic + ablation + benchmark; position vs Ref-NeRF/3DGS-DR (object-centric)
  as "fast specular for *real, large-scale* scenes via allocation."

## 5. Risks / honesty
- CVPR is a high bar; numbers must be clearly SOTA-shaped on the specular benchmarks. If they're merely
  competitive, target 3DV/BMVC/WACV with the same content.
- Densification changes can destabilize training (no local GPU to test) → implement default-off, log
  #GS-over-time, validate on one scene before the sweep.
- t1/t2 thresholds need a small sweep; keep them tied to existing `loss_thresh` + a quantile to avoid a new
  free hyperparameter per scene.
