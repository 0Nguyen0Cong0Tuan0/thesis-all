# Spec-FastGS — Code Review vs. Results (2026-06-11)

Companion to `ANALYSIS_AND_OPTIMIZATION.md`. That doc analyzed the **v2.2 run**
(122 m 29 s, 619 k Gaussians, PSNR 30.72 / SSIM 0.925 / LPIPS 0.166 on `counter`).
This review checks the **current** `spec-fastgs/` source against it: what is already
fixed, what regressed, and what to change before the next training run.

---

## 1. The results in `results/` do NOT reflect the current code

The current `train.py` ("Phase A") already contains the Tier-1 speed fixes that the
122-min run lacked:

| Fix | Status in current code |
|---|---|
| S1 single render pass (2nd "decoupled densification" pass deleted) | ✅ `train.py:170-177, 223-234` |
| S2 per-camera visibility cache for the sparse MLP | ✅ `train.py:79, 136-164, 188` |
| S3 throttle full-set MLP to every K=4 iters when cache invalid | ✅ `train.py:134, 150` |
| Specular deferred to post-densification | ✅ `specular_start_iter = 15000` default (`arguments/__init__.py:110`) |
| MLP capacity bump (`SpecularNetworkReal` → featureC=128, 4×8 lobes) | ✅ `spec_utils.py:163-185` |
| Sigmoid removed + residual skip in `ASGRender` | ✅ `spec_utils.py:120-122` |

**→ First action: retrain on the same scene/GPU and re-measure before any further
surgery.** Expect roughly 122 → ~50-65 min on the same Kaggle GPU just from what is
already merged. The 619 k Gaussian count may also drop by itself: the old 2nd-pass
geometry gradients (a bloat driver named in the analysis) are gone.

Note when comparing runtimes to FastGS paper numbers (~minutes): those are on
RTX-3090-class GPUs; Kaggle P100/T4 is ~3× slower. Benchmark the local `FastGS/`
baseline on the *same* Kaggle GPU for a fair speed-overhead claim in the thesis.

---

## 2. Regressions: task.md claims two fixes that are NOT in the code

`task.md` marks Sol 5 and Sol 6 as implemented; the Phase A rewrite of `train.py`
dropped both:

1. **Laplacian-pyramid loss (Sol 5) — gone.** `utils/loss_utils.py` contains only
   L1/L2/SSIM; `train.py:204-218` uses plain L1+SSIM. No `laplacian_pyramid_loss`
   anywhere in the tree.
2. **Soft cosine SH decay (Sol 6) — gone, hard freeze is back.** `train.py:241-258`
   re-applies the hard ×0.01 grad scale + `skip_sh` for `spec_start..spec_start+2000`.
   No `sh_grad_scale_cosine` in the tree.

Update `task.md` (and CLAUDE.md, which still says "specular activates after iter
3000") to match reality, or re-add the fixes.

---

## 3. New bugs/interactions introduced by current code (fix before retraining)

> **STATUS: 3.1–3.3 and the dead-code/param items in 3.4 were applied on
> 2026-06-11** (same session as this review). Verified by
> `spec-fastgs/test_phase_a_fixes.py` (CPU). The `vis_cache` CPU round-trip was
> deliberately left as-is (GPU index storage would cost ~300 MB VRAM at 619 k).
>
> **ADDENDUM (same day): Lambertian-conflict fixes.** Verified that FastGS's
> multi-view consistency vote is biased against specular content (the scorer
> rendered SH-only at `fast_utils.py` while highlights are inherently
> view-dependent → permanent cross-view "error" → runaway densification at
> highlights; primary suspect for the 4× bloat and the baked-in specular blur).
> Applied: (a) highlight-luminance mask on the vote
> (`highlight_mask_quantile=0.95`, `compute_metric_map`); (b) specular-aware
> scoring renders + `specular_start_iter` 15000→7000 so densification sees
> specular-explained renders, as Spec-Gaussian does (its spec starts at 3000,
> densify ends 15000). This partially supersedes the "leave the scorer alone for
> fidelity" advice in §4.3 — the change is now a deliberate, documented method
> contribution.

### 3.1 SH-freeze × gradient-accumulation compounding (quality bug)
With `specular_start_iter = 15000`, the freeze window is 15000–17000 — exactly where
`optimizer_step` (`gaussian_model.py:250-271`) switches to **stepping every 32 iters
while accumulating grads** (zero_grad only on step). But `train.py:245-252` calls
`grad.mul_(0.01)` **every iteration on the accumulated buffer**, so a gradient
contributed j iters before the step is scaled by 0.01^j ≈ 0. Effects:
- `_features_rest`: doubly dead (also `skip_sh` blocks its optimizer) — intended-ish.
- `_features_dc`: its optimizer DOES step every 32 iters, but its grads are crushed to
  ~0 → **the diffuse base color is silently frozen for 2000 iters**, which was never
  the design (CLAUDE.md: shield *SH high-frequency* only).

**Fix (few lines):** scale the **learning rate** of the `f_rest` param group for the
window instead of mutating grads (restores Sol 6's intent), and leave `f_dc` alone.

### 3.2 ASG features learn 32-64× slower than the MLP (quality limiter)
`_features_asg` sits in the main Gaussian optimizer (`gaussian_model.py:225`), which
post-15000 steps only every 32 iters (and every 64 after 20000). The specular MLP
optimizer steps **every iteration it runs** (`train.py:265-267`). Since specular now
*only* trains in 15000–30000, the per-Gaussian ASG latents — the actual high-frequency
capacity of the specular branch — get ~470 Adam steps total. This directly feeds the
measured failure mode (blurred, under-produced highlights: energyRatio 0.674, σ≈3.15).

**Fix:** give `_features_asg` its own Adam (like `shoptimizer`) and step it whenever
`ran_spec`, leaving the FastGS throttle for the geometry/SH params.

### 3.3 All-N viewdir/normal computed every iteration (pure waste)
`train.py:104-110` computes `viewdir` and `gaussians.get_normal_axis(viewdir)`
(argsort + `build_rotation` over **all N**) every iteration — including all 15 k
pre-specular iters and throttled iters where the MLP never runs. Move it inside the
`do_spec` branch and compute **only on `vis_indices`** (index `get_scaling` /
`get_rotation` before `build_rotation`). Small per-iter cost (~1-3 ms at 600 k) but
it is free to remove and also shrinks the autograd graph.

### 3.4 Minor
- `train.py:213-216`: `spec_reg = spec_sparse.pow(2).mean() * 0.0` — dead graph node;
  delete or give it a real weight.
- `vis_cache[cam.uid] = (radii > 0).cpu()` forces a GPU→CPU copy per iter; storing the
  *nonzero indices* on GPU (~45 MB for 240 cams @ 619 k, 20 % visibility) avoids the
  round-trip. Minor since `loss.item()` already syncs each iter.
- `train.py` hardcodes `spec_freeze_steps = 2000`, `sh_grad_scale = 0.01` — promote to
  `OptimizationParams` for ablation runs.

---

## 4. Remaining big speed levers (after re-measuring)

Per-iteration cost is now ~1 forward render + backward over N Gaussians, so **N and
resolution are the only big knobs left**:

1. **Gaussian budget (S5).** If retrain still lands near 600 k: raise pruning pressure —
   `remove_budget` is `0.5 × candidates` (`gaussian_model.py:539`, inherited from
   FastGS) → try 1.0; or add a hard cap à la Taming-3DGS. Target ~250-300 k. Every
   downstream cost (raster, backward, MLP, Adam, VRAM 9.7 GB) scales with N.
2. **Coarse-to-fine resolution (S4)** — Spec-Gaussian's own schedule; cheap early
   iters *and* documented to suppress Gaussian-count growth.
3. `compute_gaussian_score_fastgs` costs 10 cams × 2 renders every 100 iters
   (≈ +20 % render cost during densification). This is FastGS's core method
   (identical in `FastGS/utils/fast_utils.py`), so leave it for fidelity — only
   reduce `num_cams` if the thesis ablates it.

## 5. Quality levers (unchanged conclusions from the analysis doc)

The measured gap is structural specular blur, not brightness. Ranked:
1. Fix 3.1 + 3.2 above (they throttle exactly the branch that should learn highlights).
2. Re-add Laplacian-pyramid loss (Sol 5) — targets the high-frequency band where the
   render loses 35 %.
3. SIREN activation in `ASGRender`, or the thesis-grade **deferred screen-space
   shading (S6)** which removes per-Gaussian MLP cost entirely (see
   `ANALYSIS_AND_OPTIMIZATION.md` §3).

---

## 6. Recommended sequence

1. Apply §3 fixes (≈30 lines: LR-based SH decay, dedicated ASG optimizer, gated
   subset normal computation, dead-code cleanup).
2. **Retrain `counter` once** → re-measure time / N / PSNR-SSIM-LPIPS; re-run
   `analysis/analyze_results.py` (energyRatio, NCC, blur σ) to isolate the effect.
3. If N is still ≥ 500 k → S5 budget; if time still ≫ FastGS baseline on the same
   GPU → S4 coarse-to-fine.
4. Then the thesis contribution: S6 deferred shading (or SIREN as the cheaper variant).
