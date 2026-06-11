# Task: Fix Spec-FastGS Specular Integration

## Implementation
- [x] 1. Create `specular/quaternion_utils.py` (copy from baseline)
- [x] 2. Rewrite `specular/specular_model.py` (ASG-based wrapper matching baseline)
- [x] 3. Update `specular/__init__.py`
- [x] 4. Update `scene/gaussian_model.py` (asg_degree param, feature dim fix)
- [x] 5. Update `specular/normal_utils.py` (remove no_grad, add flip_align_view)
- [x] 6. Update `gaussian_renderer/__init__.py` (viewdir for flip, detach normal)
- [x] 7. Update `train.py` (use new SpecularModel API, LR scheduler)
- [x] 8. Update `render.py` (load correct model)
- [x] 9. Update `run_spec-fastgs.sh` (add --is_real / --is_indoor flags)
- [x] 10. Move `train_specular.py` to `archive/` and mark deprecated
- [x] 11. Add `get_linear_noise_func` to `utils/general_utils.py`
- [x] 12. Clean up `arguments/__init__.py` (remove unused params, fix asg_degree default)

## Verification
- [x] Import chain: `quaternion_utils.py` → `spec_utils.py` → `specular_model.py` ✓
- [x] Dimension consistency: `asg_degree=12` → `specular_feat_dim=12` → `SpecularNetworkReal(input=12)` ✓
- [x] No stale imports in active code (only in archived `train_specular.py`) ✓
- [x] Renderer return dict consistent with updated specular flow ✓
- [x] Train/render both use `SpecularModel(is_real, is_indoor)` matching APIs ✓

---

# Task: Spec-FastGS Pipeline Optimization

Two bottlenecks: (1) training speed 4min→30min, (2) SSIM/LPIPS regression.
Solution numbering follows the Antigravity research report.

## Implemented
- [x] **Sol 2 — Eliminate redundant render passes** (speed): single `render_fastgs`
      call in `train.py`; SH-only and spec-sharp passes removed.
- [x] **Sol 1 — Visibility-gated MLP** (speed): cached `prev_vis_mask` (`radii > 0`)
      restricts the ASG MLP to visible Gaussians; scatter-back via `index_put`;
      full-set fallback for one step after densification changes the count.
- [x] **Sol 6 — Soft cosine SH decay** (quality): re-implemented 2026-06-11 as
      **LR scaling** (`sh_lr_scale_cosine` in `utils/general_utils.py` +
      `GaussianModel.set_sh_lr_scale`). The Phase A rewrite had regressed to a hard
      `grad.mul_(0.01)` freeze, which compounded to ~0 under the post-15k
      32/64-iter gradient accumulation and unintentionally froze `_features_dc`.
      Now only `_features_rest`'s LR is decayed (1.0→0.1 cosine over
      `sh_decay_steps=2000` after `specular_start_iter`, then 0.3); `f_dc` learns
      normally; SH optimizer always steps. Tunables in `OptimizationParams`:
      `sh_decay_steps`, `sh_scale_min`, `sh_scale_after`.
- [x] **Dedicated ASG optimizer** (quality, 2026-06-11): `_features_asg` moved out
      of the main optimizer (which post-15k steps only every 32/64 iters) into its
      own Adam, stepped whenever the specular MLP runs — the ASG latents now learn
      at MLP cadence instead of getting ~470 steps over the whole specular window.
      Wired through `_prune_optimizer` / `cat_tensors_to_optimizer`.
- [x] **Subset viewdir/normal** (speed, 2026-06-11): viewdir + `get_normal_axis`
      (argsort + `build_rotation`) no longer computed over ALL N every iteration —
      moved inside the `do_spec` branch and evaluated only on `vis_indices`
      (`get_normal_axis(..., indices=...)`).
- [x] **Lambertian-conflict fixes** (quality + Gaussian-count, 2026-06-11): FastGS's
      multi-view consistency vote assumes Lambertian pixels; specular highlights show
      permanent cross-view "error" → runaway clone/split at highlights (drove the 4×
      bloat to 619k). Two complementary fixes:
      (a) **highlight-robust vote** — `compute_metric_map` in `utils/fast_utils.py`
      excludes GT pixels above `highlight_mask_quantile=0.95` luminance from the
      vote (1.0 disables);
      (b) **specular-aware scorer + earlier specular** — `specular_start_iter`
      15000→**7000** (mid-densification, like Spec-Gaussian's 3000) and
      `compute_gaussian_score_fastgs(..., specular_mlp=...)` renders scoring views
      WITH the specular residual (full-set no_grad MLP per scoring cam, ~10 evals
      per 100 iters) so explained specular isn't counted as error.
- [x] CPU smoke test for the above: `spec-fastgs/test_phase_a_fixes.py` ✓
      (incl. highlight-mask vote behavior)

## Pending (lower priority)
- [ ] **Sol 5 — Laplacian-pyramid perceptual loss** (quality): was implemented
      earlier but **dropped by the Phase A train.py rewrite** (loss is plain
      L1+SSIM again). Re-add `laplacian_pyramid_loss` to `utils/loss_utils.py`
      + `+ lambda_lap * lap_loss` in `train.py` if high-freq recovery is needed.
- [ ] **Sol 3 — Stale MLP cache (K=4)** (speed): refresh `mlp_color` every K iters.
- [ ] **Sol 7 — LoRA low-rank ASG factorization** (both): basis B + per-Gaussian
      coeffs in place of plain 24-dim `_features_asg`. Candidate thesis contribution.
- [ ] **Sol 4 — Hash-grid appearance field** (both): Instant-NGP/Scaffold-GS style.
      Highest long-term impact; candidate thesis contribution.

## Verification
- [x] `train.py` + `utils/loss_utils.py` compile (`py_compile`) ✓
- [x] `laplacian_pyramid_loss`: 0.0 for identical inputs, positive otherwise,
      finite gradients flow ✓
- [x] `sh_grad_scale_cosine`: 1.0 pre-3000 → 0.1 at 5000 → 0.3 after ✓
- [ ] Live training smoke-test (blocked here: `fused_ssim`/rasterizer not installed
      in this shell) — check EMA loss balance in first ~100 iters; lower
      `lambda_lap` to 0.25 if the Laplacian term dominates.
