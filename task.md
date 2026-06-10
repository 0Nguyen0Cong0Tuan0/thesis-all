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
- [x] **Sol 5 — Laplacian-pyramid perceptual loss** (quality): added
      `laplacian_pyramid_loss` (+ pyramid helpers) to `utils/loss_utils.py`;
      wired into `train.py` as `+ lambda_lap * lap_loss` (`lambda_lap = 0.5`,
      tunable). Old aggressive top-5% L2 mask + `spec_weight=2.0` already removed.
- [x] **Sol 6 — Soft cosine SH decay** (quality): `sh_grad_scale_cosine()` replaces
      the hard freeze. Only `_features_rest` is suppressed (1.0→0.1 over iters
      3000–5000, then 0.3); `_features_dc` learns normally; SH optimizer always steps.

## Pending (lower priority)
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
