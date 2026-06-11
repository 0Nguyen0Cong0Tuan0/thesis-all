# Tier-1 Speed Fixes — Change Log

All edits are in **`spec-fastgs/train.py`**. Goal: cut the `counter` training time
(122m 29s, 619k Gaussians) toward ~45–55 min with **no expected quality change**.

Status: edits applied, `py_compile` clean, gating cadence verified by simulation.
**Not run live** in this shell (no CUDA / rasterizer / `fused_ssim`) — measure on GPU.

---

## Summary of the 3 changes

| # | Fix | Where | Effect |
|---|-----|-------|--------|
| S1 | Delete the redundant 2nd render pass | `train.py` densification-stats block | Removes a full render + `autograd.grad` from ~12k iters |
| S2+S3 | Throttle the specular MLP | `train.py` specular block | Full-set MLP calls −75%; sparse subset runs every iter when cache is valid |
| Guard | Step specular optimizer only when MLP ran | `train.py` optimizer step | No Adam step on `None`/stale grads |

---

## S1 — Remove the redundant 2nd render pass

**Problem.** A second, SH-only `render_fastgs` + `torch.autograd.grad` ran **every iter
from `specular_start_iter` (3000) → `densify_until_iter` (15000)** purely to source
"decoupled" geometry gradients for densification. This roughly **doubled** per-iter cost
in the densification phase (~12k iterations). `task.md` claimed this was removed; it was
still in the run that produced the 122-min result.

**Fix.** Drop the 2nd pass; reuse the main render's screen-space stats for densification
(standard 3DGS/FastGS). The specular residual is small (analysis: mean ~2.5 / std ~12 on
[0,255]), and with the throttled MLP most densify-phase iters already render SH-only, so the
densification signal stays geometry-driven.

**Before:**
```python
# DECOUPLED DENSIFICATION (METHOD 2)
viewspace_point_tensor_final = viewspace_point_tensor
visibility_filter_final = visibility_filter
radii_final = radii

if iteration > opt.specular_start_iter and iteration < opt.densify_until_iter:
    render_pkg_sh = render_fastgs(cam, gaussians, pipe, background, opt.mult, mlp_color=None)  # SH-only
    image_sh = render_pkg_sh["render"]
    viewspace_point_tensor_sh = render_pkg_sh["viewspace_points"]
    visibility_filter_sh = render_pkg_sh["visibility_filter"]
    radii_sh = render_pkg_sh["radii"]

    Ll1_sh = l1_loss(image_sh, gt)
    ssim_val_sh = fast_ssim(image_sh.unsqueeze(0), gt.unsqueeze(0))
    loss_sh = (1.0 - opt.lambda_dssim) * Ll1_sh + opt.lambda_dssim * (1.0 - ssim_val_sh)

    geom_grad = torch.autograd.grad(
        outputs=loss_sh, inputs=viewspace_point_tensor_sh,
        retain_graph=False, create_graph=False, allow_unused=True)[0]
    if geom_grad is not None:
        viewspace_point_tensor_sh.grad = geom_grad

    viewspace_point_tensor_final = viewspace_point_tensor_sh
    visibility_filter_final = visibility_filter_sh
    radii_final = radii_sh
```

**After:**
```python
# DENSIFICATION STATS  (Tier-1 fix #1: single render pass)
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
```

---

## S2 + S3 — Throttle the specular MLP

**Problem.** The visibility-gated (sparse) MLP only fires when the cached mask length
matches the current Gaussian count. During densification the count changes **every 100
iters**, so the cache is almost always invalid → the MLP falls back to **all ~600k**
Gaussians for the entire densify phase. The optimization silently doesn't fire where it
matters.

**Fix.**
- Cache **valid** (stable count → mostly post-densification): run the MLP on the small
  visible subset **every iter** — cheap and view-correct.
- Cache **invalid** (count just changed, or first sight): the only option is the full set,
  which is expensive, so **throttle to once every K iters** (`K = spec_full_interval = 4`);
  other iters render SH-only.
- **Do not** reuse a cached specular buffer across iters — it is view-dependent and views
  are sampled randomly, so a stale buffer would be the wrong view. On throttled iters we
  skip specular (SH-only render) instead.

**Before:**
```python
spec_sparse = None; vis_indices = None; mlp_color = None

if iteration > opt.specular_start_iter:
    n_gs = gaussians.get_xyz.shape[0]
    asg_feat = gaussians.get_asg_features  # [N, 24]

    cached_mask = vis_cache.get(cam.uid)
    if cached_mask is not None and cached_mask.shape[0] == n_gs:
        vis_indices = cached_mask.nonzero(as_tuple=False).squeeze(1).to("cuda")
    else:
        vis_indices = torch.arange(n_gs, device="cuda")   # full set, EVERY iter

    if vis_indices.numel() > 0:
        spec_sparse = specular_mlp.step(asg_feat[vis_indices], viewdir[vis_indices], normal[vis_indices])
        mlp_color = torch.zeros((n_gs, 3), device="cuda").index_put((vis_indices,), spec_sparse)
```

**After:**
```python
spec_sparse = None; vis_indices = None; mlp_color = None
ran_spec = False                          # did the MLP actually run this iter?

spec_full_interval = 4  # K: full-set MLP cadence while cache is invalid

if iteration > opt.specular_start_iter:
    n_gs = gaussians.get_xyz.shape[0]

    cached_mask = vis_cache.get(cam.uid)
    cache_valid = cached_mask is not None and cached_mask.shape[0] == n_gs

    if cache_valid:
        vis_indices = cached_mask.nonzero(as_tuple=False).squeeze(1).to("cuda")
        do_spec = True                                   # cheap subset → every iter
    else:
        vis_indices = torch.arange(n_gs, device="cuda")  # full set
        do_spec = (iteration % spec_full_interval == 0)  # throttle the costly path

    if do_spec and vis_indices.numel() > 0:
        asg_feat = gaussians.get_asg_features  # [N, 24]
        spec_sparse = specular_mlp.step(asg_feat[vis_indices], viewdir[vis_indices], normal[vis_indices])
        mlp_color = torch.zeros((n_gs, 3), device="cuda").index_put((vis_indices,), spec_sparse)
        ran_spec = True
```

---

## Guard — Step the specular optimizer only when the MLP ran

**Problem.** `SpecularModel.optimizer_step()` calls `step()` then `zero_grad(set_to_none=True)`.
On throttled iters the specular graph isn't built, so its grads are `None`; stepping is a
no-op at best, noise at worst.

**Fix.** Gate on `ran_spec`.

**Before:**
```python
if iteration > opt.specular_start_iter:
    specular_mlp.update_learning_rate(iteration - opt.specular_start_iter)
    specular_mlp.optimizer_step()
```

**After:**
```python
# Only step when the MLP actually ran this iter (ran_spec): on throttled iters
# the specular graph wasn't built, so its grads are None and a step would be a
# no-op at best. update_learning_rate must precede optimizer_step.
if iteration > opt.specular_start_iter and ran_spec:
    specular_mlp.update_learning_rate(iteration - opt.specular_start_iter)
    specular_mlp.optimizer_step()
```

---

## Expected impact (simulated, 250-cam / 30k-iter counter run)

Of the 27k specular-active iterations:

| Path | Share | Cost |
|------|-------|------|
| sparse (visible subset) | **56 %** | cheap |
| full-set MLP (throttled) | **11 %** | expensive |
| SH-only (specular skipped) | **33 %** | no MLP |

- Full-set MLP calls: **~11,900 → ~3,000 (−75 %)**.
- Plus S1 removes a full render + backward from ~12k densify-phase iters.
- Combined target: **~45–55 min**, no expected quality change (specular still trains every
  iter once the Gaussian count stabilizes post-densification).

---

## Verification

- `python -m py_compile train.py` → OK.
- No stale references to removed 2nd-pass vars (`image_sh`, `loss_sh`, `render_pkg_sh`,
  `geom_grad`).
- `spec_reg` block already handles `spec_sparse is None` (and it is `* 0.0`, a no-op).
- Gating cadence verified by pure-Python simulation (table above).
- **Live training NOT run here** (no CUDA / rasterizer / `fused_ssim` in this shell).

---

## How to run + measure + tune

1. Run the usual `counter` training command (Kaggle/GPU).
2. Check `output/counter/train_info.json` → `training_time_formatted`, `final_gaussians`.
3. Re-run `python analysis/analyze_results.py` → confirm SSIM/PSNR/LPIPS held.
4. **Tuning knob:** `spec_full_interval = 4` (`train.py`, in the specular block).
   - Raise to 6–8 for more speed if quality holds.
   - Drop to 2 if specular looks under-trained.

---

## Not yet done (next: Tier 2)

- **S4** Coarse-to-fine / progressive resolution (Spec-Gaussian's own strategy) — also curbs
  the 619k over-densification.
- **S5** Tighten the FastGS Gaussian budget (target ~250k).
- Optional: expose `spec_full_interval` as a CLI flag (`--spec_full_interval`) in
  `arguments/__init__.py` for sweeping without code edits.
