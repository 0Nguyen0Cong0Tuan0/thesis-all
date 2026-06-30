# Assessment: teapot A/B (spec_densify) + colleague's Reflection Score (RS)

Date: 2026-07-01

## Part 1 — teapot A/B: spec_densify (v3.0) on a real specular scene

Clean A/B, identical config except `--spec_densify` (verified run_tag base/r7, v3.0,
specular_start 3000, white bg).

| metric | base | r7 (spec_densify) | Δ |
|---|---|---|---|
| PSNR | 34.458 | 34.045 | **−0.41** |
| SSIM | 0.9829 | 0.9821 | −0.0009 |
| LPIPS | 0.0311 | 0.0326 | +0.0015 |
| #Gaussians | 149,096 | 148,716 | −380 (≈equal) |
| spec NCC | 0.935 | 0.925 | −0.010 |
| spec energyRatio | 0.849 | 0.840 | −0.009 |
| spec best_σ | 1.10 | 1.32 | +0.22 (blurrier) |

**Verdict: spec_densify is NULL→slightly NEGATIVE on teapot.** Two readings, both matter:
1. **#Gaussians is essentially identical** → the specular-deficit allocation (Half-2)
   barely fired. The mechanism is not meaningfully reallocating Gaussians. Likely the
   model-residual locator (`spec_frac = (e_diff−e_full)/e_diff > 0.5`) rarely triggers,
   and/or is unreliable.
2. **No headroom:** teapot base is ALREADY near-ceiling (NCC 0.935, energyRatio 0.849 —
   vs counter's ~0.52/0.46). On a clean synthetic scene the specular branch already nails
   it, so allocation has little to fix and the perturbation just adds noise.
→ teapot is a poor scene to test allocation. The right test is a HARDER specular scene
  (low base NCC / high base σ) where there is headroom. But the bigger issue is (1): the
  allocation isn't firing, which points at the **locator**.

## Part 2 — Reflection Score (RS), `v3_new_architecture/extract_reflection_prior.py`

### What it is
A **model-free, multi-view photometric-variance map** that flags specular/reflective
pixels. Pipeline (`calc_ref_score`): train 5k iters for coarse geometry → render per-view
depth (z-as-color hack) → for each ref camera pick ~10 multi-view neighbors → back-project
ref pixels to 3D via depth, homography-warp neighbor RGB **patches** to the ref frame →
per-pixel **std of colors across neighbors** = RS → normalize, save grayscale PNG to
`<source>/ref_priors/<name>_ref_score.png`.

Principle: a **Lambertian** point looks identical from all views → low variance → low RS;
a **specular** point changes with view → high variance → high RS. This is exactly the
physical signature of specularity.

### Provenance (attribution — important)
The machinery is **PGSR** (Planar-based Gaussian Splatting, arXiv:2406.06521):
`patch_offsets`, `patch_warp`, `get_points_from_depth`, `get_points_depth_in_depth_map`,
the homography `H = R − t·nᵀ/d`, and the camera `Fx/Fy/Cx/Cy` intrinsics are PGSR's
multi-view photometric-consistency code. The colleague's contribution is the
**repurposing**: PGSR uses this as a geometric *regularizer* (minimize warped-patch NCC);
here the per-pixel *variance* is read out as a *specular detector*. Any use must cite PGSR
and credit the colleague; RS itself is not our novelty.

### CRITICAL: the code is an incomplete draft (does NOT run as-is)
None of its dependencies exist in `v3_new_architecture`:
- `from utils.graphics_utils import patch_offsets, patch_warp` → **not defined** (ImportError)
- `gaussians.get_points_from_depth(...)`, `gaussians.get_points_depth_in_depth_map(...)` → **not defined**
- `viewpoint_cam.Fx/Fy/Cx/Cy` → Camera only has `FoVx/FoVy`, **no pinhole intrinsics**
So adopting RS = porting PGSR's multi-view machinery + adding intrinsics + a depth-render
path into OUR codebase (~150–250 lines), then validating. Bounded but real.

### Does it help our specular detection? Multi-viewpoint.
- **Signal (PRO):** model-FREE and available before/independent of the ASG branch — fixes
  the chicken-and-egg weakness of our current locators (residual mask & spec_frac both
  need the model to already work). Physically grounded.
- **Signal (CON):** conflates specularity with **geometry/occlusion error**. The impl uses
  a fronto-parallel normal ([0,0,1]) + coarse 5k depth, so slanted surfaces, depth edges,
  occlusion boundaries and thin structures yield high variance that is NOT specular →
  false positives. The `d_mask`/`pixel_noise<2` filters help but don't remove this.
- **Fit (where it matters most):** root cause A (magnitude) is already solved, so RS as the
  loss mask is only a robustness upgrade. The high-value use is **driving spec_densify**:
  spec_densify is currently failing, plausibly because its locator is weak/circular; a
  model-free RS locator is the natural fix and the real test of "is allocation the lever?"
- **Disk:** RS maps are 1-channel PNGs (tens of KB each) → NOT the OOM problem the normal
  `.npy` priors were. Fine on Kaggle.
- **Diagnostic bonus:** a model-free specular mask would also make OUR diagnostic more
  rigorous (currently the mask is model-derived).

### Recommendation (gated, not blind adoption)
RS is worth pursuing as a **model-free specular locator that replaces our model-residual
heuristics** — primarily to drive spec_densify, secondarily as the loss mask + diagnostic
mask. But gate it:
1. **Port + complete** the code (PGSR helpers, pinhole intrinsics on Camera, depth render).
2. **Generate RS maps for teapot and a HARD specular scene, and EYEBALL them** — do they
   light up the actual highlights, or just geometry edges? If they're clean specular masks,
   proceed; if they're dominated by edges, fix (better normals than [0,0,1]) or reconsider.
3. Only then wire RS into spec_densify (replace `spec_frac`) and re-run the A/B on a scene
   WITH headroom (not teapot).
This keeps us from repeating the normal-prior mistake (committing compute to a weak signal).
RS is the most promising idea we've had for the locator problem — but it must earn its place
on a visual + quantitative check first.
