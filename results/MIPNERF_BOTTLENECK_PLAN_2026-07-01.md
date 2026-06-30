# Conquering the Mip-NeRF bottleneck for specular-aware densification

Date: 2026-07-01

## 1. Diagnosis (data-grounded)
spec_densify (v3.0) behaves like this:
- **Shiny Blender / Anisotropic Synthetic: works** (user-confirmed; clean specular).
- **teapot (synthetic, but EASY): null/slightly-neg** — base NCC already 0.935 (no headroom),
  #GS unchanged (allocation barely fires).
- **counter (Mip-NeRF 360, real): NEUTRAL** — clean A/B vs control:
  - global: PSNR 30.356→30.419, SSIM/LPIPS/#GS flat.
  - specular: NCC 0.524→0.532, σ 4.82→4.73, energyRatio 0.457→0.450 — all within run-noise.
  - counter base has PLENTY of headroom (NCC 0.52), yet spec_densify captures none of it.

**Root bottleneck = the LOCATOR.** spec_densify decides "specular" via the model residual
`spec_frac = (e_diff − e_full)/e_diff > 0.5`. On real scenes this is entangled:
1. `e_diff` (SH-only error) is high almost everywhere on real scenes (texture, complex
   geometry), not just at highlights.
2. the ASG branch reduces error in many textured/view-dependent regions that are NOT truly
   specular (pose/exposure residual, anisotropic texture) → `spec_frac` fires broadly/wrongly.
3. specular is ~5% of pixels and entangled → a model-derived residual cannot isolate it.
On synthetic the diffuse base is clean (e_diff ≈ 0 off-highlight) so `spec_frac` is sharp →
that's why it works there and dies on real.

→ Making the SAME locator more selective (e.g. an absolute-energy gate) does NOT help: the
problem is "no benefit / can't find specular," not "too aggressive / harmful."

## 2. The fix: a model-FREE, geometry-gated specular locator (RS)
Replace the model-residual locator with **multi-view photometric variance under verified
geometry** — the colleague's Reflection Score (RS), adapted from PGSR (arXiv:2406.06521),
made robust for real scenes:
- A Lambertian point looks identical across views → low color variance → not specular.
- A specular point changes with view → high color variance → specular.
- **Geometry gate (the real-scene innovation):** only trust the color variance where the
  multi-view DEPTH/reprojection is consistent. This rejects the two real-scene false-positive
  sources: (a) occlusion/depth errors (geometry-inconsistent → excluded), (b) textured-but-
  diffuse regions (view-CONSISTENT in color after a correct warp → low variance → excluded).
- **Use REAL per-Gaussian normals** in the homography warp (`get_normal_axis`), not the
  fronto-parallel `[0,0,1]` the draft uses — essential on slanted real geometry.
Output: a per-image RS map (1-channel PNG, disk-light) flagging true specular pixels,
model-free and available from coarse geometry. Wire RS into BOTH:
  - spec_densify (replace `spec_frac` with RS) — fixes the locator on real scenes;
  - the residual specular loss mask — more reliable supervision on real scenes;
  - (bonus) the diagnostic's specular mask — model-independent ground-truth-ish.

## 3. Validation gates (do NOT commit compute blindly — lesson from the normal priors)
1. Port PGSR geometry helpers + pinhole intrinsics into our codebase; **CPU round-trip test**
   the unproject→reproject math on real teapot/Bike cameras (no GPU needed) — must recover
   the original pixels. (Foundation step; safe + verifiable.)
2. Generate RS maps for **counter** (Mip-NeRF) and EYEBALL: do they light up the espresso
   machine / metal / glossy counter, NOT just edges? Gate the whole effort on this.
3. Only then wire RS into spec_densify + loss, and re-run the counter A/B (it has headroom).

## 4. Attribution
RS machinery = PGSR (arXiv:2406.06521); the variance-as-detector repurposing is the
colleague's. Cite PGSR + credit colleague. Our novelty = the integration: specular-aware
densification driven by a geometry-gated model-free reflection prior, for real scenes.

## 5. Status / next
- Diagnosis complete (this doc). Foundation port = next concrete step (CPU-verifiable).
- counter R7 diag: results/analysis_out/v3_counter_r7; teapot A/B: v3_teapot_base|r7.
