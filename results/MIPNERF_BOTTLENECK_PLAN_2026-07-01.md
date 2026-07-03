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

## 6. CNN / classical specular-region detection — survey + verdict (2026-07-01)

**Question asked:** could a CNN (or an existing filter/kernel) detect specular regions in
the raw input images, as an alternative/complement to RS?

**Established research field, with released pretrained models (all real, checked):**
- Fu et al., *A Multi-Task Network for Joint Specular Highlight Detection and Removal*,
  CVPR 2021 — introduced the **SHIQ** dataset (16k real quadruples: input/highlight-
  mask/albedo/removed) and a U-Net-style multi-task CNN.
  (github.com/fu123456/SHIQ)
- Fu et al., *SHDNet* — multi-scale contrast features, BCE + IoU-edge loss, trained on the
  **WHU-Specular** dataset. (github.com/fu123456/SHDNet)
- *SpecSeg* — a small U-Net (5 encoder/4 decoder blocks), trains in **40 min on a single
  P100** (directly Kaggle-feasible), pretrained weights released.
  (github.com/Atif-Anwer/SpecSeg; S-measure 0.676 / F-measure 0.502 / MAE 0.008 on
  WHU-Specular — notably below SHDNet's 0.793/0.676, i.e. a real accuracy gap.)
- *M2-Net* — pip-installable, ships **two pretrained checkpoints** ("nature", "text_face"),
  one-line inference (`python infer.py --infer_model nature`).
  (github.com/kjzju/specular-removal)
- *TSHRNet*, *SpecularityNet-PSD* — further CNN highlight-removal networks with released
  weights, physics-based 3-stage pipelines.
- *StableDelight* (github.com/Stable-X/StableDelight) — **diffusion-based** (Marigold/
  StableNormal-style foundation model), Apache-2.0, one-line Torch Hub inference, trained
  on Hypersim + Lumos + TSHRNet's highlight-removal sets. Outputs a *delit* image (specular
  removed), not a mask directly — the mask would be `original − delit` residual, same
  pattern as our own diffuse-vs-full decomposition.
- Classical (non-learned) precedent: Shafer's **dichromatic reflection model** (1985) and
  Klinker et al. — separate diffuse/specular via chromaticity (a highlight dilutes local
  saturation toward the illuminant color) — the "kernel" that predates all of the above.
- Notably, `gmichaeljaison/specularity-removal` already does **multi-view homography-based**
  specular detection for planar textured objects — i.e., the same *core idea* as our RS
  (cross-view disagreement = specular), just restricted to a single plane. Our RS (per-pixel
  depth-based reprojection) is a strict generalization of this to arbitrary 3D geometry —
  useful confirmation that the RS *principle* has independent precedent.

**"Do we already have a kernel/filter for this" — precise answer:**
No single conv kernel is a specular detector, but CNN first layers empirically learn
color-contrast / center-surround / Gabor-like filters (Olah et al., Distill 2020,
"An Overview of Early Vision in InceptionV1") that respond to local brightness gradients —
relevant to, but not dedicated to, specular detection; this is incidental, not a deployed
tool. The literal zero-training equivalent is the **dichromatic/HSV threshold**
(bright + desaturated), which we implemented and tested directly (see below).

**Empirical validation (done, not hypothetical) — `tools/classical_specular_mask.py`:**
Ran the classical bright+desaturated threshold on our own GT images (no GPU, no external
model, ~1s):
- **counter (Mip-NeRF, real scene):** flags 1.2–3.5% of pixels, and — visually confirmed —
  correctly lights up TRUE specular highlights: glints inside the steel mixing bowl, the
  reflective jar lid, the sheen on the metal baking tray. Good true positives.
- **counter false positive:** a bright white PLASTIC tube (matte, diffuse) is flagged solid
  red — a shiny-looking but non-specular material.
- **teapot (synthetic, white background):** flags **78.6%** of the image — almost the ENTIRE
  white background gets flagged, because a white background is, by definition, bright and
  desaturated, exactly like a true highlight. The real glossy streaks on the teapot body are
  only faintly/partially traced.

**Verdict — structural limitation, not a tuning problem.** Any *appearance-only* detector
(this classical filter, OR a curated-dataset CNN) cannot distinguish a true specular highlight
from a bright, desaturated DIFFUSE surface (white plastic, white background, light countertop,
stainless-steel matte panels) from a single image — both satisfy the same low-level statistics.
This is the EXACT same confound already diagnosed twice this session: the v2.5-R1 GT-luminance
mask (which drove +31% Gaussian bloat by catching bright diffuse) and — by the same logic —
the reason a pretrained CNN (trained on curated closeups of glossy objects/documents/faces,
a different domain than full-room Mip-NeRF360 photography) carries the same domain-confound
risk that the Marigold monocular normal prior did (weak-positive at best, per R4/R5 results).

**RS remains structurally superior for the Mip-NeRF fix specifically**, because it tests the
actual PHYSICAL DEFINITION of specularity — view-DEPENDENT color under verified multi-view
geometric consistency — rather than single-image appearance statistics. It cannot be fooled by
a bright white diffuse surface, because that surface's color will NOT change across
geometrically-consistent reprojected views (RS → low), whereas a true highlight's color WILL
change (RS → high). This is precisely the property no appearance-only method (classical or CNN)
can have.

**Recommendation (multi-viewpoint, hedged):**
1. **Do not add a pretrained CNN as the primary real-scene locator** — expected ceiling is
   similar to the normal-prior experiments (weak-positive at best) given the same domain-gap +
   appearance-only-confound profile, for real engineering cost (porting, disk, Kaggle inference
   time).
2. **Do** keep `tools/classical_specular_mask.py` (already built, zero-cost, zero-dependency) as
   a **fast cross-check / ensemble signal**: overlay it against the RS maps once generated —
   pixels where BOTH agree are the highest-confidence true specular locations; overlay
   disagreement is itself diagnostic (e.g. flags where RS needs the depth-consistency gate
   tightened, or where the classical filter is being fooled by diffuse white material).
3. If a learned appearance prior is wanted later, **StableDelight** (diffusion-based,
   foundation-model generalization, Apache-2.0, trivial inference) is the single best candidate
   to trial — analogous reasoning to why Marigold (not a closed-domain CNN) was chosen for the
   normal prior — but budget for the same "weak-positive" ceiling based on our monocular-prior
   track record (R4/R5/R6 all null-to-weak).
4. Proceed with the RS eyeball-gate (Section 3, item 2) as the primary next step; use the
   classical mask as a same-day, zero-cost companion check on the SAME images.

