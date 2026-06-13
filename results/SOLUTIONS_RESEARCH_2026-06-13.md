# Spec-FastGS — Cross-Field Solution Catalogue (deep research, 2026-06-13)

## The problem, stated precisely

The specular branch (ASG latent → `SpecularNetwork` MLP → additive residual on SH color)
**under-produces specular energy**. Measured on the top-5% specular-energy mask, the
diagnostic signature across v2.2 → v2.3 → v2.4 is stable and three-fold:

| Symptom | Measured | Reading |
|---|---|---|
| `gain a* ≈ 1.20` | render highlight ~20% **too dim** | amplitude deficit |
| `best_σ ≈ 4.9` | render is **over-blurred** vs GT | high-frequency deficit |
| `NCC ≈ 0.52`, `structural% ≈ 98` | highlight **misplaced** | direction/normal error |
| `energyRatio ≈ 0.39` | render emits ~39% of GT specular energy | net of the above |

**v2.4 falsified** the "densification suppressors are the cause" hypothesis (relaxing
them changed nothing). Code-level tracing of commit `c2b77a6` isolates **two structural
root causes**:

- **(A) Supervision deficit** — the only loss is global `L1 + DSSIM`. Highlights are
  ~5% of pixels, so their gradient is drowned by the 95% diffuse region. The MLP
  converges to a *safe, dim, smooth* residual → explains `a*>1` and high `σ`.
- **(B) Geometry/direction deficit** — `reflect_dir` is built from a **geometric
  pseudo-normal** (`get_minimum_axis`, unsupervised). For near-isotropic Gaussians the
  shortest axis is noise → wrong ASG lobe direction → explains low `NCC` / high
  `structural%`, and contributes to blur.

A candidate solution is *relevant* only if it attacks (A), (B), or the representational
capacity that bounds both. Below, every entry is tagged with the symptom(s) it targets,
its source field, an arXiv handle, and whether the PDF is already in `papers/`.

Legend: 🅰 = attacks supervision deficit · 🅱 = attacks normal/direction deficit ·
🅲 = raises high-frequency capacity · ⭐ = already in local library · 🧪 = strong thesis-
novelty candidate.

---

## Category 1 — Supervision & loss design (root cause A: dim + blur)

These are the **cheapest, highest-leverage** fixes: they add gradient drive to the
highlight pixels without touching the architecture. This is where the diagnostic signature
points first.

| # | Solution | Field | What it does / why it fits | Cite |
|---|---|---|---|---|
| 1.1 🅰 | **Specular-mask weighted loss** | 3DGS/NeRF | Add `λ·L1` on the top-k specular-energy mask (the exact mask the diagnostic already computes). Directly counters the 5%-pixel drowning → lifts `a*` toward 1. The simplest possible fix and the natural first experiment. | (our diagnostic; same idea as Ref-NeRF's specular split [2112.03907]) |
| 1.2 🅰🅲 | **Focal Frequency Loss (FFL)** | CV / image synth | Loss in the 2D Fourier domain that *adaptively up-weights the frequency components that are hard to synthesize* — precisely the high-band our render loses (`σ`=4.9). Drop-in, complementary to spatial L1. Proven to de-blur autoencoder/GAN outputs. | Jiang et al., ICCV 2021, [2012.12821] |
| 1.3 🅰🅲 | **Progressive frequency regularization (FreGS)** | 3DGS | Matches amplitude+phase of render vs GT spectrum, annealed low→high frequency during training. Built *for 3DGS over-reconstruction blur* — the same blur we see. Pairs with densification. | Zhang et al., CVPR 2024, [2403.06908] |
| 1.4 🅰🅲 | **Laplacian-pyramid / multi-scale detail loss** | CV / super-res | Decompose render & GT into a Laplacian pyramid; weight the high-freq bands. Long-standing recipe for forcing sharp detail (LAPGAN, LapSRN). CLAUDE.md already lists this as fix #3. | Denton LAPGAN [1506.05751]; Lai LapSRN [1704.03915] |
| 1.5 🅰 | **Hard-example mining / focal weighting on pixels** | CV / detection | Treat bright/high-error pixels as the "hard" class: OHEM samples highest-loss regions; focal loss `(1-p)^γ` down-weights easy diffuse pixels. Re-balances the 95/5 imbalance in gradient space. | OHEM Shrivastava CVPR 2016 [1604.03540]; Focal Loss Lin ICCV 2017 [1708.02002] |
| 1.6 🅰 | **Error-/loss-guided pixel importance sampling** | NeRF | Instead of reweighting, *sample* highlight pixels more often (probability ∝ loss/MAE/edge/entropy). Concentrates optimizer steps on highlights; cheap, orthogonal to 1.1. | Sun et al. [2308.15547]; "Not All Samples Equally Hard" [2408.03193]; edge-based [SCITEPRESS 123462] |
| 1.7 🅰 | **HDR tone-mapped / log-domain loss** | comp. photography | Specular peaks are near-HDR; a plain L1 under-weights bright-pixel error after the [0,1] clamp. Apply the loss through a log/tonemap curve so bright highlights keep gradient. Explains and fixes the `a*>1` dimming directly. | RawNeRF/"NeRF in the Dark" Mildenhall CVPR 2022 [2111.13679]; HDR-NeRF Huang CVPR 2022 [2111.14451] |
| 1.8 🅰🅱 | **Image-gradient / edge loss** | CV | Penalize ‖∇render − ∇GT‖ to sharpen highlight boundaries (helps both `σ` and placement crispness). Trivial add-on to 1.1–1.4. | (classic; used widely in depth/normal & SR pipelines) |

---

## Category 2 — High-frequency view-dependent representation (root cause A: capacity)

Even with perfect supervision, a ReLU MLP has a **spectral bias** (it provably struggles
to fit high frequencies). Specular highlights *are* high-frequency in the view/reflection
direction. These swap the MLP's activation/encoding to lift its frequency ceiling.

| # | Solution | Field | What it does / why it fits | Cite |
|---|---|---|---|---|
| 2.1 🅲 | **SIREN (sine activations)** | INR / signal | Replace ReLU with `sin(ω₀·x)`; designed to represent high-frequency signals and their derivatives. Highest-expected single-swap gain on specular; needs principled init + fresh training. (CLAUDE.md Option 3.) | Sitzmann et al., NeurIPS 2020 [2006.09661] ⭐ |
| 2.2 🅲 | **Fourier-feature / positional encoding of view dir** | ML theory / NeRF | Map `viewdirs` through `[sin(2^k π x), cos(...)]` before the MLP to defeat spectral bias (NTK-grounded). We already PE the view dir at `viewpe=2`; *increasing the bandwidth* is a one-line lever. | Tancik et al., NeurIPS 2020 [2006.10739]; (NeRF PE original) |
| 2.3 🅲 | **WIRE (complex Gabor wavelet activation)** | INR / signal | Gabor wavelet = optimal space-frequency localization; beats SIREN/Gaussian on accuracy *and* robustness to noise. A drop-in nonlinearity for `ASGRender`. | Saragadam et al., CVPR 2023 [2301.05187] |
| 2.4 🅱🅲 | **Integrated Directional Encoding (Ref-NeRF)** | NeRF / optics | Encodes a *distribution* of reflection vectors weighted by roughness, and reparameterizes by the reflected view dir — making specular smooth & interpolable. Directly raises angular fidelity (helps `σ` and `NCC`). | Verbin et al., CVPR 2022 (Best Paper) [2112.03907] ⭐ |
| 2.5 🅱🅲 | **Learnable Gaussian directional encoding (SpecNeRF)** | NeRF | A learnable set of directional Gaussians as the encoding basis for near-field, high-frequency specular — purpose-built for our exact failure. | Ma et al., CVPR 2024 [2312.13102] ⭐ |
| 2.6 🅲 | **Spherical-Gaussian / ASG mixture as explicit lobe basis** | graphics / BRDF | Represent the specular lobe as a sum of (anisotropic) spherical Gaussians — compact, closed under products/rotations, supports *arbitrarily high specularity*. Our ASG is already one lobe family; widen/deepen the mixture or fit it more faithfully. | Wang et al. (All-Frequency SVBRDF, SG mixtures); PhySG Zhang CVPR 2021 [2104.00674]; GlossGau (anisotropic SG) [2502.14129] |
| 2.7 🅲🧪 | **KAN / Kolmogorov–Arnold (Fourier) Networks** | ML (2024–25) | Learnable univariate edge functions instead of fixed activations; KAN-Fourier variants capture high-freq with few params. Higher-risk but a fresh, citable architecture angle for the thesis. | Liu et al., ICLR 2025 [2404.19756]; KA-Fourier Nets [2502.06018] |
| 2.8 🅲 | **Hash-grid / anchor appearance field** | graphics | Move high-freq appearance off the per-Gaussian MLP into a multiresolution hash grid (Instant-NGP) or anchor-decoded features (Scaffold-GS). Decouples capacity from Gaussian count. (CLAUDE.md fix #7.) | Müller Instant-NGP, SIGGRAPH 2022 [2201.05989] ⭐; Scaffold-GS CVPR 2024 [2312.00109] ⭐ |

---

## Category 3 — Normal / reflection-direction quality (root cause B: misplacement)

These attack the `NCC ≈ 0.52`, `structural% ≈ 98` signature — the highlight is in the
*wrong place* because the pseudo-normal is noise.

| # | Solution | Field | What it does / why it fits | Cite |
|---|---|---|---|---|
| 3.1 🅱 | **Normal regularizer + reflection reparam (Ref-NeRF)** | NeRF | Penalize predicted-normal inconsistency and back-facing normals; combine with reflected-dir reparam. The canonical recipe that made specular reflections accurate. | Verbin et al., CVPR 2022 [2112.03907] ⭐ |
| 3.2 🅱 | **Monocular normal priors (DN-Splatter / GS-2DGS)** | 3DGS + foundation models | Supervise Gaussian normals against a pretrained monocular normal predictor (Omnidata/StableNormal). Foundation-model normals are *robust to reflective surfaces* (unlike MVS), giving the ASG a trustworthy `reflect_dir`. | DN-Splatter WACV 2025 [2403.17822]; GS-2DGS [2506.13110] |
| 3.3 🅱 | **Deferred-reflection normal propagation (3DGS-DR)** | 3DGS / graphics | Per-pixel reflection gradients from deferred shading propagate near-correct normals across neighboring Gaussians — solves the discontinuous-normal-gradient bottleneck we hit, at ~vanilla-3DGS frame rate. | Ye et al., SIGGRAPH 2024 [2404.18454] ⭐ |
| 3.4 🅱 | **2DGS surfel normals** | 3DGS | Switch primitives to view-consistent oriented disks whose normal is well-defined (vs. the ambiguous min-axis of a 3D blob). Used as the geometry backbone by most reflective-GS methods. | Huang et al., SIGGRAPH 2024 [2403.17888] |

---

## Category 4 — Parameter-efficient / conditioned capacity (capacity without bloat; thesis novelty)

How to *add* representational power to the per-Gaussian appearance without exploding the
24-dim×N storage or per-iter MLP cost. These are the flagged thesis-contribution angles.

| # | Solution | Field | What it does / why it fits | Cite |
|---|---|---|---|---|
| 4.1 🅲🧪 | **LoRA / low-rank ASG factorization** | NLP (PEFT) | Replace dense 24-dim per-Gaussian latents with a shared low-rank basis + per-Gaussian coefficients (B·A). Cuts memory, improves generalization, regularizes the latent. (CLAUDE.md fix #6.) | LoRA Hu et al., ICLR 2022 [2106.09685] ⭐; cf. TensoRF factorization [2203.09517] ⭐ |
| 4.2 🅲🧪 | **VQ codebook for ASG latents** | CV / compression | Quantize the per-Gaussian latent to a small learned codebook (CompGS). Shared "appearance words" act as a strong prior and shrink storage to kB range. | CompGS Navaneet et al., ECCV 2024 [2311.18159]; noise-substituted VQ [2504.03059] |
| 4.3 🅲 | **FiLM / hypernetwork conditioning** | ML (VQA→rendering) | Instead of feeding the 24-dim latent as MLP input, use it to *modulate* a shared MLP (`γ⊙h+β`). Far fewer per-Gaussian params than per-Gaussian weights, more expressive than concatenation. | FiLM Perez et al., AAAI 2018 [1709.07871] |
| 4.4 🅲 | **Residual / Hyper-Connections in `ASGRender`** | ML | Skip connections (already added: `h2 = relu(fc2(h1)) + h1`) ease optimization of the deeper specular MLP; Hyper-Connections generalize residual mixing. (CLAUDE.md Options 2/4.) | ResNet [1512.03385] ⭐; Residual-as-iterative-inference [1710.04773] ⭐; Hyper-Connections ICLR 2025 [2409.19606] ⭐ |

---

## Category 5 — Architectural relocation (whole-branch redesign)

If incremental fixes plateau, move specular out of the per-Gaussian forward shading model.

| # | Solution | Field | What it does / why it fits | Cite |
|---|---|---|---|---|
| 5.1 🅰🅱🧪 | **Deferred screen-space specular shading** | graphics | Render G-buffer (base color, normal, reflection strength) then shade specular per *pixel*, not per Gaussian. Removes the per-Gaussian MLP cost (speed) *and* gives smooth per-pixel normal gradients (quality) — hits both thesis goals at once. The single strongest redesign. | 3DGS-DR Ye et al., SIGGRAPH 2024 [2404.18454] ⭐ |
| 5.2 🅱 | **Shading-function reflective GS (GaussianShader)** | 3DGS | Per-Gaussian shading with explicit normals + a simplified BRDF; an alternative to the ASG residual that bakes in reflection structure. | Jiang et al., CVPR 2024 [2311.17977] ⭐ |

---

## Ranked recommendation (what to try, in order)

Ordered by **expected impact ÷ effort**, with the diagnostic symptom each is predicted to move:

1. **Specular-mask weighted L1 (1.1) + HDR/log tone-mapped loss (1.7).** ~20 lines.
   Directly attacks the dominant root cause (A). Predicted: `a*` → ~1.0, `energyRatio` up.
   *Do this first — it is the cheapest test of the whole diagnosis.*
2. **Focal Frequency Loss (1.2) and/or Laplacian-pyramid loss (1.4).** Adds the high-band
   gradient that de-blurs. Predicted: `σ` down. Stacks with #1.
3. **SIREN (2.1) or WIRE (2.3) activation in `ASGRender`.** Raises the frequency ceiling
   once supervision is fixed. Predicted: `σ` down, `energyRatio` up. Needs fresh training.
4. **Monocular normal priors (3.2) or Ref-NeRF normal reg + IDE (2.4/3.1).** Attacks the
   placement error. Predicted: `NCC` up, `structural%` down. Bigger change.
5. **Thesis-novelty track (parallel):** LoRA-ASG (4.1) or VQ-ASG (4.2) for the
   efficiency/representation contribution; **deferred shading (5.1)** as the headline
   redesign that unifies the speed and quality goals.

**Why this order:** the diagnostic says the deficit is *first* a drive problem (A), *then*
a capacity problem (A), *then* a geometry problem (B). Fixing capacity (SIREN) or normals
before fixing supervision would under-test each, because a high-capacity MLP with no
highlight gradient still learns the dim/smooth solution.

---

## Source map (arXiv)

Loss/supervision: FFL [2012.12821] · FreGS [2403.06908] · LAPGAN [1506.05751] ·
LapSRN [1704.03915] · OHEM [1604.03540] · Focal Loss [1708.02002] · ray/pixel importance
[2308.15547],[2408.03193] · RawNeRF [2111.13679] · HDR-NeRF [2111.14451].
Representation: SIREN [2006.09661] · Fourier features [2006.10739] · WIRE [2301.05187] ·
Ref-NeRF/IDE [2112.03907] · SpecNeRF [2312.13102] · PhySG [2104.00674] ·
GlossGau [2502.14129] · KAN [2404.19756] · KA-Fourier [2502.06018] ·
Instant-NGP [2201.05989] · Scaffold-GS [2312.00109] · TensoRF [2203.09517].
Normals: DN-Splatter [2403.17822] · GS-2DGS [2506.13110] · 3DGS-DR [2404.18454] ·
2DGS [2403.17888].
Efficiency/conditioning: LoRA [2106.09685] · CompGS [2311.18159] · noise-VQ [2504.03059] ·
FiLM [1709.07871] · ResNet [1512.03385] · Residual-iterative [1710.04773] ·
Hyper-Connections [2409.19606].
Redesign: 3DGS-DR [2404.18454] · GaussianShader [2311.17977].

*Items marked ⭐ already have PDFs under `papers/3dgs_papers/`. New-to-library candidates
worth downloading: FFL, FreGS, WIRE, Fourier-features, DN-Splatter, RawNeRF, HDR-NeRF,
PhySG, OHEM, Focal Loss, CompGS, FiLM, KAN.*
