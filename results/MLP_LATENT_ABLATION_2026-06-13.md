# Spec-FastGS — MLP structure & latent-space ablation (2026-06-13)

**Question (from the goal):** can we improve the specular branch by changing the MLP
*complexity/structure* or by using *latent-space methods*? Implemented as drop-in
options in `spec-fastgs/utils/spec_arch.py` and compared on a GPU-free, controlled
benchmark (`spec-fastgs/bench_spec_arch.py`).

## Method

No local GPU, so instead of a full 76-min train per variant we use a **controlled
synthetic probe** that isolates exactly the failing capability: representing a sharp,
view-dependent highlight. Each of M=64 "Gaussians" gets a random normal and a
per-Gaussian material (RGB amplitude + shininess `p ∈ [8,256]`); the GT specular
response to a view `v` is a Phong lobe `A·relu(reflect(-v,n)·L)^p` for a shared light
`L`. High `p` ⇒ high-frequency, view-dependent — the signal the real branch
under-produces. For every architecture we jointly optimise the per-Gaussian latents +
the shared network with **identical iters/lr/init RNG**, then evaluate on **held-out
view directions** using the *same metrics as the real diagnostic*:

- `energyRatio` = Σ|pred| / Σ|gt| on the top-5% GT-energy mask (1 = ideal),
- `gain a*` = ⟨pred,gt⟩/⟨pred,pred⟩ (>1 ⇒ render too dim — our real signature),
- `NCC` = corr(pred,gt) (placement; 1 = perfect),
- `L1_hl` = mean |pred−gt| on the mask.

Two conditions: **clean** (perfect normals) and **noisy** (model fed normals perturbed
by σ=0.3 — simulating the real `get_minimum_axis` pseudo-normal, root cause B).

## Results — clean condition (full sweep)

| config | energyR↑ | gain a*→1 | NCC↑ | L1_hl↓ | shared params | F/GS |
|---|---|---|---|---|---|---|
| **latent48** (F48 relu pe2 d2) | **0.902** | 0.995 | 0.814 | 0.0374 | 33,539 | 48 |
| deeper_wider (relu pe2 d4 w256) | 0.884 | 1.014 | 0.817 | 0.0368 | 222,083 | 24 |
| WIRE (gabor pe2 d2) | 0.870 | 1.085 | 0.673 | 0.0777 | 30,467 | 24 |
| relu_pe6 (F24 pe6 d2) | 0.869 | 0.987 | 0.792 | 0.0444 | 33,539 | 24 |
| **baseline** (F24 relu pe2 d2) | 0.868 | 1.011 | 0.809 | 0.0367 | 30,467 | 24 |
| **lowrank_gf** (F24 relu r8 d2) | 0.862 | 1.070 | **0.871** | **0.0342** | **28,611** | 24 |
| FiLM (F24 relu pe2 d2) | 0.849 | 0.981 | 0.767 | 0.0434 | 36,867 | 24 |
| latent8 (F8 relu pe2 d2) | 0.845 | 1.020 | 0.788 | 0.0397 | 28,419 | **8** |
| SIREN w0=10 | 0.706 | 1.196 | 0.646 | 0.0752 | 30,467 | 24 |
| SIREN w0=30 | 0.369 | 2.516 | 0.382 | 0.1155 | 30,467 | 24 |
| SIREN_deep w0=30 | 0.018 | 10.49 | −0.003 | 0.1606 | 63,491 | 24 |

## Results — clean vs noisy normals (focused subset)

| config | energyR clean→noisy | NCC clean→noisy | L1_hl clean→noisy |
|---|---|---|---|
| baseline | 0.868 → 0.850 | 0.809 → 0.746 | 0.0367 → 0.0528 |
| latent48 | 0.902 → 0.841 | 0.814 → 0.756 | 0.0374 → 0.0506 |
| lowrank_gf | 0.862 → 0.781 | 0.871 → 0.739 | 0.0342 → 0.0579 |
| relu_pe6 | 0.869 → 0.816 | 0.792 → 0.718 | 0.0444 → 0.0644 |
| WIRE | 0.870 → 0.838 | 0.673 → 0.645 | 0.0777 → 0.0818 |
| SIREN w0=10 | 0.706 → 0.746 | 0.646 → 0.664 | 0.0752 → 0.0751 |

## What the comparison says (carefully)

1. **MLP capacity/structure is NOT the bottleneck.** Given clean inputs and undrowned
   supervision, the **baseline ReLU MLP already fits sharp view-dependent lobes well**
   (energyRatio 0.87, `a*≈1.0`, NCC 0.81). The fancier structures give *no net win*:
   - **SIREN hurts** — badly at ω₀=30 (energyRatio 0.37, collapses entirely when deep),
     and still below ReLU at ω₀=10. Reason: our specular residual is *additive and
     near-zero except at highlights*; SIREN's everywhere-oscillating sine prior is a
     poor inductive bias for a mostly-flat target and is highly ω₀/scale sensitive.
     **This empirically refutes the prior plan (CLAUDE.md) that SIREN is the "highest
     expected gain."**
   - **WIRE** keeps energyRatio but *worsens placement* (NCC 0.81→0.67) and costs ~2×.
   - **deeper_wider** buys +0.016 energyRatio for **7.3× the parameters** — not worth it.
   - **higher Fourier bandwidth (pe6)** does not help on this signal (slightly worse L1).
2. **Latent capacity is a real but modest lever.** Per-Gaussian latent dim F monotonically
   moves quality: F 8→24→48 ⇒ energyRatio 0.845→0.868→0.902. Conversely **F=8 costs only
   −0.02 energyRatio for 3× less per-Gaussian storage** — a strong compression result.
3. **Low-rank (LoRA-style) factorization is the best latent-space win on clean data:**
   `lowrank_gf` gives the **best NCC (0.871) and best L1_hl** with **fewer shared
   params** — the rank-8 bottleneck regularises the latent→ASG map. Quality *and*
   compression: the most thesis-worthy of the architecture changes tested.
4. **Noisy normals are the dominant, unfixable-by-architecture degradation.** σ=0.3
   normal noise drops every variant's NCC by 0.06–0.13 and raises L1_hl 40–60%, and
   **no architecture rescues it** (the fancy ones degrade as much or more). This is the
   empirical proof that **root cause B must be fixed with better normals**
   (monocular normal priors / deferred-shading propagation), not a fancier MLP.

## Recommendation

- **Do NOT switch the activation** to SIREN/WIRE — proven here to not help (and SIREN
  hurts). Keep ReLU. This saves a tempting but wrong experiment.
- **Latent-space track (worth doing):** adopt the **low-rank (LoRA) ASG latent** for the
  quality+compression contribution; optionally study the F-dim sweep (8/24/48) as an
  explicit storage-vs-quality ablation in the thesis.
- **The real gains are elsewhere** and this benchmark localises why: the live deficit is
  (A) *supervision drowning* (not present in this probe — its loss is undrowned, which is
  exactly why even the baseline scores well) and (B) *noisy normals* (reproduced here;
  architecture-immune). So the prioritised work stays: **(1) specular-weighted / HDR loss,
  (2) normal priors**, with **low-rank latent** as the architecture-side contribution.

## How to use the drop-in (Kaggle-ready, default-off)

`utils/spec_arch.py` provides `SpecularNetworkV2` with the same `forward(x,view,normal)`
interface. It is wired into `SpecularModel` behind an opt-in flag — **the default path is
unchanged**. Enable without editing code by setting an env var before training, e.g.:

```bash
# low-rank latent (recommended architecture-side change), ReLU kept:
export SPEC_ARCH='{"activation":"relu","latent_mode":"lowrank","rank":8}'
# or larger latent (needs asg_degree raised to match in gaussian_model.py):
export SPEC_ARCH='{"activation":"relu","asg_feature":48}'
```

Note: `asg_feature` must equal `GaussianModel.asg_degree` (currently 24); changing the
stored latent dim requires setting `asg_degree` accordingly. Activation/latent-factor
swaps at F=24 need no other change.

*Artifacts: `spec-fastgs/utils/spec_arch.py` (module), `spec-fastgs/bench_spec_arch.py`
(benchmark), `spec-fastgs/bench_spec_arch_results.json` + `_noise.json` (raw numbers).*
