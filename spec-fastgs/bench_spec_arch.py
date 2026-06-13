"""
GPU-free ablation of specular MLP structure & latent-space choices.

Synthetic task mirrors the real failure mode: each "Gaussian" has a random normal
and a per-Gaussian material (RGB amplitude + shininess); the GT specular response to
a view direction is a sharp Phong lobe  A * relu(reflect(-v,n) . L)^p  for a shared
light L.  High shininess p => high-frequency, view-dependent highlight -- the exact
signal the ASG+MLP branch under-produces (dim a*>1, blurry sigma~4.9) in training.

For every architecture we train the SAME way (same iters / lr / init RNG): jointly
optimise the per-Gaussian latents + the shared network, then evaluate on HELD-OUT
view directions.  Metrics replicate the real diagnostic:
  energyRatio = sum|pred| / sum|gt|        on the top-5% GT-energy mask  (1=ideal)
  gain a*     = <pred,gt>/<pred,pred>       on the mask  (>1 => pred too dim)
  NCC         = corr(pred,gt)               on the mask  (1=perfect placement)
  L1_hl       = mean |pred-gt|              on the mask
plus shared-param count and per-Gaussian latent storage (the latent-space cost).
"""
import sys, os, time, math, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

from utils.spec_arch import SpecularNetworkV2, count_params

DEV = "cpu"
torch.set_num_threads(max(1, os.cpu_count() // 2))


def unit(x, dim=-1):
    return x / (x.norm(dim=dim, keepdim=True) + 1e-8)


def make_dataset(M=96, Vtr=192, Vte=96, seed=0):
    g = torch.Generator().manual_seed(seed)
    normals = unit(torch.randn(M, 3, generator=g))
    L = unit(torch.randn(1, 3, generator=g))                      # shared light dir
    ampl = 0.3 + 0.7 * torch.rand(M, 3, generator=g)             # per-Gaussian RGB
    shininess = (2.0 ** (3 + 5 * torch.rand(M, 1, generator=g))) # 8 .. 256 (sharpness)
    V = Vtr + Vte
    views = unit(torch.randn(V, 3, generator=g))                  # camera directions
    vtr, vte = views[:Vtr], views[Vtr:]

    def target(viewdirs):
        # reflect(-v, n) per (g,v):  r = 2(n.d)n - d  with d = -v
        n = normals[:, None, :]                                   # [M,1,3]
        d = -viewdirs[None, :, :]                                 # [1,V,3]
        r = unit(2 * (n * d).sum(-1, keepdim=True) * n - d)       # [M,V,3]
        rl = (r * L[:, None, :]).sum(-1, keepdim=True).clamp_min(0)  # [M,V,1]
        spec = ampl[:, None, :] * rl.pow(shininess[:, None, :])   # [M,V,3]
        return spec
    return normals, vtr, vte, target(vtr), target(vte), M


def build(cfg, M):
    torch.manual_seed(123)                                        # identical init RNG
    net = SpecularNetworkV2(
        asg_feature=cfg["F"], activation=cfg["act"], viewpe=cfg["pe"],
        depth=cfg["depth"], featureC=cfg["w"], latent_mode=cfg["latent"],
        rank=cfg.get("rank", 8), film=cfg.get("film", False),
        omega_0=cfg.get("omega0", 30.0), device=DEV,
    ).to(DEV)
    latents = torch.nn.Parameter(torch.zeros(M, cfg["F"], device=DEV))  # init 0 (as real code)
    return net, latents


def run(cfg, data, iters=2000, lr=5e-3, normal_noise=0.0):
    normals, vtr, vte, ytr, yte, M = data
    net, latents = build(cfg, M)
    Vtr = vtr.shape[0]
    # Optionally corrupt the normals the MODEL sees (targets keep clean normals):
    # simulates the noisy get_minimum_axis pseudo-normal (root cause B).
    if normal_noise > 0:
        gg = torch.Generator().manual_seed(7)
        mnormals = unit(normals + normal_noise * torch.randn(normals.shape, generator=gg))
    else:
        mnormals = normals
    # pre-expand inputs to [M*V, .]
    n_tr = mnormals[:, None, :].expand(M, Vtr, 3).reshape(-1, 3)
    v_tr = vtr[None, :, :].expand(M, Vtr, 3).reshape(-1, 3)
    y_tr = ytr.reshape(-1, 3)
    opt = torch.optim.Adam(list(net.parameters()) + [latents], lr=lr)
    t0 = time.time()
    for it in range(iters):
        lat = latents[:, None, :].expand(M, Vtr, cfg["F"]).reshape(-1, cfg["F"])
        pred = net(lat, v_tr, n_tr)
        loss = (pred - y_tr).abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    train_t = time.time() - t0

    # ---- eval on held-out views ----
    net.eval()
    Vte = vte.shape[0]
    with torch.no_grad():
        n_te = mnormals[:, None, :].expand(M, Vte, 3).reshape(-1, 3)
        v_te = vte[None, :, :].expand(M, Vte, 3).reshape(-1, 3)
        lat = latents[:, None, :].expand(M, Vte, cfg["F"]).reshape(-1, cfg["F"])
        pred = net(lat, v_te, n_te)
        gt = yte.reshape(-1, 3)
        mag = gt.abs().mean(-1)
        thr = torch.quantile(mag, 0.95)
        hl = mag >= thr
        a = pred[hl].reshape(-1); b = gt[hl].reshape(-1)
        energyRatio = (a.abs().sum() / (b.abs().sum() + 1e-9)).item()
        gain = (torch.dot(a, b) / (torch.dot(a, a) + 1e-9)).item()
        am, bm = a - a.mean(), b - b.mean()
        ncc = (torch.dot(am, bm) / (am.norm() * bm.norm() + 1e-9)).item()
        l1_hl = (a - b).abs().mean().item()
        l1_all = (pred - gt).abs().mean().item()

    shared = count_params(net)
    per_gauss = cfg["F"]                                          # floats / Gaussian
    return dict(name=cfg["name"], energyRatio=energyRatio, gain=gain, ncc=ncc,
                l1_hl=l1_hl, l1_all=l1_all, shared=shared,
                per_gauss=per_gauss, train_t=train_t)


CONFIGS = [
    dict(name="baseline (F24 relu pe2 d2 w128)", F=24, act="relu", pe=2, depth=2, w=128, latent="dense"),
    dict(name="latent8  (F8  relu pe2 d2 w128)", F=8,  act="relu", pe=2, depth=2, w=128, latent="dense"),
    dict(name="latent48 (F48 relu pe2 d2 w128)", F=48, act="relu", pe=2, depth=2, w=128, latent="dense"),
    dict(name="relu_pe6 (F24 relu pe6 d2 w128)", F=24, act="relu", pe=6, depth=2, w=128, latent="dense"),
    dict(name="SIREN    (F24 sin  pe2 d2 w128)", F=24, act="siren", pe=2, depth=2, w=128, latent="dense", omega0=30.0),
    dict(name="WIRE     (F24 gabor pe2 d2 w128)",F=24, act="wire", pe=2, depth=2, w=128, latent="dense", omega0=10.0),
    dict(name="deeper_wider (relu pe2 d4 w256)", F=24, act="relu", pe=2, depth=4, w=256, latent="dense"),
    dict(name="SIREN_deep (F24 sin pe2 d4 w128)",F=24, act="siren", pe=2, depth=4, w=128, latent="dense", omega0=30.0),
    dict(name="lowrank_gf (F24 relu r8 d2)",     F=24, act="relu", pe=2, depth=2, w=128, latent="lowrank", rank=8),
    dict(name="FiLM     (F24 relu pe2 d2 w128)", F=24, act="relu", pe=2, depth=2, w=128, latent="dense", film=True),
]


def main():
    data = make_dataset()
    rows = [run(c, data) for c in CONFIGS]
    rows_sorted = sorted(rows, key=lambda r: -r["energyRatio"])
    hdr = f"{'config':<32}{'energyR':>8}{'gain a*':>8}{'NCC':>7}{'L1_hl':>8}{'L1_all':>8}{'shared':>9}{'F/GS':>6}{'t(s)':>7}"
    print("\nSynthetic specular ablation  (held-out views; energyRatio↑ gain a*→1 NCC↑ L1↓)")
    print("=" * len(hdr)); print(hdr); print("-" * len(hdr))
    for r in rows_sorted:
        print(f"{r['name']:<32}{r['energyRatio']:>8.3f}{r['gain']:>8.3f}{r['ncc']:>7.3f}"
              f"{r['l1_hl']:>8.4f}{r['l1_all']:>8.4f}{r['shared']:>9d}{r['per_gauss']:>6d}{r['train_t']:>7.1f}")
    print("=" * len(hdr))
    with open(os.path.join(os.path.dirname(__file__), "bench_spec_arch_results.json"), "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
