"""
Adapted analysis for the NEW results/ layout (single Spec-FastGS model, 30 test views).

Original analysis/*.py compared TWO models (spec_gauss vs fastgs) on 3 hand-picked
decomposition folders under mipnerf_counter_images4. The new run produces ONE model
(spec-fastgs) over all 30 mip-NeRF360 `counter` test views, with a per-view
decomposition saved under test/ours_30000/spec/XXXXX/.

Per-view files used here:
  render  = test/ours_30000/renders/XXXXX.png      (the actual model output)
  gt      = test/ours_30000/gt/XXXXX.png
  diffuse = test/ours_30000/spec/XXXXX/1_diffuse.png   (SH-only base)
  resid   = test/ours_30000/spec/XXXXX/5_residual.png  (highlight locator)

Outputs: per-view metrics aggregated (mean / std / worst), plus error heatmaps and a
specular-layer panel for a sample view, written to results/analysis_out/.
"""
import os, glob, numpy as np
from PIL import Image
import cv2
from scipy.ndimage import gaussian_filter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "results", "output", "counter", "test", "ours_30000")
OUT  = os.path.join(ROOT, "results", "analysis_out")
os.makedirs(OUT, exist_ok=True)

def load(p):
    return np.array(Image.open(p).convert("RGB")).astype(np.float64)

def gray(img):
    return cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2GRAY).astype(np.float64)

# ----- sharpness / frequency -----
def lap_var(g):      return cv2.Laplacian(g, cv2.CV_64F).var()
def tenengrad(g):
    gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
    return np.mean(gx**2 + gy**2)
def high_freq_energy(g):
    mag = np.abs(np.fft.fftshift(np.fft.fft2(g)))
    h, w = g.shape; cy, cx = h//2, w//2
    Y, X = np.ogrid[:h, :w]
    r = np.sqrt((Y-cy)**2 + (X-cx)**2); rmax = np.sqrt(cy**2+cx**2)
    return mag[r > 0.25*rmax].sum() / mag.sum()
def radial_spectrum(g):
    mag = np.abs(np.fft.fftshift(np.fft.fft2(g)))
    h, w = g.shape; cy, cx = h//2, w//2
    Y, X = np.ogrid[:h, :w]
    r = np.sqrt((Y-cy)**2 + (X-cx)**2).astype(int)
    tbin = np.bincount(r.ravel(), mag.ravel()); nr = np.bincount(r.ravel())
    return tbin / np.maximum(nr, 1)

def ssim_map(pred, gt):
    pg, gg = gray(pred), gray(gt)
    C1=(0.01*255)**2; C2=(0.03*255)**2
    mu1=cv2.GaussianBlur(pg.astype(np.float32),(11,11),1.5)
    mu2=cv2.GaussianBlur(gg.astype(np.float32),(11,11),1.5)
    s1=cv2.GaussianBlur((pg*pg).astype(np.float32),(11,11),1.5)-mu1**2
    s2=cv2.GaussianBlur((gg*gg).astype(np.float32),(11,11),1.5)-mu2**2
    s12=cv2.GaussianBlur((pg*gg).astype(np.float32),(11,11),1.5)-mu1*mu2
    return ((2*mu1*mu2+C1)*(2*s12+C2))/((mu1**2+mu2**2+C1)*(s1+s2+C2))

def psnr(pred, gt):
    mse = np.mean((pred-gt)**2)
    return 10*np.log10(255**2/mse) if mse > 0 else 99.0

views = sorted(os.path.splitext(os.path.basename(p))[0]
               for p in glob.glob(os.path.join(BASE, "gt", "*.png")))
print(f"Found {len(views)} test views under {BASE}\n")

# =====================================================================
# Accumulators
# =====================================================================
rows = []          # per-view scalar metrics
spec_rows = []     # per-view specular-decomposition metrics
midband, highband = [], []

for v in views:
    gt  = load(os.path.join(BASE, "gt", v+".png"))
    ren = load(os.path.join(BASE, "renders", v+".png"))
    dif = load(os.path.join(BASE, "spec", v, "1_diffuse.png"))
    res = load(os.path.join(BASE, "spec", v, "5_residual.png"))
    gg  = gray(gt); rg = gray(ren)

    # global
    p = psnr(ren, gt); sm = ssim_map(ren, gt); ssim = float(sm.mean())
    lv, tg, hf = lap_var(rg), tenengrad(rg), high_freq_energy(rg)
    gt_lv = lap_var(gg)

    # color bias
    d = ren - gt
    dR, dG, dB = d[...,0].mean(), d[...,1].mean(), d[...,2].mean()

    # error by luminance region (GT luminance)
    dark=gg<64; mid=(gg>=64)&(gg<=180); bright=gg>180
    se = ((ren-gt)**2).mean(axis=2)
    def rmse(m): return float(np.sqrt(se[m].mean())) if m.sum()>0 else 0.0
    # SSIM by region
    ss_dark, ss_mid, ss_bright = sm[dark].mean(), sm[mid].mean(), sm[bright].mean()

    # highlight (top-5% residual energy) error
    resmag = np.abs(res - res.mean()).mean(axis=2)
    hl = resmag >= np.percentile(resmag, 95)
    hl_rmse = float(np.sqrt(se[hl].mean())); hl_bias = float(d.mean(axis=2)[hl].mean())

    # radial spectrum ratio (render/gt)
    pg_spec = radial_spectrum(gg); p_spec = radial_spectrum(rg)
    m = min(len(pg_spec), len(p_spec))
    ratio = p_spec[:m] / np.maximum(pg_spec[:m], 1e-6)
    mb = float(np.mean(ratio[int(0.25*m):int(0.5*m)]))
    hb = float(np.mean(ratio[int(0.5*m):int(0.9*m)]))
    midband.append(mb); highband.append(hb)

    # specular-layer decomposition on highlight mask = top5% |gt_spec|
    gts = gt - dif; rs = ren - dif
    mag = np.abs(gts).mean(axis=2); shl = mag >= np.percentile(mag, 95)
    a = rs[shl].ravel(); b = gts[shl].ravel()
    gain = float(np.dot(a, b) / np.dot(a, a)) if np.dot(a, a) > 0 else 0.0
    am, bm = a - a.mean(), b - b.mean()
    ncc = float(np.dot(am, bm) / (np.linalg.norm(am)*np.linalg.norm(bm) + 1e-9))
    E_total = np.sum((a-b)**2); E_struct = np.sum((gain*a-b)**2)
    magnitude = 100*(E_total-E_struct)/E_total if E_total>0 else 0
    structural = 100*E_struct/E_total if E_total>0 else 0
    energyRatio = float(np.sum(np.abs(rs[shl])) / (np.sum(np.abs(gts[shl]))+1e-9))

    # blur sigma best-match on highlight pixels
    sigmas=[0,0.5,1,1.5,2,2.5,3,4,5]; errs=[]
    for s in sigmas:
        gb = gts if s==0 else np.stack([gaussian_filter(gts[...,c], s) for c in range(3)], axis=2)
        errs.append(np.sqrt(((rs[shl]-gb[shl])**2).mean()))
    errs=np.array(errs); bi=int(errs.argmin())
    err_drop = 100*(errs[0]-errs[bi])/errs[0] if errs[0]>0 else 0

    rows.append(dict(v=v, psnr=p, ssim=ssim, lv=lv, gt_lv=gt_lv, tg=tg, hf=hf,
                     dR=dR, dG=dG, dB=dB,
                     dark=rmse(dark), mid=rmse(mid), bright=rmse(bright),
                     bright_px=100*bright.mean(),
                     ss_dark=ss_dark, ss_mid=ss_mid, ss_bright=ss_bright,
                     hl_rmse=hl_rmse, hl_bias=hl_bias, mb=mb, hb=hb))
    spec_rows.append(dict(v=v, gain=gain, ncc=ncc, magnitude=magnitude,
                          structural=structural, energyRatio=energyRatio,
                          best_sigma=sigmas[bi], raw=errs[0], best=errs[bi], drop=err_drop))

def col(rs, k): return np.array([r[k] for r in rs], float)
def ms(rs, k): a=col(rs,k); return a.mean(), a.std()

print("="*100)
print("GLOBAL QUALITY  (mean +/- std over %d test views)" % len(views))
print("="*100)
for k,lab in [("psnr","PSNR (dB)"),("ssim","SSIM"),("lv","render LapVar"),
              ("gt_lv","GT LapVar"),("tg","Tenengrad"),("hf","HF-energy frac")]:
    mu,sd = ms(rows,k); print(f"  {lab:16}: {mu:10.3f}  +/- {sd:8.3f}")
sharp_ratio = ms(rows,"lv")[0]/ms(rows,"gt_lv")[0]
print(f"  render/GT LapVar ratio (sharpness; <1 = blurrier than GT): {sharp_ratio:.3f}")

print("\n"+"="*100)
print("COLOR BIAS  (render mean - GT mean), per channel")
print("="*100)
for k in ["dR","dG","dB"]:
    mu,sd = ms(rows,k); print(f"  d{k[1]}: {mu:7.3f} +/- {sd:5.3f}")

print("\n"+"="*100)
print("RMSE BY LUMINANCE REGION  (dark<64, mid 64-180, bright>180)")
print("="*100)
for k,lab in [("dark","dark RMSE"),("mid","mid RMSE"),("bright","bright RMSE"),
              ("bright_px","bright %px"),("hl_rmse","highlight RMSE (top5% resid)"),
              ("hl_bias","highlight lum bias (+=too bright)")]:
    mu,sd = ms(rows,k); print(f"  {lab:34}: {mu:8.3f} +/- {sd:7.3f}")

print("\n"+"="*100)
print("SSIM BY LUMINANCE REGION  (higher=better)")
print("="*100)
for k,lab in [("ss_dark","dark"),("ss_mid","mid"),("ss_bright","bright (highlights)")]:
    mu,sd = ms(rows,k); print(f"  {lab:22}: {mu:7.3f} +/- {sd:5.3f}")

print("\n"+"="*100)
print("RADIAL POWER SPECTRUM ratio render/GT  (<1 => render loses high-freq detail)")
print("="*100)
print(f"  mid band : {np.mean(midband):.3f} +/- {np.std(midband):.3f}")
print(f"  high band: {np.mean(highband):.3f} +/- {np.std(highband):.3f}")

print("\n"+"="*100)
print("SPECULAR-LAYER DECOMPOSITION  (render_spec vs gt_spec on top-5% |gt_spec| mask)")
print("  gain a*: brightness scale to best match (1=correct, <1=too dim)")
print("  NCC: structural correlation (1=perfect placement)")
print("  magnitude%%: error fixable by brightness | structural%%: blur/misplacement")
print("  energyRatio: render specular energy / GT  (<1 = under-produces specular)")
print("  best_sigma: blur applied to GT_spec to match render (>0 = render is blurrier)")
print("="*100)
for k,lab in [("gain","gain a*"),("ncc","NCC"),("magnitude","magnitude%"),
              ("structural","structural%"),("energyRatio","energyRatio"),
              ("best_sigma","best_sigma"),("drop","blur err_drop%")]:
    mu,sd = ms(spec_rows,k); print(f"  {lab:16}: {mu:8.3f} +/- {sd:7.3f}")

print("\n"+"="*100)
print("WORST-5 VIEWS BY SSIM")
print("="*100)
worst = sorted(rows, key=lambda r: r["ssim"])[:5]
print(f"  {'view':6} {'SSIM':>7} {'PSNR':>7} {'hl_RMSE':>8} {'hl_bias':>8} {'spec_NCC':>9}")
sd_by_v = {r["v"]: r for r in spec_rows}
for r in worst:
    sp = sd_by_v[r["v"]]
    print(f"  {r['v']:6} {r['ssim']:7.3f} {r['psnr']:7.2f} {r['hl_rmse']:8.2f} "
          f"{r['hl_bias']:8.2f} {sp['ncc']:9.3f}")

# =====================================================================
# Visualizations for a representative view (the median-SSIM one)
# =====================================================================
med = sorted(rows, key=lambda r: r["ssim"])[len(rows)//2]["v"]
gt  = load(os.path.join(BASE, "gt", med+".png"))
ren = load(os.path.join(BASE, "renders", med+".png"))
dif = load(os.path.join(BASE, "spec", med, "1_diffuse.png"))

err = np.sqrt(((ren-gt)**2).mean(axis=2))
errn = np.clip(err/40*255, 0, 255).astype(np.uint8)
cv2.imwrite(os.path.join(OUT, f"{med}_errheat.png"),
            cv2.applyColorMap(errn, cv2.COLORMAP_JET))

def norm(x): x=np.abs(x).mean(axis=2); return np.clip(x/30*255,0,255).astype(np.uint8)
panels=[norm(gt-dif), norm(ren-dif)]
hs=[cv2.applyColorMap(p, cv2.COLORMAP_INFERNO) for p in panels]
gap=np.full((hs[0].shape[0],8,3),255,np.uint8)
cv2.imwrite(os.path.join(OUT, f"{med}_speclayer_GT_vs_render.png"),
            np.hstack([hs[0],gap,hs[1]]))
print(f"\nWrote visualizations for median-SSIM view {med} -> {OUT}")
print(f"  {med}_errheat.png  (error heatmap)")
print(f"  {med}_speclayer_GT_vs_render.png  (GT specular | render specular energy)")
