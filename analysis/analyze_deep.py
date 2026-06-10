import os, numpy as np
from PIL import Image
import cv2

ROOT = r"D:\Thesis\All\mipnerf_counter_images4"
OUT = r"D:\Thesis\All\analysis_out"
os.makedirs(OUT, exist_ok=True)
folders = ["00000", "00001", "00002"]

def load(p): return np.array(Image.open(p).convert("RGB")).astype(np.float64)

def radial_spectrum(g):
    f = np.fft.fftshift(np.fft.fft2(g))
    mag = np.abs(f)
    h,w = g.shape; cy,cx=h//2,w//2
    Y,X=np.ogrid[:h,:w]
    r=np.sqrt((Y-cy)**2+(X-cx)**2).astype(int)
    rmax=r.max()
    tbin=np.bincount(r.ravel(), mag.ravel())
    nr=np.bincount(r.ravel())
    prof=tbin/np.maximum(nr,1)
    return prof

print("="*90)
print("1) PER-CHANNEL COLOR BIAS  (render mean - GT mean), per channel R/G/B")
print("="*90)
print(f"{'folder':8} {'render':14} {'dR':>7} {'dG':>7} {'dB':>7} {'|dRGB|':>7}")
for fld in folders:
    d=os.path.join(ROOT,fld); gt=load(os.path.join(d,"3_gt.png"))
    for name in ["2_spec_gauss.png","2_fastgs.png"]:
        img=load(os.path.join(d,name))
        diff=img-gt
        dR,dG,dB=diff[...,0].mean(),diff[...,1].mean(),diff[...,2].mean()
        print(f"{fld:8} {name:14} {dR:7.2f} {dG:7.2f} {dB:7.2f} {np.sqrt(dR**2+dG**2+dB**2):7.2f}")
    print()

print("="*90)
print("2) ERROR BY LUMINANCE REGION  (RMSE of render vs GT, split by GT luminance)")
print("   dark = GT lum<64, mid = 64-180, bright = >180 (highlights/specular)")
print("="*90)
print(f"{'folder':8} {'render':14} {'dark_RMSE':>10} {'mid_RMSE':>9} {'bright_RMSE':>11} {'bright_%px':>10}")
for fld in folders:
    d=os.path.join(ROOT,fld); gt=load(os.path.join(d,"3_gt.png"))
    gl=cv2.cvtColor(gt.astype(np.float32),cv2.COLOR_RGB2GRAY)
    dark=gl<64; mid=(gl>=64)&(gl<=180); bright=gl>180
    for name in ["2_spec_gauss.png","2_fastgs.png"]:
        img=load(os.path.join(d,name))
        se=((img-gt)**2).mean(axis=2)
        def rmse(m): return np.sqrt(se[m].mean()) if m.sum()>0 else 0
        print(f"{fld:8} {name:14} {rmse(dark):10.2f} {rmse(mid):9.2f} {rmse(bright):11.2f} {100*bright.mean():10.2f}")
    print()

print("="*90)
print("3) ERROR IN SPECULAR/HIGHLIGHT REGIONS (top 5% by GT residual energy)")
print("   uses 5_residual.png to locate true highlights")
print("="*90)
print(f"{'folder':8} {'render':14} {'highlight_RMSE':>14} {'highlight_lumBias':>17}")
for fld in folders:
    d=os.path.join(ROOT,fld); gt=load(os.path.join(d,"3_gt.png"))
    rp=os.path.join(d,"5_residual.png")
    res=load(rp); resmag=np.abs(res-res.mean()).mean(axis=2) if res.ndim==3 else res
    thr=np.percentile(resmag,95); hl=resmag>=thr
    for name in ["2_spec_gauss.png","2_fastgs.png"]:
        img=load(os.path.join(d,name))
        se=((img-gt)**2).mean(axis=2)
        rmse=np.sqrt(se[hl].mean())
        bias=(img-gt).mean(axis=2)[hl].mean()  # +ve = render too bright
        print(f"{fld:8} {name:14} {rmse:14.2f} {bias:17.2f}")
    print()

print("="*90)
print("4) RADIAL POWER SPECTRUM ratio (render/GT) at HIGH freq band")
print("   value <1 => render has LESS high-freq detail than GT (=blurrier)")
print("="*90)
print(f"{'folder':8} {'render':14} {'mid_band':>9} {'high_band':>10}")
for fld in folders:
    d=os.path.join(ROOT,fld); gt=load(os.path.join(d,"3_gt.png"))
    gg=cv2.cvtColor(gt.astype(np.float32),cv2.COLOR_RGB2GRAY).astype(np.float64)
    pg=radial_spectrum(gg); n=len(pg)
    for name in ["2_spec_gauss.png","2_fastgs.png"]:
        img=load(os.path.join(d,name))
        g=cv2.cvtColor(img.astype(np.float32),cv2.COLOR_RGB2GRAY).astype(np.float64)
        p=radial_spectrum(g)
        m=min(len(p),n); ratio=p[:m]/np.maximum(pg[:m],1e-6)
        mid=np.mean(ratio[int(0.25*m):int(0.5*m)])
        high=np.mean(ratio[int(0.5*m):int(0.9*m)])
        print(f"{fld:8} {name:14} {mid:9.3f} {high:10.3f}")
    print()

# Visualizations: error heatmaps and difference for folder 00000
print("Writing visualizations to", OUT)
for fld in folders:
    d=os.path.join(ROOT,fld); gt=load(os.path.join(d,"3_gt.png"))
    for name in ["2_spec_gauss.png","2_fastgs.png"]:
        img=load(os.path.join(d,name))
        err=np.sqrt(((img-gt)**2).mean(axis=2))
        errn=np.clip(err/40*255,0,255).astype(np.uint8)
        hm=cv2.applyColorMap(errn,cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(OUT,f"{fld}_{name.replace('.png','')}_errheat.png"),hm)
print("done")
