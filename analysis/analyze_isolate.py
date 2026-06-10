import os, numpy as np
from PIL import Image
import cv2
from scipy.ndimage import gaussian_filter

ROOT=r"D:\Thesis\All\mipnerf_counter_images4"; OUT=r"D:\Thesis\All\analysis_out"
os.makedirs(OUT,exist_ok=True)
folders=["00000","00001","00002"]
def load(p): return np.array(Image.open(p).convert("RGB")).astype(np.float64)

def ssim_map(pred,gt):
    pg=cv2.cvtColor(pred.astype(np.float32),cv2.COLOR_RGB2GRAY)
    gg=cv2.cvtColor(gt.astype(np.float32),cv2.COLOR_RGB2GRAY)
    C1=(0.01*255)**2;C2=(0.03*255)**2
    mu1=cv2.GaussianBlur(pg,(11,11),1.5);mu2=cv2.GaussianBlur(gg,(11,11),1.5)
    s1=cv2.GaussianBlur(pg*pg,(11,11),1.5)-mu1**2
    s2=cv2.GaussianBlur(gg*gg,(11,11),1.5)-mu2**2
    s12=cv2.GaussianBlur(pg*gg,(11,11),1.5)-mu1*mu2
    return ((2*mu1*mu2+C1)*(2*s12+C2))/((mu1**2+mu2**2+C1)*(s1+s2+C2))

print("="*95)
print("A) SSIM by luminance region (higher=better). dark<64, mid 64-180, bright>180")
print("="*95)
print(f"{'folder':8} {'render':12} {'dark':>7} {'mid':>7} {'bright':>7} {'global':>7}")
for fld in folders:
    d=os.path.join(ROOT,fld);gt=load(os.path.join(d,"3_gt.png"))
    gl=cv2.cvtColor(gt.astype(np.float32),cv2.COLOR_RGB2GRAY)
    dark=gl<64;mid=(gl>=64)&(gl<=180);bright=gl>180
    for n in ["2_spec_gauss.png","2_fastgs.png"]:
        sm=ssim_map(load(os.path.join(d,n)),gt)
        print(f"{fld:8} {n.replace('2_','').replace('.png',''):12} {sm[dark].mean():7.3f} {sm[mid].mean():7.3f} {sm[bright].mean():7.3f} {sm.mean():7.3f}")
    print()

print("="*95)
print("B) SPECULAR-LAYER DECOMPOSITION  (spec = render - diffuse ; gt_spec = gt - diffuse)")
print("   evaluated on highlight mask = top 5% of |gt_spec|")
print("   gain a* = optimal global brightness scale that best matches render_spec to gt_spec")
print("   E_total split: magnitude%% (fixable by brightness)  vs  structural%% (blur/misplaced)")
print("="*95)
print(f"{'folder':8} {'render':12} {'gain_a*':>8} {'NCC':>6} {'magnitude%':>11} {'structural%':>12} {'energyRatio':>12}")
for fld in folders:
    d=os.path.join(ROOT,fld)
    gt=load(os.path.join(d,"3_gt.png"));dif=load(os.path.join(d,"1_diffuse.png"))
    gts=gt-dif
    mag=np.abs(gts).mean(axis=2);thr=np.percentile(mag,95);hl=mag>=thr
    for n in ["2_spec_gauss.png","2_fastgs.png"]:
        rs=load(os.path.join(d,n))-dif
        a=rs[hl];b=gts[hl]
        a_flat=a.ravel();b_flat=b.ravel()
        gain=np.dot(a_flat,b_flat)/np.dot(a_flat,a_flat)
        ncc=np.dot(a_flat-a_flat.mean(),b_flat-b_flat.mean())/(np.linalg.norm(a_flat-a_flat.mean())*np.linalg.norm(b_flat-b_flat.mean()))
        E_total=np.sum((a_flat-b_flat)**2)
        E_struct=np.sum((gain*a_flat-b_flat)**2)
        magnitude=100*(E_total-E_struct)/E_total
        structural=100*E_struct/E_total
        energyRatio=np.sum(np.abs(a))/np.sum(np.abs(b))  # <1 = render produces less specular energy
        print(f"{fld:8} {n.replace('2_','').replace('.png',''):12} {gain:8.3f} {ncc:6.3f} {magnitude:11.1f} {structural:12.1f} {energyRatio:12.3f}")
    print()

print("="*95)
print("C) BLUR TEST: find Gaussian sigma where blur(gt_spec, s) best matches render_spec (highlight px)")
print("   larger best-sigma => render applied MORE blur to specular. err_gain = how much blur reduces err")
print("="*95)
sigmas=[0,0.5,1,1.5,2,2.5,3,4,5]
print(f"{'folder':8} {'render':12} {'best_sigma':>10} {'rawErr':>8} {'bestErr':>8} {'err_drop%':>9}")
for fld in folders:
    d=os.path.join(ROOT,fld)
    gt=load(os.path.join(d,"3_gt.png"));dif=load(os.path.join(d,"1_diffuse.png"))
    gts=gt-dif
    mag=np.abs(gts).mean(axis=2);thr=np.percentile(mag,95);hl=mag>=thr
    for n in ["2_spec_gauss.png","2_fastgs.png"]:
        rs=load(os.path.join(d,n))-dif
        errs=[]
        for s in sigmas:
            gb=gts if s==0 else np.stack([gaussian_filter(gts[...,c],s) for c in range(3)],axis=2)
            errs.append(np.sqrt(((rs[hl]-gb[hl])**2).mean()))
        errs=np.array(errs);bi=errs.argmin()
        drop=100*(errs[0]-errs[bi])/errs[0]
        print(f"{fld:8} {n.replace('2_','').replace('.png',''):12} {sigmas[bi]:10.1f} {errs[0]:8.2f} {errs[bi]:8.2f} {drop:9.1f}")
    print()

# Visualize: GT specular layer vs each render specular layer (folder 00000), normalized
fld="00000";d=os.path.join(ROOT,fld)
gt=load(os.path.join(d,"3_gt.png"));dif=load(os.path.join(d,"1_diffuse.png"))
def norm(x):
    x=np.abs(x).mean(axis=2);return np.clip(x/30*255,0,255).astype(np.uint8)
panels=[("GT_spec",norm(gt-dif)),
        ("spec_gauss_spec",norm(load(os.path.join(d,"2_spec_gauss.png"))-dif)),
        ("fastgs_spec",norm(load(os.path.join(d,"2_fastgs.png"))-dif))]
hs=[cv2.applyColorMap(p,cv2.COLORMAP_INFERNO) for _,p in panels]
gap=np.full((hs[0].shape[0],8,3),255,np.uint8)
combo=np.hstack([hs[0],gap,hs[1],gap,hs[2]])
cv2.imwrite(os.path.join(OUT,"00000_speclayer_GT_spec_fast.png"),combo)
print("saved 00000_speclayer_GT_spec_fast.png  (GT | spec_gauss | fastgs specular energy)")
