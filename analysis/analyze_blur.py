import os, numpy as np
from PIL import Image
import cv2

ROOT = r"D:\Thesis\All\mipnerf_counter_images4"
folders = ["00000", "00001", "00002"]

def load(p):
    return np.array(Image.open(p).convert("RGB")).astype(np.float64)

def lap_var(img_gray):
    return cv2.Laplacian(img_gray, cv2.CV_64F).var()

def tenengrad(g):
    gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
    return np.mean(gx**2 + gy**2)

def high_freq_energy(g):
    f = np.fft.fft2(g)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    h, w = g.shape
    cy, cx = h//2, w//2
    # mask out low freq (central 1/8 radius)
    Y, X = np.ogrid[:h, :w]
    r = np.sqrt((Y-cy)**2 + (X-cx)**2)
    rmax = np.sqrt(cy**2+cx**2)
    high = mag[r > 0.25*rmax].sum()
    total = mag.sum()
    return high/total

def metrics_pair(pred, gt):
    # pred, gt float64 0-255
    mse = np.mean((pred-gt)**2)
    psnr = 10*np.log10(255**2/mse) if mse>0 else 99
    # SSIM (gray)
    pg = cv2.cvtColor(pred.astype(np.float32), cv2.COLOR_RGB2GRAY)
    gg = cv2.cvtColor(gt.astype(np.float32), cv2.COLOR_RGB2GRAY)
    C1=(0.01*255)**2; C2=(0.03*255)**2
    mu1=cv2.GaussianBlur(pg,(11,11),1.5); mu2=cv2.GaussianBlur(gg,(11,11),1.5)
    mu1s=mu1**2; mu2s=mu2**2; mu12=mu1*mu2
    s1=cv2.GaussianBlur(pg*pg,(11,11),1.5)-mu1s
    s2=cv2.GaussianBlur(gg*gg,(11,11),1.5)-mu2s
    s12=cv2.GaussianBlur(pg*gg,(11,11),1.5)-mu12
    ssim_map=((2*mu12+C1)*(2*s12+C2))/((mu1s+mu2s+C1)*(s1+s2+C2))
    ssim=ssim_map.mean()
    return psnr, ssim

print(f"{'folder':8} {'img':14} {'res':12} {'LapVar':>9} {'Tenengrad':>10} {'HFenergy':>9} {'meanLum':>8} {'PSNR':>6} {'SSIM':>6}")
print("-"*100)

summary = {}
for fld in folders:
    d = os.path.join(ROOT, fld)
    gt = load(os.path.join(d, "3_gt.png"))
    gtg = cv2.cvtColor(gt.astype(np.float32), cv2.COLOR_RGB2GRAY)
    for name in ["3_gt.png","2_spec_gauss.png","2_fastgs.png","2_composite.png"]:
        p = os.path.join(d, name)
        if not os.path.exists(p): continue
        img = load(p)
        g = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2GRAY).astype(np.float64)
        lv = lap_var(g)
        tg = tenengrad(g)
        hf = high_freq_energy(g)
        lum = img.mean()
        if name=="3_gt.png":
            psnr,ssim = 0,0
        else:
            psnr,ssim = metrics_pair(img, gt)
        res = f"{img.shape[1]}x{img.shape[0]}"
        print(f"{fld:8} {name:14} {res:12} {lv:9.1f} {tg:10.1f} {hf:9.4f} {lum:8.2f} {psnr:6.2f} {ssim:6.3f}")
    print()
