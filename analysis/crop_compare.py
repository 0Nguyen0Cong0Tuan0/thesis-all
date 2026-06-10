import os, numpy as np
from PIL import Image
ROOT=r"D:\Thesis\All\mipnerf_counter_images4"; OUT=r"D:\Thesis\All\analysis_out"
# bowl region in 00000 (bottom-left metallic bowl) - approx
d=os.path.join(ROOT,"00000")
crop=(40,300,360,519)  # left,top,right,bottom
imgs=[]
for n in ["3_gt.png","2_spec_gauss.png","2_fastgs.png"]:
    im=Image.open(os.path.join(d,n)).convert("RGB").crop(crop)
    im=im.resize((im.width*2,im.height*2), Image.NEAREST)
    imgs.append(im)
W=sum(i.width for i in imgs)+20; H=max(i.height for i in imgs)
canvas=Image.new("RGB",(W,H),(255,255,255)); x=0
for i in imgs:
    canvas.paste(i,(x,0)); x+=i.width+10
canvas.save(os.path.join(OUT,"00000_bowl_GT_spec_fast.png"))
print("saved bowl crop: GT | spec_gauss | fastgs")
