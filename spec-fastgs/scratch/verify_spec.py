import imageio
import numpy as np

img = imageio.imread("/home/baodinh/baodinh_thesis/spec-fastgs/output/test_asg_active/test/ours_150/spec/00000_only_asg.png")
max_val = np.max(img)
mean_val = np.mean(img)
std_val = np.std(img)

print(f"Max pixel value: {max_val}")
print(f"Mean pixel value: {mean_val}")
print(f"Std pixel value: {std_val}")
