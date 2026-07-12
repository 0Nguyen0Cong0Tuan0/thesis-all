# Thiết Kế Supervision Signal — Tăng Cường Tín Hiệu Học Cho Specular

Dựa trên phân tích trong `Pipeline_analysis.md`, vấn đề cốt lõi của trục Supervision Signal là vùng specular quá nhỏ (thường 1-10% diện tích ảnh), dẫn đến gradient đóng góp vào tổng loss rất bé. Nếu chỉ dùng L1 + SSIM trung bình toàn ảnh, model sẽ tối ưu vùng diffuse (chiếm 90%) và bỏ qua specular.

Dưới đây là các phương án implementation để tăng cường tín hiệu học cho vùng specular, sắp xếp theo độ ưu tiên.

---

## 1. Specular-Weighted Loss (Ưu tiên Cao - Dễ thực hiện)

**Mục tiêu:** Ép model chịu phạt (penalty) nặng hơn nếu render sai ở các pixel được đánh dấu là specular (dựa trên `cam.ref_score`).

**Thiết kế:**
Thay vì lấy trung bình L1 toàn ảnh, ta nhân ma trận lỗi pixel-wise L1 với một ma trận trọng số (weight map). Vùng có `ref_score` cao sẽ có trọng số lớn hơn.
**Quan trọng:** Cần chuẩn hóa (normalize) loss để tổng cường độ gradient không bị bùng nổ khi weight thay đổi.

**Implementation (trong `train.py`):**

```python
# Cần thêm vào arguments/__init__.py: 
# self.lambda_spec_l1_weight = 0.0  # Default 0.0 để giữ nguyên baseline

gt = cam.original_image.cuda()
pixel_l1 = torch.abs(image - gt) # [3, H, W]

if opt.lambda_spec_l1_weight > 0 and hasattr(cam, 'ref_score'):
    # ref_score nằm trong [0, 1]
    ref_w = cam.ref_score.cuda().unsqueeze(0) # [1, H, W]
    
    # Weight map = 1.0 (base) + lambda * ref_score
    # Vùng ref_score = 1 sẽ có weight = 1 + lambda
    weight_map = 1.0 + opt.lambda_spec_l1_weight * ref_w
    
    # Nhân trọng số và chuẩn hóa
    # Chia cho sum(weight_map) để giữ giá trị loss tương đương L1 bình thường
    Ll1 = (pixel_l1 * weight_map).sum() / (3.0 * weight_map.sum().clamp_min(1e-6))
else:
    Ll1 = pixel_l1.mean()
```

**Tác động:** Trực tiếp tăng gradient chảy ngược về cả SH và ASG tại vùng specular. Nên kết hợp với R1 (SH Masking) để gradient tăng thêm này đi hoàn toàn vào ASG.

---

## 2. Kích hoạt Specular Regularization (Ưu tiên Cao - Fix bug)

**Mục tiêu:** Ngăn chặn ASG predict ra các giá trị màu bùng nổ (quá sáng) hoặc giá trị âm (để bù trừ lỗi của SH).

**Thiết kế:**
Tham số `lambda_spec_reg` đã có sẵn nhưng chưa được cộng vào loss. Cần phạt L2 norm của đầu ra mạng ASG (`spec_sparse`).

**Implementation (trong `train.py`):**

```python
loss = photometric_loss

# Thêm regularization
if opt.lambda_spec_reg > 0 and spec_sparse is not None:
    # Phạt độ lớn của màu do ASG sinh ra (trước khi đưa vào render)
    # L2 loss ép ASG chỉ phát huy tác dụng khi thực sự cần thiết, 
    # và có xu hướng đưa ASG về 0 ở vùng không có highlight
    spec_reg_loss = (spec_sparse ** 2).mean()
    loss = loss + opt.lambda_spec_reg * spec_reg_loss
```

**Chú ý Ablation:** Đổi default `lambda_spec_reg = 0.0` thay vì `0.01` hiện tại, và ablate riêng để xem liệu penalty này có làm giảm `only_asg` quá mức hay không.

---

## 3. Patch-based Specular LPIPS (Ưu tiên Thấp - Khó, nâng cao)

**Mục tiêu:** L1 loss có tính chất "làm mờ" (blurry). Để highlight sắc nét và chân thực, cần loss nhận thức (Perceptual Loss như LPIPS/VGG). Tuy nhiên, chạy LPIPS trên toàn ảnh độ phân giải cao rất tốn VRAM và chậm.

**Thiết kế:**
- Dùng `cam.ref_score` để tìm các tâm (centers) của vùng specular.
- Crop các patch nhỏ (ví dụ 64x64 hoặc 128x128) xung quanh các tâm này từ ảnh render và ảnh GT.
- Tính LPIPS loss riêng trên các patch này.

**Implementation (Cơ bản):**
- Đòi hỏi viết thêm hàm crop patch tensor dựa trên tọa độ center.
- Tích hợp model LPIPS vào training loop (cần lưu ý quản lý VRAM cẩn thận).

*Đề xuất: Chỉ đưa vào làm optional feature nếu Spec_PSNR đã cao nhưng ảnh nhìn bằng mắt thường (visual quality) vẫn bị mờ nhòe.*

---

## Lộ trình Ablation (Nhóm S - Supervision)

Sau khi nhóm R (Role Separation) đã ổn định, ta chạy nhóm S:
- **S0:** Baseline (Đã có R1). `lambda_spec_l1_weight=0.0`, `lambda_spec_reg=0.0`
- **S1:** Kích hoạt Specular-Weighted Loss (`lambda_spec_l1_weight=1.0` hoặc `2.0`). Kỳ vọng: `Spec_PSNR` tăng.
- **S2:** Kích hoạt Regularization (`lambda_spec_reg=0.01`). Kỳ vọng: `only_asg` sạch hơn, ít bị noise ở vùng diffuse, nhưng có thể `Spec_PSNR` giảm nhẹ. Cần tìm điểm cân bằng.
