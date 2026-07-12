Tôi cần đọc file implementation plan để so sánh với feedback.

Viewed Geometric%20Coverage_implementation_plan.md:1-356

Tôi đã đọc cả hai file. Bây giờ tôi sẽ review kỹ từng phần implementation so với feedback.

---

## Review: Implementation Plan Đã Đúng Chưa?

### Tổng thể — Cấu trúc tư duy đúng ✅

Logic tổng thể B1→B2→A, lý do chẩn đoán vấn đề, pseudocode đều coherent. Không có chỗ nào sai về lý.

---

### Vấn đề 1 — Residual Dùng Full Render Thay Vì SH-Only

**Dòng 76 trong plan:**
```python
render_img = render_fastgs(cam, gaussians, pipe, background, opt.mult)["render"]
```

Đây là full render — có cả `mlp_color=None` nhưng **không truyền gì** nên render SH-only trong context này. Tuy nhiên **bạn không truyền `mlp_color` vào** nên thực ra đang render SH-only đúng rồi — nhưng cần làm rõ **tại sao** trong comment, vì người đọc code sau có thể không hiểu đây là chủ đích:

```python
# Render SH-only (mlp_color=None) — nhất quán với compute_gaussian_score_fastgs()
# Residual từ SH-only phản ánh đúng phần ASG/specular model chưa giải thích được
render_img = render_fastgs(cam, gaussians, pipe, background, opt.mult)["render"]
#                                                                mlp_color không truyền → None
```

Feedback (dòng 71) đặc biệt nhấn mạnh điểm này: cần **nhất quán có chủ đích** với `compute_gaussian_score_fastgs()`, không phải vô tình.

---

### Vấn đề 2 — Thiếu `min_strength` Clamp Cho B2

**Dòng 226-227 trong plan:**
```python
ref_score_strength = (1.0 - ratio) ** args.refscore_decay_power
use_ref_score = (ref_score_strength > 0.05)
```

Vấn đề: nếu `decay_power = 2` và `ratio = 0.85` thì:
```
strength = (1 - 0.85)^2 = 0.15^2 = 0.0225
→ use_ref_score = False  (vì 0.0225 < 0.05)
→ ref_score tắt khi còn 15% budget
```

Nghĩa là ref_score bị tắt **trước khi budget đầy** — ngược với mục tiêu. Cần thêm:

```python
# Sau khi tính strength:
ref_score_strength = max(ref_score_strength, getattr(args, 'refscore_min_strength', 0.1))
use_ref_score = True  # luôn True khi n_current < n_budget, strength kiểm soát threshold
```

Và bỏ `use_ref_score = (ref_score_strength > 0.05)` đi — thay vào đó để B2 chỉ điều chỉnh `dynamic_threshold`, không tắt hoàn toàn trước budget.

---

### Vấn đề 3 — `ref_score_static` Không Được Clone Đúng

**Dòng 310-311 trong plan:**
```python
cam.ref_score = ref_tensor
cam.ref_score_static = ref_tensor   # ← THIẾU .clone()
```

Nhưng trong section "Thay đổi cần thiết" ở trên (dòng 105-106) thì **có** `.clone()`:
```python
cam.ref_score_static = ref_tensor.clone()  # ✅
```

Hai chỗ trong cùng file không nhất quán. Phần "Tổng Hợp Thay Đổi" (dòng 311) **thiếu `.clone()`** — nếu implement theo phần tổng hợp này, `ref_score` và `ref_score_static` sẽ trỏ vào **cùng một tensor**, khi EMA update `cam.ref_score` thì `ref_score_static` cũng thay đổi theo → phá vỡ toàn bộ cơ chế intersection.

**Phải sửa dòng 311 thành:**
```python
cam.ref_score_static = ref_tensor.clone()  # BẮT BUỘC: tách biệt memory
```

---

### Vấn đề 4 — Thiếu `refscore_min_strength` Trong `arguments/__init__.py`

**Dòng 279-295 trong plan** liệt kê các hyperparameters mới, nhưng thiếu `refscore_min_strength` — cần thêm:

```python
self.refscore_min_strength = 0.1  # Sàn tối thiểu của strength, tránh tắt sớm
```

---

### Vấn đề 5 — Chi Phí Render Adaptive Prior Chưa Được Ước Lượng

Feedback (dòng 77) nhận xét "tốn thêm thời gian" nhưng không có số cụ thể. Trong bảng rủi ro plan (dòng 132):

> *"Chi phí thêm: render toàn bộ train set mỗi 3K iter | Thấp | Chỉ render với no_grad, không backward"*

Đánh giá "Thấp" mà không có con số cụ thể là thiếu thuyết phục. Nên thêm ước lượng:

```
counter/images_8: ~100 train cameras × ~0.05s/render ≈ 5s mỗi lần update
Với adaptive_prior_interval=3000, densify_until_iter=15K:
  → Cập nhật tại iter: 5K, 8K, 11K, 14K = 4 lần
  → Tổng overhead: ~20s / 15 phút train ≈ 2%
```

---

## Tóm Tắt Các Chỗ Cần Sửa

| # | Vị trí | Vấn đề | Mức độ |
|---|--------|---------|--------|
| 1 | Dòng 76 (render trong adaptive prior) | Thiếu comment giải thích dùng SH-only có chủ đích | ⚠️ Minor — logic đúng nhưng thiếu rõ ràng |
| 2 | Dòng 226-227 (B2 strength threshold) | `strength < 0.05` tắt ref_score trước khi budget đầy | 🔴 **Cần sửa** — sai logic |
| 3 | Dòng 311 (ref_score_static) | Thiếu `.clone()` → static và adaptive trỏ cùng tensor | 🔴 **Cần sửa** — bug nghiêm trọng |
| 4 | Dòng 279-295 (arguments) | Thiếu `refscore_min_strength` hyperparameter | ⚠️ Minor — cần thêm để B2 hoàn chỉnh |
| 5 | Dòng 132 (bảng rủi ro) | Chi phí adaptive prior đánh giá "Thấp" mà không có số | ⚠️ Minor — nên thêm ước lượng |

> [!CAUTION]
> **Vấn đề 3 là bug nghiêm trọng nhất** — nếu không `.clone()`, mọi cập nhật EMA sẽ làm hỏng `ref_score_static`, phá vỡ hoàn toàn cơ chế intersection. Phải sửa trước khi implement.