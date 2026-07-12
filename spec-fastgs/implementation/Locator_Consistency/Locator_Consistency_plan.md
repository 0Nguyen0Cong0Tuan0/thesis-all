# Locator Consistency — Tại Sao R001–R045 Bế Tắc Ở 3/4 Trục, Và Việc Cần Làm Tiếp

> Tài liệu này là kết quả của quá trình đọc lại toàn bộ source code hiện tại (`train.py`,
> `gaussian_model.py`, `spec_utils.py`, `fast_utils.py`, `specular_model.py`,
> `gaussian_renderer/__init__.py`, `render.py`, `metrics.py`, `extract_reflection_prior.py`)
> và đối chiếu với 4 tài liệu thiết kế trong `implementation/` cùng 45 run thực nghiệm trong
> `run_note.md`. Mục tiêu: tìm ra giả thuyết có cơ sở code-level mạnh nhất để giải thích vì
> sao Geometric Coverage là trục duy nhất có gain thật, còn Representation Capacity /
> Supervision Signal / Normal Quality đều thất bại có hệ thống — và đề xuất kế hoạch code cụ
> thể để kiểm chứng.

---

## Bước 1 — Chi tiết ẩn phát hiện được khi đọc lại toàn bộ pipeline

### 1.1. `ASG_Residual_IoU` / `ASG_Energy_In_Residual` là proxy metric có thể bị "đánh lừa"

`metrics.py::computeAuxMetrics()`:

```python
residual_gray = residual_real.mean(dim=1, keepdim=True)   # clamp(GT - SH_only, min=0)
asg_gray = only_asg.mean(dim=1, keepdim=True)              # clamp(Full - SH_only, min=0)
spec_mask = residual_gray > 0.02
asg_mask  = asg_gray > 0.02
ASG_Residual_IoU = |asg_mask ∩ spec_mask| / |asg_mask ∪ spec_mask|
```

Đây là **occupancy IoU tuyệt đối ngưỡng 0.02/255**, không đo màu sắc/độ lớn có khớp hay
không, chỉ đo "ASG có active ở đây không" so với "SH có thiếu sót ở đây không". Hệ quả:

- Ngưỡng 0.02 trên ảnh [0,1] rất lỏng — bắt được cả anti-aliasing, texture SH bậc 3 chưa
  fit hết, sai lệch alignment nhỏ, không chỉ specular thật.
- IoU này **cực kỳ dễ tăng giả**: chỉ cần ASG active rộng hơn (dù màu sai) là `asg_mask`
  rộng ra, trùng nhiều hơn với `spec_mask` (vốn đã rộng do SH kém đi khi bị mask gradient).
- Đây giải thích chính xác pattern quan sát được trong R025/R026, R040, R042/R043: mọi lần
  ép SH nhường vai trò (hard/soft mask, residual supervision) đều làm `ASG_Residual_IoU`
  tăng, nhưng PSNR/SSIM/LPIPS/Spec_PSNR giảm — vì ASG "phủ rộng hơn" không đồng nghĩa với
  "tái tạo đúng specular".

**Kết luận quan trọng**: `ASG_Residual_IoU`/`ASG_Energy_In_Residual` không nên tiếp tục
dùng làm tín hiệu quyết định "role separation có hiệu quả hay không". May mắn là
`run_note.md` đã tự nhận ra điều này qua thực nghiệm (luôn ưu tiên PSNR/SSIM/LPIPS/Spec_PSNR
khi mâu thuẫn với IoU), nên **kết luận cuối của 45 run vẫn đáng tin** — nhưng bất kỳ
ablation Representation Capacity nào trong tương lai không nên dùng lại IoU làm tiêu chí
early-stop hoặc tiêu chí chọn hyperparameter.

### 1.2. Locator hiện tại trong `extract_reflection_prior.py` là bản CHƯA có top-hat fix

`extract_reflection_prior.py::tan_ikeuchi_score()` và `shafer_klinker_score()` là detector
đơn-pixel thuần túy (bright + desaturated / Imin+bright_floor), **không có bước lọc
morphological top-hat**. Không có file `tools/classical_specular_mask.py` nào tồn tại trong
repo này (đã `grep -r` toàn repo để xác nhận) — chỉ có một bản fallback nằm trong
`test_specular_algorithms_comparison.ipynb` (dùng `scipy.ndimage.grey_opening`):

```python
def shafer_score(img01, sat_thresh=0.25, val_thresh=0.75, tophat_radius=12, tophat_thresh=0.08):
    maxc = img01.max(axis=-1); minc = img01.min(axis=-1)
    V = maxc
    S = np.where(maxc > 1e-6, (maxc - minc) / (maxc + 1e-6), 0.0)
    raw_mask = (V > val_thresh) & (S < sat_thresh)
    opened = grey_opening(V, footprint=_disk_footprint(tophat_radius))
    tophat = np.clip(V - opened, 0.0, None)
    mask = raw_mask & (tophat > tophat_thresh)
    score = tophat * raw_mask
    ...
```

Ý nghĩa vật lý: top-hat = "độ nổi bật cục bộ so với vùng lân cận cùng kích thước
`tophat_radius`". Nó loại bỏ chính xác false positive mà `tan_ikeuchi_score`/
`shafer_klinker_score` hiện tại mắc phải — nền trắng phẳng, vật liệu diffuse sáng lớn (đúng
là điều `Pipeline_analysis.md` mục Trục 1 đã cảnh báo: *"Nền trắng, vật liệu diffuse sáng,
vùng overexposed đều bị nhầm là specular"*). Notebook này (trong đúng repo `spec-fastgs`
đang xét) xác nhận bằng số: locator hiện tại (không top-hat) flag ~15% mean/25% max pixel
trên `counter`, trong khi bản top-hat chỉ ~1.4-1.5% mean — một khác biệt rất lớn.

**Đây là bằng chứng code-level trực tiếp, không phải suy đoán**: repo `spec-fastgs` (bản
đang phân tích, "colleague's tree") vẫn dùng locator thế hệ cũ, chưa merge fix top-hat mà
chính notebook trong repo này đã tự phát hiện và tự viết ra. Nói cách khác: **công cụ sửa
lỗi đã tồn tại sẵn trong repo (trong notebook), nhưng chưa được port vào
`extract_reflection_prior.py`/`train.py` pipeline production.**

### 1.3. Locator lỏng lẻo ảnh hưởng đến CẢ 3 trục đang thất bại, không chỉ Trục 1

Đây là insight nối các mảnh lại với nhau:

- **Densification (`cam.ref_score`)**: dùng locator lỏng → force-densify cả vùng diffuse
  sáng, lãng phí budget Gaussian đúng như `Pipeline_analysis.md` đã cảnh báo.
- **`use_sh_spec_mask` (Representation Capacity)**: `spec_metric_map = (cam.ref_score >
  threshold)` — nếu `ref_score` sai (flag cả vùng diffuse sáng), SH bị chặn gradient ở
  ĐÚNG những vùng nó cần để giữ diffuse tốt → giải thích trực tiếp vì sao hard mask (R026)
  làm PSNR giảm mạnh dù IoU tăng.
- **`use_asg_residual_supervision`**: `ref_mask = (cam.ref_score > threshold)` — cùng vấn đề,
  dạy ASG "giải thích residual" tại các vùng không phải specular thật → ASG học sai mục
  tiêu, nhất quán với R042/R043 (loss active nhưng không cải thiện reconstruction).
- **`lambda_spec_l1_weight`**: cũng dùng `cam.ref_score` làm pixel weight → weighted loss
  tăng gradient tại đúng vùng bị flag sai, giải thích vì sao R031/R041 không ổn định
  (R031 dùng cùng lúc với hard mask ở ngưỡng khác, R041 lại âm).
- **`normal_smooth_use_ref_mask`**: cùng cơ chế — R045 (dùng ref_mask) còn tệ hơn R044
  (không dùng ref_mask), phù hợp với giả thuyết locator nhiễu lan truyền tác động xấu sang
  bất kỳ cơ chế nào dựa vào nó.

**Đây là phát hiện quan trọng nhất của bước 1**: `cam.ref_score` là một **input signal dùng
chung** cho toàn bộ 4 trục (Geometry Coverage, Representation Capacity, Supervision Signal,
Normal Quality-ref-mask). Chất lượng của nó là **common-mode bottleneck**. Nếu locator sai,
mọi trục dựa vào nó (3/4 trục) đều bị nhiễu bởi cùng một nguồn — độc lập với thiết kế
cơ chế downstream có đúng hay không.

### 1.4. Sparse ASG evaluation dùng visibility của camera TRƯỚC, `full_asg_interval` không bao giờ được ablate

`train.py` dòng ~277-307: nếu `prev_vis_mask` khớp kích thước, ASG chỉ evaluate trên
`vis_indices = prev_vis_mask.nonzero()` — visibility của **camera ngẫu nhiên ở iteration
trước**, không phải camera hiện tại. `full_asg_interval` (refresh định kỳ toàn bộ ASG) có
default `0` = tắt hoàn toàn.

Grep toàn bộ `run_note.md` xác nhận: **`full_asg_interval=0` xuất hiện ở MỌI run R001-R045
— chưa từng bị ablate**, dù `Representation_capacity_solutions.md` mục R4 đã đề xuất chạy
`FULL_ASG_INTERVAL=1000/3000` để kiểm tra giả thuyết "Gaussian visible ở camera hiện tại
nhưng absent ở camera trước sẽ không nhận gradient ASG ở đúng iteration cần". Cameras được
sample ngẫu nhiên mỗi iteration (`randint`), nên với dataset nhiều camera (`counter` ~100
train views ở `images_8`), xác suất 2 camera liên tiếp có tập Gaussian-visible trùng nhau
cao là thấp — đặc biệt với Gaussian ở rìa/góc nhìn hẹp, chính là loại Gaussian có khả năng
mang thông tin specular anisotropic mạnh nhất (highlight phụ thuộc góc nhìn hẹp).

Đây là hypothesis **chưa từng bị bác bỏ bằng thực nghiệm** — khác với 3 trục kia đã có
3-4 negative run rõ ràng. Rủi ro code = 0 (flag đã tồn tại, đã wire đầy đủ), chỉ cần đổi
tham số chạy thử.

### 1.5. `ASGRender.mlp` không zero-init (khác `_normal_delta`)

`spec_utils.py::ASGRender.__init__`: chỉ có
`torch.nn.init.constant_(self.mlp[-1].bias, 0)` — bias lớp cuối = 0, nhưng **weight** vẫn
random init theo PyTorch default (Kaiming/uniform). Nghĩa là ngay tại `specular_start_iter`,
`spec_sparse` KHÔNG bắt đầu từ 0 mà là một hàm ngẫu nhiên nhỏ nhưng khác 0 của
`(asg_feature, viewdir, normal)`. So sánh với `_normal_delta` (`gaussian_model.py`):
khởi tạo `torch.zeros(...)`, có L2 reg, có norm-clamp — được thiết kế cẩn thận như một
"correction bắt đầu từ 0". ASG thì không có cơ chế tương tự (`lambda_spec_reg` tồn tại
nhưng default `0.0`, không neo ASG về 0 ở vùng non-specular).

Đây không phải "bug" theo nghĩa cổ điển — model vẫn học được vì loss sẽ ép ASG≈0 ở vùng
diffuse qua thời gian train — nhưng nó là một nguồn nhiễu khởi đầu nhỏ, dễ sửa, chưa từng
được thử nghiệm tách biệt.

---

## Bước 2 — Tự tranh biện từng giả thuyết (nhiều vòng)

### Vòng 1: "Có nên tiếp tục đầu tư vào Normal Quality (normal_delta) không?"

**Lập luận ủng hộ tiếp tục**: `_normal_delta` học được thật (norm > 0, không collapse),
cơ chế đúng về mặt lý thuyết (GaussianShader/Relightable 3DGS đều dùng ý tưởng tương tự),
và ASG thực sự dùng normal để tính reflection direction nên về logic normal quality NÊN
quan trọng.

**Lập luận phản đối**: R030 (toaster), R036 (counter, `real_use_reflection_dir=True`), R044,
R045 — **4 run độc lập, bao gồm cả run "sạch" đã tắt hết Representation/Supervision để cô
lập biến số** — đều cho kết quả tiêu cực trên fidelity. R045 (`normal_smooth_use_ref_mask=True`)
còn tệ hơn R044, và mục 1.3 ở trên giải thích tại sao: `normal_smooth_use_ref_mask` dùng
`cam.ref_score`/`sh_spec_grad_mask` — locator nhiễu lan sang normal smoothness, ép Gaussian
smooth normal ở vùng bị flag sai (diffuse sáng) thay vì vùng specular thật.

**Phán quyết**: Normal Quality (dạng normal-delta + smoothness hiện tại) nên **tạm dừng
đầu tư thêm engineering mới**, nhưng KHÔNG kết luận "root cause B is unfixable" — vì mọi run
Normal Quality đến nay đều bị nhiễu bởi locator sai (mục 1.3) khi dùng `ref_mask`, và ngay
cả khi không dùng ref_mask (R044), locator vẫn gián tiếp ảnh hưởng qua `use_ref_score`
densification (Gaussian coverage sai chỗ → normal proxy noisy hơn). **Kết luận đúng đắn ở
đây là: chưa đủ bằng chứng sạch để phán quyết Normal Quality đã chết — cần retest SAU khi
locator được sửa, không phải bây giờ.** Ưu tiên thấp cho vòng tiếp theo, nhưng giữ nguyên
code (đã có đủ flag, rủi ro giữ = 0).

### Vòng 2: "Locator fix có thực sự là đòn bẩy, hay chỉ là another confound để đổ lỗi?"

**Phản biện tự đặt ra**: Liệu đây có phải nguỵ biện kiểu "cứ đổ lỗi cho input data, không
bao giờ đổ lỗi cho thiết kế thuật toán"? Cần kiểm tra xem locator fix có thực sự đổi được
kết luận, hay chỉ là một biến số nữa cộng thêm vào danh sách đã dài.

**Phản biện lại**: Bằng chứng cho thấy đây không phải suy đoán chung chung:
1. Cơ chế toán học rõ ràng: `V > val_thresh & S < sat_thresh` không có ràng buộc
   spatial-compactness, nên **về mặt toán học chắc chắn** flag nhầm vùng diffuse sáng lớn
   phẳng — đây là positive theo định nghĩa, không phải nhiễu ngẫu nhiên có thể trung bình
   hoá đi.
2. Ảnh hưởng có hướng nhất quán, không ngẫu nhiên: mọi lần dùng `ref_mask`/`ref_score` cho
   một cơ chế penalize/reweight, kết quả xấu hơn phiên bản không dùng ref_mask trong CÙNG
   một cặp so sánh có kiểm soát (R044 vs R045). Đây là dấu hiệu bias hệ thống, không phải
   variance.
3. Notebook trong chính repo đã đo bằng số (~15% mean vs ~1.4% mean flagged) — không phải
   suy luận lý thuyết suông.

**Phán quyết**: Giả thuyết locator có positive evidence đủ mạnh để đầu tư một patch nhỏ,
rủi ro thấp (measurement/input-layer fix, không đổi loss/optimizer/kiến trúc), **nhưng phải
tự đặt điều kiện dừng rõ ràng**: nếu sau khi fix locator mà re-run R1-style role-separation
ablation vẫn cho kết quả âm, phải chấp nhận Representation Capacity thật sự không phải
bottleneck ở quy mô Gaussian-count hiện tại (ủng hộ giả thuyết 1 trong cuộc thảo luận trước:
"model chưa đủ Gaussian, chưa cần role separation").

### Vòng 3: "Fix locator có nên áp dụng ngay vào production default, hay chỉ opt-in để đo?"

**Lập luận cho áp dụng ngay**: Nếu locator sai rõ ràng, tại sao không sửa default luôn?

**Lập luận cho opt-in trước**: Toàn bộ 45 run hiện tại của colleague dùng locator cũ. Nếu
đổi default ngay, MỌI so sánh trong tương lai với các run cũ sẽ lẫn 2 biến số cùng lúc
(locator mới + bất kỳ thay đổi khác). Đúng tinh thần mà `Geometric_Coverage_feedback.md` đã
nhấn mạnh nhiều lần: *"implement nên có đủ flag... nhưng ablation đầu tiên chỉ nên bật hành
vi cụ thể"*. Ngoài ra fix chưa được validate trên `counter`/`toaster` cụ thể của colleague
(chỉ có số liệu từ notebook, chưa chắc cùng threshold tối ưu).

**Phán quyết**: Thêm như **method thứ 3 opt-in** (`ref_prior_method=shafer_tophat`), giữ
`tan`/`shafer` cũ nguyên vẹn làm baseline so sánh trực tiếp. Đây đúng tinh thần toàn bộ
codebase hiện tại (mọi thứ đều opt-in, default giữ nguyên baseline).

### Vòng 4: "Có nên đồng thời sửa `metrics.py`'s `spec_mask` không?"

**Lập luận cho**: Nếu không sửa, `Spec_PSNR`/`ASG_Residual_IoU` vẫn tiếp tục là proxy nhiễu,
ablation tương lai vẫn khó đọc.

**Lập luận cẩn trọng**: Sửa `metrics.py` sẽ làm **mọi con số Spec_PSNR/IoU của 45 run cũ
không còn so sánh trực tiếp được với run mới** (giống vấn đề vòng 3). Đây là thay đổi ở
tầng đo lường (evaluation), không phải training, nên rủi ro kỹ thuật thấp nhưng rủi ro
"gãy chuỗi lịch sử so sánh" cao.

**Phán quyết**: Làm, nhưng **thêm cột mới** (`Spec_PSNR_v2`, `ASG_Residual_IoU_v2` dùng
mask từ locator top-hat đã lưu sẵn trên đĩa) **song song** với cột cũ, không thay thế.
`PSNR`/`SSIM`/`LPIPS` toàn ảnh (đã đáng tin, không đổi) vẫn là trọng tài chính như
`run_note.md` đã làm đúng. Việc này thuần tuý bổ sung, không phá vỡ khả năng so sánh với
lịch sử.

### Vòng 5: "Ưu tiên nào trước — fix locator, hay ablate `full_asg_interval` trước (rẻ hơn)?"

`full_asg_interval` ablation không cần viết code mới (flag đã có), rủi ro = 0, chi phí = một
vài run 15-20 phút. Nên chạy **song song, không phụ thuộc nhau** — hai giả thuyết độc lập,
không cần chờ nhau.

**Phán quyết cuối bước 2**: 2 hướng ưu tiên cao nhất, độc lập, chạy song song:
1. **Locator top-hat fix** (đòn bẩy tiềm năng lớn nhất, ảnh hưởng cross-cutting cả 3 trục).
2. **`full_asg_interval` ablation** (rẻ nhất, zero code risk, giả thuyết chưa từng bị bác bỏ).

Sau khi có kết quả 2 hướng này, **mới quay lại retest role separation (R1-style)** với
locator đã sửa — đây là bước 3 có điều kiện, không làm ngay.

---

## Bước 3 — Kế hoạch implementation chi tiết

### Phase 0 (bắt buộc trước, rủi ro = 0): Port locator top-hat vào `extract_reflection_prior.py`

**File**: `extract_reflection_prior.py`

Thêm hàm mới, không sửa 2 hàm cũ:

```python
from scipy.ndimage import grey_opening

def _disk_footprint(radius):
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return (x ** 2 + y ** 2 <= radius ** 2)

def shafer_tophat_score(img01, sat_thresh=0.25, val_thresh=0.75,
                         tophat_radius=12, tophat_thresh=0.08):
    """Shafer/Klinker dichromatic threshold gated by a morphological white
    top-hat filter, rejecting flat bright-diffuse regions/backgrounds that
    the plain threshold in shafer_klinker_score() false-positives on.
    Ported from test_specular_algorithms_comparison.ipynb (validated on
    counter/teapot: raw ~15% mean flagged -> top-hat ~1.4% mean flagged).
    """
    maxc = img01.max(axis=-1)
    minc = img01.min(axis=-1)
    V = maxc
    S = np.where(maxc > 1e-6, (maxc - minc) / (maxc + 1e-6), 0.0)
    raw_mask = (V > val_thresh) & (S < sat_thresh)
    opened = grey_opening(V, footprint=_disk_footprint(tophat_radius))
    tophat = np.clip(V - opened, 0.0, None)
    mask = raw_mask & (tophat > tophat_thresh)
    score = tophat * raw_mask
    if score.max() > 0:
        score = score / score.max()
    return np.where(mask, score, 0.0)
```

Thêm `"shafer_tophat"` vào `choices` của `--ref_prior_method`, thêm 4 argument mới
**độc lập** (`tophat_val_thresh=0.75`, `tophat_sat_thresh=0.25`, `tophat_radius=12`,
`tophat_thresh=0.08`) — **không** tái dùng `sk_intensity`/`sk_saturation`, vì hai tham số đó
mang default cũ của `shafer_klinker_score` (0.7/0.2), khác với default đã validate của
`shafer_tophat_score` trong notebook (0.75/0.25); dùng chung sẽ âm thầm đổi ngưỡng so với
bản đã kiểm chứng. Default giữ nguyên hành vi cũ vì `ref_prior_method` mặc định vẫn là
`"tan"`.

**File**: `arguments/__init__.py` — thêm `self.tophat_val_thresh = 0.75`,
`self.tophat_sat_thresh = 0.25`, `self.tophat_radius = 12`,
`self.tophat_thresh = 0.08` (chỉ dùng khi `ref_prior_method="shafer_tophat"`).

**Không đổi**: `train.py` load logic (`cam.ref_score` từ PNG) — không cần sửa gì vì locator
mới cũng xuất ra `.png` cùng format, cùng thư mục `reflection_prior/`.

**Trạng thái implementation**: ĐÃ CODE xong (2026-07-12) —
`extract_reflection_prior.py` (hàm `shafer_tophat_score` + dispatch + argparse),
`arguments/__init__.py` (4 tham số mới, log-only, không đổi default training),
`train.py` (log 4 tham số mới vào `train_info.json`), `environment.yml` (thêm `scipy`).
`python -m py_compile` sạch trên cả 3 file Python.

**GATE đã chạy — eyeball trước khi train (2026-07-12)**: Không thể import
`extract_reflection_prior.py` trực tiếp trong môi trường dev (kéo theo
`scene/gaussian_model.py` → `simple_knn._C`, extension CUDA chưa build ở máy dev). Đã
validate bằng cách chạy y hệt 3 hàm (`tan_ikeuchi_score`, `shafer_klinker_score`,
`shafer_tophat_score`, copy verbatim, không sửa logic) trực tiếp trên 6 ảnh JPG gốc của
`dataset/mipnerf360/counter/images` (lấy đều theo thứ tự, không chọn lọc ảnh đẹp), cả ở
downscale ×4 và full-resolution:

| | tan (default hiện tại) | shafer (plain) | shafer_tophat (mới) |
|---|---:|---:|---:|
| Mean flagged % (6 ảnh, ×4 downscale) | 16.22% | 3.42% | 2.47% |
| Full-res, 3 ảnh mẫu | 11.67% / 20.72% / 15.26% | 2.45% / 3.62% / 5.31% | 1.20% / 2.68% / 0.88% |

**Phát hiện quan trọng cần điều chỉnh lại kỳ vọng ban đầu**: xem bằng mắt (RGB vs mask) xác
nhận `tan` — locator mặc định dùng trong CẢ 45 run — cực kỳ lỏng lẻo, gần như flag mọi vùng
không đen tuyền trên toàn cảnh (tường, sàn, thảm, hộp trứng, găng lò) chứ không riêng gì
specular. `shafer` (plain, chưa top-hat) trên dataset `counter` cụ thể này đã tương đối
"sạch" sẵn (2-5%), không lỏng như con số ~15% mà notebook đo trên `teapot` (nền trắng đồng
nhất, khác hẳn cảnh bếp lộn xộn của `counter`). `shafer_tophat` cắt giảm thêm so với
`shafer` plain, nhưng ở mức **khiêm tốn hơn nhiều** so với kỳ vọng ban đầu (~15%→~1.4%) —
trên `counter`, hiệu ứng chính quan sát được là top-hat loại bỏ đúng như thiết kế: dải
tường/cửa sổ overexposed phẳng ở rìa ảnh, trong khi vẫn giữ lại highlight thật (viền bình
nước, đốm đá hoa cương, mép khay nướng, nắp bình giữ nhiệt).

**Kết luận điều chỉnh cho ablation**: đòn bẩy lớn nhất không nằm ở "top-hat vs plain
Shafer" (chênh lệch khiêm tốn trên `counter`), mà nằm ở **"tan vs shafer-family"** — đây
mới là khác biệt biên độ lớn (16-20% → 2-5%) và là thay đổi chưa từng được thực nghiệm
trong 45 run (`ref_prior_method=tan` xuất hiện ở mọi run; `shafer` chỉ vài run sớm R020 rồi
bị bỏ quên khi nhóm chuyển sang `tan+adaptive` cho các best run R024/R025/R038). Vì vậy
ablation cần tách rõ 2 biến thay vì gộp chung:

**Ablation đề xuất (đã sửa từ bản gốc)**:
- **R046a** = R038 với `REF_PRIOR_METHOD=shafer` (plain, đã có sẵn trong code, chưa dùng
  cùng adaptive prior + budget mới) — đo tác động của việc rời khỏi `tan`.
- **R046b** = R038 với `REF_PRIOR_METHOD=shafer_tophat` — đo thêm phần top-hat đóng góp so
  với R046a.

So sánh 3-way (R038 tan / R046a shafer / R046b shafer_tophat) cho phép quy kết đúng nguồn
gốc của bất kỳ cải thiện nào, thay vì gộp 2 thay đổi (rời `tan` + thêm top-hat) vào một run
duy nhất.

**Cập nhật (2026-07-12) — đã wire vào Kaggle notebook, phát hiện + sửa 1 bug hạ tầng trước
khi wire**: `run_spec-fastgs_big.sh` trước đây gán MỌI biến (`SCENE`, `ASG_DEGREE`,
`CUDA_VISIBLE_DEVICES`, `REF_PRIOR_METHOD`, ...) bằng cú pháp gán cứng (`VAR=value`), không
phải `VAR=${VAR:-value}`. Hệ quả: mọi override từ notebook qua env var (kể cả cell 14 sweep
ASG_DEGREE=32/48 có sẵn từ trước) bị **âm thầm ghi đè và bỏ qua** — cả 2 leg "song song" của
sweep ASG thực ra đều chạy trên GPU 0 (vì `export CUDA_VISIBLE_DEVICES=0` cứng ở đầu file),
và `OUTPUT_SUFFIX` chưa từng được dùng ở đâu trong script nên 2 leg còn race-write vào cùng
`output/counter`. Đã sửa toàn bộ sang `${VAR:-default}` + thêm `OUTPUT_SUFFIX` vào đường dẫn
output (`${OUTPUT_ROOT}/${SCENE}${OUTPUT_SUFFIX}`) — verify bằng cách chạy riêng phần
gán-biến+echo của script với override giả lập, xác nhận `CUDA_VISIBLE_DEVICES`, `ASG_DEGREE`,
`REF_PRIOR_METHOD`, `OUTPUT_SUFFIX` đều nhận đúng giá trị override. Điều này sửa luôn cho cả
sweep ASG_DEGREE cũ, không chỉ locator experiment mới.

`hcmus-spec-fastgs-thesis.ipynb` được thêm 4 cell mới sau cell archive cũ (không sửa cell
0-16 hiện có, giữ nguyên pipeline ASG sweep cũ chạy được):
1. Cell cấu hình `LOCATOR_METHODS = ["shafer", "shafer_tophat"]`.
2. Cell `%%bash` chạy R046a (`shafer`) rồi R046b (`shafer_tophat`) **tuần tự** (không song
   song, vì 2 leg dùng chung `reflection_prior/` của cùng 1 scene — chạy song song sẽ ghi đè
   nhau; khác với sweep ASG_DEGREE vốn share chung 1 prior `tan` hợp lệ). Mỗi leg tự
   `EXTRACT_REF_PRIOR=True` + `BACKUP_REF_PRIOR=True` (script tự backup prior cũ trước khi
   ghi đè), `OUTPUT_SUFFIX=_${METHOD}_ref` để tách thư mục output.
3. Cell so sánh nhanh: đọc `results_grouped.json` + `train_info.json` của cả 2 output dir,
   in PSNR/SSIM/LPIPS + Spec_PSNR/ASG_Residual_IoU/ASG_Energy_In_Residual song song — đây là
   GATE nhanh, không phải phân tích cuối; vẫn cần đối chiếu tay với số R038 trong
   `run_note.md`.
4. Cell archive output thành `.zip`, theo đúng pattern cell 16 cũ.

`python -m py_compile` + `bash -n` sạch trên script đã sửa; `nbformat.validate()` sạch trên
notebook đã thêm cell. Chưa chạy thật trên Kaggle (cần GPU) — đây là bước tiếp theo do người
dùng thực hiện.

### Phase 0b (song song, độc lập, rủi ro = 0): `full_asg_interval` ablation

Không cần sửa code. Chạy lại **R038** (baseline hiện tại) với
`FULL_ASG_INTERVAL=1000` và `FULL_ASG_INTERVAL=3000` (R048, R049) — đúng đề xuất có sẵn
trong `Representation_capacity_solutions.md` mục R4, chưa từng được thực thi. Theo dõi
`avg_asg_eval_count` (đã log sẵn trong `train_info.json`) để xác nhận full refresh có làm
tăng chi phí đáng kể không, và theo dõi PSNR/SSIM/LPIPS/Spec_PSNR/ASG_Residual_IoU so với
R038.

### Phase 1 (chỉ chạy nếu Phase 0 hoặc 0b cho tín hiệu dương): mở rộng `metrics.py`

**File**: `metrics.py`

Thêm tuỳ chọn nạp mask locator top-hat đã lưu (từ `reflection_prior/` nếu
`ref_prior_method=shafer_tophat` được dùng ở lần train tương ứng) làm `spec_mask` thay thế,
tính thêm `Spec_PSNR_v2`/`ASG_Residual_IoU_v2`/`ASG_Energy_In_Residual_v2` **song song** với
bản cũ (không xoá cột cũ). Việc này chỉ có ý nghĩa sau khi Phase 0 xác nhận locator mới hợp
lý trên `counter`/`toaster`.

### Phase 2 (điều kiện): retest role separation với locator đã sửa

Chỉ thực hiện nếu Phase 0 cho thấy locator mới thu hẹp vùng flag rõ rệt (giống notebook:
~15%→~1.4%) VÀ không làm giảm coverage specular thật (kiểm tra bằng mắt bắt buộc trước).

Lặp lại đúng bộ ablation R025/R026 (hard mask) và R040 (soft mask) nhưng với
`REF_PRIOR_METHOD=shafer_tophat` thay vì `tan`. Nếu `use_sh_spec_mask=True` vẫn làm giảm
PSNR/SSIM/LPIPS dù locator đã sạch hơn nhiều, đó là bằng chứng mạnh hơn nhiều để **kết luận
dứt khoát** Representation Capacity không phải bottleneck ở coverage hiện tại (ủng hộ hướng
"dồn toàn lực vào Geometric Coverage", đúng như xu hướng 45 run đã cho thấy).

### Phase 3 (rủi ro thấp, có thể làm song song, không phụ thuộc Phase 0-2): ASG zero-init nhẹ

**File**: `utils/spec_utils.py`, class `ASGRender`/`ASGRenderReal.__init__`

Thêm khởi tạo nhỏ cho layer cuối tương tự cách `_normal_delta` được thiết kế cẩn trọng:

```python
# Sau torch.nn.init.constant_(self.mlp[-1].bias, 0):
torch.nn.init.zeros_(self.mlp[-1].weight)
```

Zero-init cả weight lẫn bias của layer cuối cùng → `spec_sparse` = 0 chính xác tại
`specular_start_iter`, loại bỏ nhiễu khởi đầu ngẫu nhiên. Đây là thay đổi 1 dòng, độc lập
hoàn toàn với locator, có thể ablate riêng (R050: R038 + zero-init) hoặc gộp vào bất kỳ run
nào khác vì rủi ro gây hại gần như không có (layer cuối vẫn học bình thường sau vài
iteration, giống cách các paper NeRF/3DGS residual-branch thường zero-init output layer).

### Không làm ngay (theo phán quyết Vòng 1)

- Không tiếp tục tune quanh `normal_delta`/`lambda_normal_smooth` cho tới khi Phase 0-2 xong.
- Không tiếp tục tune `lambda_spec_l1_weight`/`lambda_spec_reg` cho tới khi locator sạch hơn
  (vì cả hai đều dùng `cam.ref_score` làm input).
- Không tăng `ASG_DEGREE`/kiến trúc MLP thêm nữa (đã đủ bằng chứng ASG capacity không phải
  bottleneck khi role separation chưa chứng minh được lợi ích).

### Bảng tổng hợp rủi ro / lợi ích

| Phase | Thay đổi | File | Rủi ro | Có thể ablate độc lập? | Phụ thuộc |
|---|---|---|---|---|---|
| 0 | Locator top-hat mới (opt-in) | `extract_reflection_prior.py`, `arguments/__init__.py` | Rất thấp (offline, opt-in) | Có | Không |
| 0b | `full_asg_interval` ablation | Không đổi code | Không | Có | Không |
| 1 | Metrics v2 mask song song | `metrics.py` | Thấp (evaluation-only) | Có | Phase 0 tích cực |
| 2 | Retest role separation | `run_spec-fastgs_big.sh` config | Trung bình (thời gian) | Có | Phase 0 tích cực |
| 3 | ASG zero-init | `utils/spec_utils.py` | Rất thấp | Có | Không |

### Thứ tự chạy đề xuất (chỉ tốn ~6 run × ~17-20 phút để có tín hiệu quyết định)

1. R046a = R038 + `shafer` locator (rời khỏi `tan`, chưa top-hat) — đo lever lớn nhất.
2. R046b = R038 + `shafer_tophat` locator (thêm top-hat so với R046a) — đo phần top-hat.
3. R048 = R038 + `full_asg_interval=1000`.
4. R049 = R038 + `full_asg_interval=3000`.
5. R050 = R038 + ASG zero-init (Phase 3, độc lập).
6. Nếu R046a/R046b dương rõ: R051 = R046b + `use_sh_spec_mask` (Phase 2, hard mask retest).

Mỗi run chỉ đổi đúng 1 biến so với R038 — giữ nguyên kỷ luật ablation một-biến-một-lần mà
`run_note.md` đã áp dụng xuyên suốt 45 run trước.
