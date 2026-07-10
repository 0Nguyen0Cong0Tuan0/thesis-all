# Kế Hoạch Ablation Cho Spec-FastGS

File này liệt kê các thí nghiệm ablation cần chạy để đánh giá liệu ASG có cải thiện tái tạo vùng specular trong FastGS hay không, và từng thành phần ảnh hưởng thế nào đến chất lượng, tốc độ, VRAM, số Gaussian, cũng như ảnh visualize `only_asg`.

## Mục Tiêu

1. Đo chất lượng tái tạo toàn ảnh bằng PSNR, SSIM, LPIPS.
2. Đo riêng chất lượng tại vùng specular, không chỉ dựa vào metrics toàn ảnh.
3. Kiểm tra specular có thật sự được ASG biểu diễn hay bị SH hấp thụ.
4. Định lượng trade-off giữa metrics, thời gian train, VRAM và số Gaussian cuối.
5. Tách ảnh hưởng thật của method khỏi randomness và đặc thù từng dataset.

## Output, Metrics Cần Giữ Lại Và Ý Nghĩa

Với mỗi run, cần giữ nguyên cả folder output sau train/render/metrics. Tối thiểu phải có các file và folder sau:

| Output | Nơi đọc | Dùng để trả lời câu hỏi |
| --- | --- | --- |
| `train_info.json` | `<model_path>/train_info.json` | Run này dùng config nào, có bật `use_ref_score` không, `asg_degree`, `num_score_cameras`, `specular_start_iter`, lịch `f_rest`, thời gian train và các thông tin metadata khác. |
| `results.json` | `<model_path>/results.json` | Metrics chính toàn ảnh, giữ format cũ để so sánh nhanh hoặc dùng lại script cũ. |
| `per_view.json` | `<model_path>/per_view.json` | Metrics chính theo từng view; dùng để phát hiện view nào tụt mạnh thay vì chỉ nhìn trung bình. |
| `results_grouped.json` | `<model_path>/results_grouped.json` | File chính nên đọc cho ablation mới; tách rõ `main_metrics` và `aux_metrics`. |
| `per_view_grouped.json` | `<model_path>/per_view_grouped.json` | Metrics chính/phụ theo từng view; dùng để debug vì sao run A tốt hơn run B. |
| `test/ours_30000/renders/*.png` | Render cuối | Ảnh final dùng để kiểm tra lỗi thị giác thật. |
| `test/ours_30000/gt/*.png` | Ground truth test | Ảnh tham chiếu khi kiểm tra view lỗi. |
| `test/ours_30000/spec/*_final.png` | Debug render | Final render được lưu cùng bộ diagnostic specular. |
| `test/ours_30000/spec/*_only_sh.png` | Debug diffuse/SH | Phần ảnh do SH/diffuse biểu diễn; nếu highlight xuất hiện rõ ở đây thì ASG đang bị mất vai trò. |
| `test/ours_30000/spec/*_only_asg.png` | Debug ASG | Phần ASG/specular học được; đây là output quan trọng nhất để xem ASG có bắt đúng highlight không. |
| `test/ours_30000/spec/*_residual_real.png` | Debug residual | Vùng `GT - only_sh`; dùng như proxy cho phần còn thiếu mà ASG nên giải thích. |
| `point_cloud/iteration_30000/asg.pt` | Checkpoint ASG feature | Kiểm tra/khôi phục feature ASG cuối. |
| `specular/iteration_30000/specular.pth` | Checkpoint specular MLP | Kiểm tra/khôi phục MLP specular cuối. |
| `reflection_prior/` | Dataset scene | Prior 2D dùng cho `ref_score`; cần backup theo version vì thay prior có thể làm thay đổi kết quả rất mạnh. |

### Nhóm Metrics Chính

Các metrics này dùng để báo cáo chất lượng tái tạo chuẩn của 3DGS. Đây là nhóm nên đặt trong bảng chính của thesis/paper.

| Metric | Tốt hơn khi | Ý nghĩa | Cách diễn giải trong ablation |
| --- | --- | --- | --- |
| `PSNR` | Cao hơn | Sai số pixel toàn ảnh thấp hơn. | Metric chính để so sánh fidelity tổng thể. Nếu PSNR tăng nhưng ASG metric không tăng, cải thiện có thể đến từ diffuse/geometry chứ không phải specular. |
| `SSIM` | Cao hơn | Cấu trúc ảnh giống GT hơn. | Hữu ích khi PSNR tăng nhỏ nhưng ảnh nhìn tự nhiên hơn. |
| `LPIPS` | Thấp hơn | Khoảng cách perceptual thấp hơn. | Quan trọng với vùng texture/highlight; nếu LPIPS giảm nhưng PSNR gần như không đổi thì cải thiện có thể là perceptual. |

Nguồn đọc:

```text
results.json
results_grouped.json -> main_metrics
per_view.json
per_view_grouped.json -> main_metrics
```

### Nhóm Metrics Phụ Cho Specular / ASG

Các metrics này không thay thế PSNR/SSIM/LPIPS, mà dùng để giải thích vì sao một config tốt hoặc tệ. Đây là nhóm nên đặt trong bảng phụ hoặc appendix.

| Metric | Tốt hơn khi | Ý nghĩa | Cách diễn giải trong ablation |
| --- | --- | --- | --- |
| `Spec_L1` | Thấp hơn | Lỗi L1 trên vùng specular proxy, lấy từ mask `residual_real > threshold`. | Nếu giảm, model fit vùng highlight tốt hơn. |
| `Spec_PSNR` | Cao hơn | PSNR riêng trên vùng specular proxy. | Metric trực tiếp nhất để xem ASG/ref score có giúp vùng specular không. |
| `NonSpec_L1` | Thấp hơn | Lỗi L1 ngoài vùng specular proxy. | Dùng để kiểm tra specular improvement có làm hỏng vùng diffuse không. |
| `NonSpec_PSNR` | Cao hơn | PSNR ngoài vùng specular proxy. | Nếu `Spec_PSNR` tăng nhưng `NonSpec_PSNR` giảm mạnh, config đang hy sinh diffuse để đổi specular. |
| `ASG_Mean` | Không có hướng cố định | Năng lượng ASG trung bình trong ảnh. | Quá thấp có thể nghĩa là ASG yếu; quá cao có thể nghĩa là ASG bị leak sang diffuse. |
| `ASG_Max` | Không có hướng cố định | Cường độ ASG lớn nhất. | Dùng để biết ASG có đủ sức biểu diễn highlight rất sáng không. |
| `Residual_Mean` | Thấp hơn sau khi SH tốt hơn, nhưng cần đọc cẩn thận | Mức residual thật `GT - only_sh`. | Residual lớn cho thấy SH còn thiếu nhiều; không nhất thiết xấu nếu ASG sau đó giải thích đúng phần thiếu này. |
| `ASG_Energy_In_Residual` | Cao hơn | Tỉ lệ năng lượng ASG rơi vào vùng residual/specular. | Cao nghĩa là ASG tập trung vào vùng nó nên giải thích. |
| `ASG_Residual_IoU` | Cao hơn | Độ overlap giữa vùng ASG sáng và vùng residual/specular. | Tốt để đo ASG có đánh đúng vị trí highlight không. |

Nguồn đọc:

```text
results_grouped.json -> aux_metrics
per_view_grouped.json -> aux_metrics
test/ours_30000/spec/*_only_asg.png
test/ours_30000/spec/*_residual_real.png
```

### Cách Đọc Metrics Khi So Sánh Hai Run

Khi so sánh hai config, không nên chỉ nhìn một số đơn lẻ. Thứ tự đọc đề xuất:

1. Đọc `PSNR`, `SSIM`, `LPIPS` để biết chất lượng tổng thể.
2. Đọc `Spec_PSNR` và `Spec_L1` để biết vùng specular có cải thiện thật không.
3. Đọc `NonSpec_PSNR` và `NonSpec_L1` để kiểm tra vùng diffuse có bị giảm chất lượng không.
4. Đọc `ASG_Energy_In_Residual` và `ASG_Residual_IoU` để biết ASG có học đúng vùng highlight không.
5. Mở `only_asg`, `only_sh`, `residual_real` của các view có `Spec_PSNR` thấp nhất trong `per_view_grouped.json`.
6. Đọc `train_info.json` để xác nhận khác biệt metric đến từ config đang ablate, không phải do nhầm prior, nhầm `asg_degree`, hoặc nhầm `num_score_cameras`.

### Dấu Hiệu Nên Kết Luận

- Config tốt thật sự: `PSNR/SSIM` không giảm, `LPIPS` không tăng, `Spec_PSNR` tăng, `Spec_L1` giảm, `ASG_Residual_IoU` hoặc `ASG_Energy_In_Residual` tăng.
- Config làm ASG tốt hơn nhưng tổng thể chưa chắc tốt hơn: `Spec_PSNR` tăng, `only_asg` visualize highlight rõ hơn, nhưng `PSNR` toàn ảnh gần như không đổi.
- Config làm ASG leak sang diffuse: `ASG_Mean` tăng mạnh, `ASG_Energy_In_Residual` thấp, `ASG_Residual_IoU` thấp, ảnh `only_asg` sáng lan ra vùng không specular.
- Config để SH hấp thụ specular: `only_sh` đã chứa highlight rõ, `only_asg` yếu, `ASG_Mean` thấp, `Spec_PSNR` không cải thiện.
- Config hy sinh diffuse: `Spec_PSNR` tăng nhưng `NonSpec_PSNR` giảm rõ và LPIPS tăng.
- Config prior/ref-score không hiệu quả: bật `USE_REF_SCORE=True` nhưng `ASG_Energy_In_Residual`, `ASG_Residual_IoU`, `Spec_PSNR` không cải thiện so với `USE_REF_SCORE=False`.

### Metrics Hiệu Năng Nên Ghi Thủ Công Hoặc Log Thêm

Các metrics dưới đây chưa phải nhóm chất lượng ảnh, nhưng rất quan trọng để chứng minh spec-fastgs có lợi thế tốc độ so với SpecularGaussian:

| Metric | Tốt hơn khi | Nơi lấy hiện tại | Ý nghĩa |
| --- | --- | --- | --- |
| `Train_Time_Min` | Thấp hơn | `train_info.json` nếu đã log, hoặc terminal log | Chứng minh tốc độ training. |
| `Render_FPS` | Cao hơn | log của `render.py` | Chứng minh tốc độ render/test. |
| `Peak_VRAM_MB` | Thấp hơn | `nvidia-smi` hoặc log nếu bổ sung sau | Chứng minh hiệu quả bộ nhớ. |
| `Final_Gaussians` | Tùy mục tiêu | `train_info.json` nếu đã log, hoặc checkpoint/model stats | Dùng để hiểu VCD/VCP/ADC tạo nhiều hay ít Gaussian. |
| `Checkpoint_Size_MB` | Thấp hơn | kích thước folder output/checkpoint | Dùng để so sánh chi phí lưu trữ. |

## Dataset Cần Chạy

### Nhóm Chính

- `mipnerf360/counter`, `images_8`: scene chính để debug và viết luận. Nên chạy mọi ablation trên scene này trước.
- `mipnerf360/counter`, `images_4`: test độ phân giải cao hơn và chi phí train lớn hơn.

### Nhóm Real Scenes Phụ

Chỉ chạy sau khi kết luận trên nhóm chính đã ổn định:

- `mipnerf360/kitchen`, `images_8` hoặc scale có sẵn.
- `mipnerf360/room`, `images_8` hoặc scale có sẵn.
- `mipnerf360/bonsai`, scale có sẵn.

### Nhóm Synthetic / Shiny

- `Ref-NeRF/refnerf/toaster`: object shiny có kiểm soát, chạy qua `run_shiny.sh`.

## Config Mặc Định Cố Định

Dùng config này làm default, trừ khi ablation yêu cầu thay đổi:

```bash
ASG_DEGREE=24
USE_REF_SCORE=True
EXTRACT_REF_PRIOR=False
NUM_SCORE_CAMERAS=10
FULL_ASG_INTERVAL=0
F_REST_WARMUP_UNTIL=0
F_REST_INTERVAL_EARLY=16
F_REST_INTERVAL_MID=32
F_REST_INTERVAL_LATE=64
--specular_start_iter 3000
--densification_interval 100
--optimizer_type default
--sh_degree 3
```

Với `reflection_prior`, mỗi ablation nên dùng một folder prior cố định. Không regenerate prior trong lúc train, trừ khi ablation đó đang kiểm tra thuật toán tạo prior.

## Nhóm Ablation A: Tái Lập Baseline

Mục đích: tái lập run tốt đã biết `spec-fastgs_v3_new_architecture_20260706_152007`.

### A0: Compatibility Với Known-Good Run

```bash
ASG_DEGREE=24
USE_REF_SCORE=True
EXTRACT_REF_PRIOR=False
NUM_SCORE_CAMERAS=10
FULL_ASG_INTERVAL=0
F_REST_WARMUP_UNTIL=0
```

Kỳ vọng tham chiếu trên `counter/images_8`:

- PSNR khoảng `30.77`
- SSIM khoảng `0.9388`
- LPIPS khoảng `0.0651`
- thời gian train khoảng `15m`

### A1: Không Dùng Ref Score

```bash
USE_REF_SCORE=False
EXTRACT_REF_PRIOR=False
```

Mục đích: đo đóng góp thật của ref-score-guided densification.

### A2: Dùng Ref Score Nhưng Không Tạo Lại Prior

```bash
USE_REF_SCORE=True
EXTRACT_REF_PRIOR=False
```

Đây là setting so sánh công bằng với A1.

## Nhóm Ablation B: Thuật Toán Tạo Reflection Prior

Mục đích: xác định prior 2D nào giúp `only_asg` visualize specular tốt hơn.

Luôn chạy `run_extract_reflection_prior.sh` trước, sau đó train với:

```bash
USE_REF_SCORE=True
EXTRACT_REF_PRIOR=False
```

Script extract sẽ backup prior cũ vào:

```text
datasets/<dataset>/<scene>/backups/reflection_prior_<timestamp>
```

### B0: Tan-Ikeuchi 0.35 / 0.65

```bash
REF_PRIOR_METHOD=tan
TI_THRESH=0.35
TI_BRIGHT=0.65
```

Giả thuyết: mask rộng và hữu ích hơn cho multi-view specular so với Shafer/Klinker, có khả năng giúp `only_asg` visualize specular tốt hơn.

### B1: Tan-Ikeuchi 0.35 / 0.60

```bash
REF_PRIOR_METHOD=tan
TI_THRESH=0.35
TI_BRIGHT=0.60
```

Giả thuyết: mask rộng hơn, có thể tăng coverage cho ASG nhưng rủi ro đưa cả vùng diffuse sáng vào prior.

### B2: Shafer/Klinker 0.65 / 0.35

```bash
REF_PRIOR_METHOD=shafer
SK_INTENSITY=0.65
SK_SATURATION=0.35
```

Giả thuyết: chọn vùng highlight trắng/xám chính xác hơn, precision cao hơn nhưng coverage thấp hơn.

### B3: Shafer/Klinker 0.70 / 0.20

```bash
REF_PRIOR_METHOD=shafer
SK_INTENSITY=0.70
SK_SATURATION=0.20
```

Giả thuyết: prior rất strict, có thể quá sparse để hỗ trợ densification và ASG.

## Nhóm Ablation C: Capacity Của ASG

Mục đích: đo số chiều latent ASG ảnh hưởng thế nào đến metrics cuối và ảnh `only_asg`.

Giữ nguyên cùng một folder `reflection_prior`.

### C0: ASG 12

```bash
ASG_DEGREE=12
```

Kỳ vọng: nhanh hơn, nhẹ hơn, nhưng có thể yếu hơn ở specular.

### C1: ASG 24

```bash
ASG_DEGREE=24
```

Default hiện tại và là trade-off hợp lý nhất.

### C2: ASG 32

```bash
ASG_DEGREE=32
```

Có thể tăng capacity nhẹ. Tuy nhiên với real/indoor, ASG cuối vẫn bị nén về 32 tham số ASG, nên lợi ích có thể bão hòa.

## Nhóm Ablation D: Số Camera Dùng Cho Ref Score / VCD / VCP Scoring

Mục đích: kiểm tra cần bao nhiêu view để ADC/VCD/VCP scoring ổn định.

Giữ nguyên ASG degree và prior.

### D0: 3 Score Cameras

```bash
NUM_SCORE_CAMERAS=3
```

Kỳ vọng: nhanh hơn nhưng densification nhiễu hơn.

### D1: 5 Score Cameras

```bash
NUM_SCORE_CAMERAS=5
```

Điểm giữa giữa tốc độ và chất lượng.

### D2: 10 Score Cameras

```bash
NUM_SCORE_CAMERAS=10
```

Default, gần nhất với hành vi FastGS/spec-fastgs cũ.

## Nhóm Ablation E: Iteration Bắt Đầu ASG

Mục đích: kiểm tra ASG bắt đầu quá sớm hay quá muộn so với densification.

### E0: Bắt Đầu Từ Iter 0

```bash
--specular_start_iter 0
```

Rủi ro: training early-stage chưa ổn định; ASG cạnh tranh khi geometry chưa tốt.

### E1: Bắt Đầu Từ Iter 3000

```bash
--specular_start_iter 3000
```

Default lịch sử.

### E2: Bắt Đầu Từ Iter 5000

```bash
--specular_start_iter 5000
```

Giả thuyết: geometry ổn hơn trước khi ASG học, có thể sạch hơn nhưng ít thời gian học specular hơn.

## Nhóm Ablation F: Cạnh Tranh SH / ASG

Mục đích: kiểm tra SH có hấp thụ specular và làm `only_asg` yếu đi hay không.

### F0: Lịch f_rest Lịch Sử

```bash
F_REST_WARMUP_UNTIL=0
F_REST_INTERVAL_EARLY=16
F_REST_INTERVAL_MID=32
F_REST_INTERVAL_LATE=64
```

Setting tương thích với bản cũ.

### F1: f_rest Warmup Early

```bash
F_REST_WARMUP_UNTIL=3000
```

Giả thuyết: có thể fit diffuse tốt hơn nhưng cũng có thể để SH hấp thụ highlight trước khi ASG bắt đầu.

### F2: f_rest Sparse Hơn Ở Early Stage

```bash
F_REST_WARMUP_UNTIL=0
F_REST_INTERVAL_EARLY=32
```

Giả thuyết: giảm update SH-rest có thể đẩy residual view-dependent sang ASG nhiều hơn.

## Nhóm Ablation G: Full ASG Refresh

Mục đích: kiểm tra sparse ASG có bỏ sót Gaussian quan trọng hay không.

### G0: Tắt

```bash
FULL_ASG_INTERVAL=0
```

Default ưu tiên tốc độ.

### G1: Refresh Mỗi 3000 Iter

```bash
FULL_ASG_INTERVAL=3000
```

Kỳ vọng: chậm hơn, có thể tăng coverage của ASG.

### G2: Refresh Mỗi 1000 Iter

```bash
FULL_ASG_INTERVAL=1000
```

Rất tốn thời gian; chỉ nên chạy nếu G1 có tín hiệu tốt.

## Thứ Tự Chạy Đề Xuất

Chạy trên `counter/images_8` trước:

1. A0, A1, A2.
2. B0, B1, B2, B3.
3. C0, C1, C2 với prior tốt nhất từ nhóm B.
4. D0, D1, D2 với prior và ASG degree tốt nhất.
5. F0, F1, F2 nếu `only_asg` vẫn yếu.
6. E1, E2 nếu timing giữa geometry/specular vẫn chưa ổn.
7. Lặp lại 2 setting tốt nhất trên `counter/images_4`.
8. Validate 2 setting cuối trên ít nhất một real scene khác và `toaster`.

## Bảng Tối Thiểu Cho Thesis

Mỗi dòng thí nghiệm nên ghi:

```text
dataset / image_scale / prior_method / thresholds / use_ref_score /
asg_degree / num_score_cameras / specular_start_iter /
f_rest schedule / full_asg_interval /
PSNR / SSIM / LPIPS / time / VRAM / final_gaussians /
ASG-residual overlap / comments
```

## Ghi Chú

- Không so sánh trực tiếp các run dùng `reflection_prior` được tạo bởi thuật toán khác nhau, trừ khi đó chính là ablation đang xét.
- Sau khi đã có prior mong muốn, giữ `EXTRACT_REF_PRIOR=False` trong lúc train.
- Nếu mục tiêu là `only_asg`, PSNR toàn ảnh là chưa đủ.
- Nên chạy lặp lại nếu chênh lệch nhỏ hơn khoảng `0.05 PSNR` hoặc `0.001 SSIM`, vì randomness từ sampling/densification có thể gây nhiễu.
