# Ghi chú ablation spec-fastgs

## Results

### Mip-NeRF360

| Run | date_completed | Scene | Output | Config chính | PSNR | SSIM | LPIPS | Spec_PSNR | ASG_IoU | Số Gaussian | Thời gian | Kết luận nhanh |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| R001 | 2026-07-07 10:52:56 | counter | [output/backups/counter/spec-fastgs_v3_new_architecture_20260707_105338][R001-output] | tan, ti=0.35, bright=0.60, use_ref_score=True, asg=24 | 30.8633 | 0.9389 | 0.0639 | 23.5746 | 0.2229 | 179448 | 14m 31s | Kết quả tổng thể tốt; cần xem visual ASG để kết luận prior |
| R002 | 2026-07-07 12:40:00 | counter | [output/backups/counter/spec-fastgs_v3_new_architecture_20260707_124037][R002-output] | tan, use_ref_score=False, asg=24 | 30.7423 | 0.9384 | 0.0650 | 23.2490 | 0.2021 | 160779 | 12m 55s | Thấp hơn R001; ref-score đang có lợi nhẹ trên counter |
| R010 | 2026-07-07 14:49:45 | counter | [output/backups/counter/spec-fastgs_v3_new_architecture_20260707_145020][R010-output] | shafer, sk=0.65, sat=0.30, use_ref_score=True, asg=32 | 30.7633 | 0.9389 | 0.0655 | 23.1480 | 0.2114 | 179139 | 14m 52s | Shafer asg32 thấp hơn Tan R001 theo PSNR/Spec_PSNR |
| R011 | 2026-07-07 15:27:55 | counter | [output/backups/counter/spec-fastgs_v3_new_architecture_20260707_152830][R011-output] | shafer, sk=0.65, sat=0.30, use_ref_score=True, asg=48 | 30.8530 | 0.9387 | 0.0647 | 23.4116 | 0.2175 | 180024 | 15m 48s | Tăng ASG lên 48 cải thiện rõ so với R010 |
| R012 | 2026-07-07 15:46:39 | counter | [output/backups/counter/spec-fastgs_v3_new_architecture_20260707_154715][R012-output] | shafer, sk=0.65, sat=0.30, use_ref_score=True, asg=64 | 30.9368 | 0.9401 | 0.0636 | 23.4606 | 0.2321 | 178884 | 16m 19s | Best counter cũ trước adaptive prior |
| R019 | 2026-07-09 07:58:55 | counter | [output/backups/counter/spec-fastgs_v3_new_architecture_20260709_075930][R019-output] | shafer, adaptive=False, asg=64, auto-budget | 30.8666 | 0.9394 | 0.0633 | 23.3781 | 0.2369 | 175572 | 15m 57s | Auto-budget/soft decay không adaptive: IoU cao nhưng fidelity giảm |
| R020 | 2026-07-09 08:16:36 | counter | [output/counter][R020-output] | shafer, adaptive=True, asg=64, auto-budget | 30.9569 | 0.9403 | 0.0632 | 23.4845 | 0.2361 | 178464 | 16m 12s | Best counter mới theo PSNR/SSIM/LPIPS/Spec_PSNR |

### Ref-NeRF

| Run | date_completed | Scene | Output | Config chính | PSNR | SSIM | LPIPS | Spec_PSNR | ASG_IoU | Số Gaussian | Thời gian | Kết luận nhanh |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| R003 | 2026-07-07 12:20:08 | toaster | [output/backups/toaster/spec-fastgs_v3_new_architecture_20260707_122528][R003-output] | tan, use_ref_score=True, asg=24 | 22.0849 | 0.8931 | 0.1199 | 18.2073 | 0.5934 | 34724 | 6m 59s | Baseline toaster với ref-score=True |
| R004 | 2026-07-07 12:48:19 | toaster | [output/backups/toaster/spec-fastgs_v3_new_architecture_20260707_125351][R004-output] | tan, use_ref_score=False, asg=24 | 22.1691 | 0.8934 | 0.1201 | 18.3100 | 0.5954 | 34988 | 7m 7s | False tốt hơn R003 về PSNR/Spec_PSNR, nhưng LPIPS nhỉnh hơn |
| R005 | 2026-07-07 13:05:44 | toaster | [output/backups/toaster/spec-fastgs_v3_new_architecture_20260707_131110][R005-output] | tan, use_ref_score=True, asg=24 | 22.0108 | 0.8947 | 0.1178 | 17.9965 | 0.6012 | 36061 | 7m 15s | Run PSNR 22.0108 thực tế là ref-score=True, không phải False |
| R006 | 2026-07-07 13:20:56 | toaster | [output/backups/toaster/spec-fastgs_v3_new_architecture_20260707_132623][R006-output] | tan, use_ref_score=False, asg=24 | 22.0912 | 0.8933 | 0.1207 | 18.1837 | 0.5955 | 34911 | 7m 4s | False chạy lại thấp hơn R004; có dao động giữa các run |
| R007 | 2026-07-07 13:56:52 | toaster | [output/backups/toaster/spec-fastgs_v3_new_architecture_20260707_140203][R007-output] | shafer, sk=0.70, sat=0.20, use_ref_score=True, asg=24 | 22.0445 | 0.8930 | 0.1198 | 18.0868 | 0.6059 | 34943 | 7m 1s | Shafer strict; ASG_IoU cao nhất nhưng PSNR thấp |
| R008 | 2026-07-07 14:11:43 | toaster | [output/backups/toaster/spec-fastgs_v3_new_architecture_20260707_141652][R008-output] | shafer, sk=0.65, sat=0.30, use_ref_score=True, asg=24 | 22.1122 | 0.8937 | 0.1194 | 18.3628 | 0.5939 | 35200 | 7m 4s | Shafer 0.65/0.30 tốt nhất nhóm toaster theo Spec_PSNR |
| R009 | 2026-07-07 14:27:13 | toaster | [output/backups/toaster/spec-fastgs_v3_new_architecture_20260707_143226][R009-output] | shafer, sk=0.60, sat=0.30, use_ref_score=True, asg=24 | 22.1023 | 0.8939 | 0.1195 | 18.2660 | 0.5947 | 35020 | 7m 5s | Shafer 0.60/0.30 ổn, nhưng kém R008 về PSNR/Spec_PSNR |
| R013 | 2026-07-08 15:49:01 | toaster | [output/backups/toaster/spec-fastgs_v3_new_architecture_20260708_155436][R013-output] | shafer, sk=0.65, sat=0.30, use_ref_score=True, asg=32 | 22.1091 | 0.8943 | 0.1188 | 18.0103 | 0.6108 | 34691 | 7m 36s | ASG32 cải thiện LPIPS/IoU nhưng giảm Spec_PSNR |
| R014 | 2026-07-08 16:10:43 | toaster | [output/backups/toaster/spec-fastgs_v3_new_architecture_20260708_161615][R014-output] | shafer, sk=0.65, sat=0.30, use_ref_score=True, asg=48 | 22.0680 | 0.8939 | 0.1185 | 18.1737 | 0.6175 | 34641 | 7m 37s | ASG48 tiếp tục tăng IoU/LPIPS nhưng PSNR giảm |
| R015 | 2026-07-08 16:24:23 | toaster | [output/backups/toaster/spec-fastgs_v3_new_architecture_20260708_162950][R015-output] | shafer, sk=0.65, sat=0.30, use_ref_score=True, asg=64 | 22.0471 | 0.8951 | 0.1183 | 18.1298 | 0.6343 | 34505 | 7m 43s | Best toaster hiện tại theo SSIM/LPIPS/ASG_IoU |
| R016 | 2026-07-08 16:39:03 | toaster | [output/backups/toaster/spec-fastgs_v3_new_architecture_20260708_164439][R016-output] | shafer, sk=0.65, sat=0.30, use_ref_score=True, asg=24 | 22.1508 | 0.8939 | 0.1197 | 18.3542 | 0.5866 | 35018 | 7m 21s | Best toaster cũ theo PSNR; rerun asg24 tốt hơn R008 |
| R017 | 2026-07-09 07:20:34 | toaster | [output/backups/toaster/spec-fastgs_v3_new_architecture_20260709_072529][R017-output] | shafer, adaptive=False, asg=24, auto-budget | 22.0872 | 0.8940 | 0.1196 | 18.1198 | 0.6010 | 34884 | 7m 3s | Mốc off cho adaptive prior trên toaster |
| R018 | 2026-07-09 07:36:00 | toaster | [output/backups/toaster/spec-fastgs_v3_new_architecture_20260709_074103][R018-output] | shafer, adaptive=True, asg=24, auto-budget | 22.1166 | 0.8931 | 0.1192 | 18.2814 | 0.6104 | 35142 | 7m 4s | Adaptive prior cải thiện PSNR/LPIPS/Spec_PSNR/IoU so với R017 |
| R021 | 2026-07-09 08:28:15 | toaster | [output/backups/toaster/spec-fastgs_v3_new_architecture_20260709_083322][R021-output] | shafer, adaptive=True, asg=64, auto-budget | 22.0331 | 0.8953 | 0.1171 | 18.1242 | 0.5851 | 34638 | 7m 21s | Best Shafer toaster theo SSIM/LPIPS, nhưng IoU giảm |
| R022 | 2026-07-09 08:40:52 | toaster | [output/backups/toaster/spec-fastgs_v3_new_architecture_20260709_084600][R022-output] | shafer, adaptive=True, asg=12, auto-budget | 21.8971 | 0.8917 | 0.1208 | 18.1607 | 0.5367 | 34435 | 7m 1s | ASG capacity quá thấp làm tụt metric rõ |
| R023 | 2026-07-09 09:00:24 | toaster | [output/backups/toaster/spec-fastgs_v3_new_architecture_20260709_090531][R023-output] | shafer, adaptive=True, asg=32, auto-budget | 22.0221 | 0.8935 | 0.1190 | 18.3092 | 0.5881 | 34553 | 7m 3s | Spec_PSNR cao trong nhóm adaptive nhưng PSNR thấp |
| R024 | 2026-07-09 10:43:07 | toaster | [output/toaster][R024-output] | tan, adaptive=True, asg=32, auto-budget | 22.1927 | 0.8957 | 0.1162 | 18.2631 | 0.6138 | 35240 | 7m 5s | Best toaster mới theo PSNR/SSIM/LPIPS và IoU tốt |

## Notes

### R001

- Mục tiêu: thử ref-score guided densification với prior Tan-Ikeuchi trên scene counter.
- Output: [output/backups/counter/spec-fastgs_v3_new_architecture_20260707_105338][R001-output]
- Config: scene=counter, images=images_8, iterations=30000, asg_degree=24, use_ref_score=True, ref_prior_method=tan, ti_thresh=0.35, ti_bright=0.60, specular_start_iter=3000, num_score_cameras=10, full_asg_interval=0.
- Kết quả đáng chú ý: PSNR=30.8633, SSIM=0.9389, LPIPS=0.0639; Spec_PSNR=23.5746, NonSpec_PSNR=34.1300, ASG_Energy_In_Residual=0.5175, ASG_Residual_IoU=0.2229.
- Training: initial_gaussians=155767, final_gaussians=179448, peak_vram=1295.82 MiB, time=14m31s.
- Quan sát ảnh render / only_asg / only_sh: chưa kiểm tra visual.
- Vấn đề: ASG_Residual_IoU còn thấp, cần xem `only_asg` có leak sang vùng diffuse/bright hay không.
- Kết luận: run tốt về metric tổng thể; chưa đủ để kết luận Tan-Ikeuchi là prior tốt nhất nếu chưa so với Shafer và chưa xem visual.
- Run tiếp theo: chạy Shafer/Klinker cùng setting, ưu tiên `sk_intensity=0.65`, `sk_saturation=0.30` hoặc `0.70/0.20`.

### R002

- Mục tiêu: so sánh counter khi tắt ref-score guided densification.
- Output: [output/backups/counter/spec-fastgs_v3_new_architecture_20260707_124037][R002-output]
- Config: scene=counter, images=images_8, iterations=30000, asg_degree=24, use_ref_score=False, ref_prior_method=tan, ti_thresh=0.35, ti_bright=0.60.
- Kết quả đáng chú ý: PSNR=30.7423, SSIM=0.9384, LPIPS=0.0650; Spec_PSNR=23.2490, NonSpec_PSNR=34.0880, ASG_Energy_In_Residual=0.5465, ASG_Residual_IoU=0.2021.
- Training: initial_gaussians=155767, final_gaussians=160779, peak_vram=1156.86 MiB, time=12m55s.
- Nhận xét: so với R001, PSNR giảm 0.1209, LPIPS tăng, Spec_PSNR giảm 0.3256, ASG_IoU giảm 0.0208; ref-score có vẻ giúp counter nhẹ nhưng không quá lớn.
- Kết luận: với counter, R001 tốt hơn R002 theo metric tổng thể và specular proxy.

### R003

- Mục tiêu: chạy toaster với ref-score=True.
- Output: [output/backups/toaster/spec-fastgs_v3_new_architecture_20260707_122528][R003-output]
- Config: scene=toaster, images=images, iterations=30000, asg_degree=24, use_ref_score=True, ref_prior_method=tan, ti_thresh=0.35, ti_bright=0.60.
- Kết quả đáng chú ý: PSNR=22.0849, SSIM=0.8931, LPIPS=0.1199; Spec_PSNR=18.2073, ASG_Energy_In_Residual=0.8248, ASG_Residual_IoU=0.5934.
- Training: initial_gaussians=100000, final_gaussians=34724, peak_vram=2922.49 MiB, time=6m59s.
- Kết luận: dùng làm mốc ref-score=True đầu tiên cho toaster.

### R004

- Mục tiêu: chạy toaster với ref-score=False để so với R003.
- Output: [output/backups/toaster/spec-fastgs_v3_new_architecture_20260707_125351][R004-output]
- Config: scene=toaster, images=images, iterations=30000, asg_degree=24, use_ref_score=False, ref_prior_method=tan, ti_thresh=0.35, ti_bright=0.60.
- Kết quả đáng chú ý: PSNR=22.1691, SSIM=0.8934, LPIPS=0.1201; Spec_PSNR=18.3100, ASG_Energy_In_Residual=0.8331, ASG_Residual_IoU=0.5954.
- Training: initial_gaussians=100000, final_gaussians=34988, peak_vram=2660.85 MiB, time=7m07s.
- Nhận xét: PSNR và Spec_PSNR cao hơn R003 dù ref-score=False; LPIPS hơi xấu hơn.
- Kết luận: trên toaster, hiệu ứng ref-score chưa ổn định; cần nhiều hơn một run/seed để kết luận.

### R005

- Mục tiêu: chạy toaster với ref-score=True sau khi tạo/làm mới reflection prior.
- Output: [output/backups/toaster/spec-fastgs_v3_new_architecture_20260707_131110][R005-output]
- Config: scene=toaster, images=images, iterations=30000, asg_degree=24, use_ref_score=True, ref_prior_method=tan, ti_thresh=0.35, ti_bright=0.60.
- Kết quả đáng chú ý: PSNR=22.0108, SSIM=0.8947, LPIPS=0.1178; Spec_PSNR=17.9965, ASG_Energy_In_Residual=0.8263, ASG_Residual_IoU=0.6012.
- Training: initial_gaussians=100000, final_gaussians=36061, peak_vram=2935.19 MiB, time=7m15s.
- Kiểm tra nhầm lẫn: PSNR=22.0108089 thuộc run `use_ref_score=True`, không phải `use_ref_score=False` theo `train_info.json`.
- Nhận xét: PSNR/Spec_PSNR thấp hơn R003 và R004, nhưng SSIM/LPIPS/ASG_IoU lại tốt hơn. Đây là run có trade-off khác, không đơn giản là tốt/xấu theo một metric.
- Kết luận: nếu mục tiêu là PSNR/spec proxy thì R005 không tốt; nếu xét LPIPS/ASG_IoU thì vẫn cần xem visual.

### R006

- Mục tiêu: chạy lại toaster với ref-score=False sau R005.
- Output: [output/backups/toaster/spec-fastgs_v3_new_architecture_20260707_132623][R006-output]
- Config: scene=toaster, images=images, iterations=30000, asg_degree=24, use_ref_score=False, ref_prior_method=tan, ti_thresh=0.35, ti_bright=0.60.
- Kết quả đáng chú ý: PSNR=22.0912, SSIM=0.8933, LPIPS=0.1207; Spec_PSNR=18.1837, ASG_Energy_In_Residual=0.8307, ASG_Residual_IoU=0.5955.
- Training: initial_gaussians=100000, final_gaussians=34911, peak_vram=2657.43 MiB, time=7m04s.
- Nhận xét: cùng `use_ref_score=False` nhưng thấp hơn R004 về PSNR/Spec_PSNR; khả năng là dao động training/random sampling, không phải do reflection prior vì `use_ref_score=False` thì prior không được dùng trong densification.
- Kết luận: chưa đủ bằng chứng nói ref-score=True/False thắng trên toaster; cần so sánh bằng trung bình nhiều run hoặc cố định seed.

### Kiểm tra nhanh

- Run có PSNR=22.0108089 là R005 và metadata ghi `use_ref_score=True`.
- Các run `use_ref_score=False` trên toaster là R004 và R006, với PSNR lần lượt 22.1691 và 22.0912.
- Việc chạy lại `extract_reflection_prior.py` không nên ảnh hưởng trực tiếp tới run `use_ref_score=False`, vì train chỉ load/use prior khi bật `--use_ref_score`.

### R007

- Mục tiêu: thử Shafer/Klinker strict trên toaster.
- Output: [output/backups/toaster/spec-fastgs_v3_new_architecture_20260707_140203][R007-output]
- Config: scene=toaster, images=images, iterations=30000, asg_degree=24, use_ref_score=True, ref_prior_method=shafer, sk_intensity=0.70, sk_saturation=0.20.
- Kết quả đáng chú ý: PSNR=22.0445, SSIM=0.8930, LPIPS=0.1198; Spec_PSNR=18.0868, ASG_Energy_In_Residual=0.8320, ASG_Residual_IoU=0.6059.
- Training: initial_gaussians=100000, final_gaussians=34943, peak_vram=2922.65 MiB, time=7m01s.
- Nhận xét: ASG_IoU cao nhất trong nhóm toaster hiện tại, nhưng PSNR/Spec_PSNR không tốt.
- Kết luận: strict Shafer có thể định vị ASG tốt hơn theo IoU, nhưng chưa tốt nếu ưu tiên fidelity.

### R008

- Mục tiêu: thử Shafer/Klinker cân bằng hơn trên toaster.
- Output: [output/backups/toaster/spec-fastgs_v3_new_architecture_20260707_141652][R008-output]
- Config: scene=toaster, images=images, iterations=30000, asg_degree=24, use_ref_score=True, ref_prior_method=shafer, sk_intensity=0.65, sk_saturation=0.30.
- Kết quả đáng chú ý: PSNR=22.1122, SSIM=0.8937, LPIPS=0.1194; Spec_PSNR=18.3628, ASG_Energy_In_Residual=0.8234, ASG_Residual_IoU=0.5939.
- Training: initial_gaussians=100000, final_gaussians=35200, peak_vram=2928.02 MiB, time=7m04s.
- Nhận xét: tốt nhất trong nhóm toaster Shafer theo PSNR và Spec_PSNR; cũng vượt các run Tan true theo Spec_PSNR.
- Kết luận: Shafer 0.65/0.30 là candidate tốt nhất hiện tại cho toaster nếu ưu tiên PSNR/Spec_PSNR.

### R009

- Mục tiêu: thử Shafer/Klinker rộng hơn trên toaster.
- Output: [output/backups/toaster/spec-fastgs_v3_new_architecture_20260707_143226][R009-output]
- Config: scene=toaster, images=images, iterations=30000, asg_degree=24, use_ref_score=True, ref_prior_method=shafer, sk_intensity=0.60, sk_saturation=0.30.
- Kết quả đáng chú ý: PSNR=22.1023, SSIM=0.8939, LPIPS=0.1195; Spec_PSNR=18.2660, ASG_Energy_In_Residual=0.8233, ASG_Residual_IoU=0.5947.
- Training: initial_gaussians=100000, final_gaussians=35020, peak_vram=2928.70 MiB, time=7m05s.
- Nhận xét: gần R008 nhưng thấp hơn về PSNR/Spec_PSNR; SSIM nhỉnh hơn nhẹ.
- Kết luận: usable, nhưng chưa có lý do chọn hơn R008 nếu mục tiêu là specular fidelity.

### R010

- Mục tiêu: thử Shafer/Klinker trên counter với ASG degree cao hơn.
- Output: [output/backups/counter/spec-fastgs_v3_new_architecture_20260707_145020][R010-output]
- Config: scene=counter, images=images_8, iterations=30000, asg_degree=32, use_ref_score=True, ref_prior_method=shafer, sk_intensity=0.65, sk_saturation=0.30.
- Kết quả đáng chú ý: PSNR=30.7633, SSIM=0.9389, LPIPS=0.0655; Spec_PSNR=23.1480, ASG_Energy_In_Residual=0.5132, ASG_Residual_IoU=0.2114.
- Training: initial_gaussians=155767, final_gaussians=179139, peak_vram=1311.76 MiB, time=14m52s.
- Nhận xét: thấp hơn R001 Tan true về PSNR, LPIPS, Spec_PSNR và ASG_IoU; chỉ gần tương đương SSIM.
- Kết luận: R010 chưa tốt bằng R001; các run ASG degree cao hơn cần được so tiếp.

### R011

- Mục tiêu: tăng ASG degree lên 48 cho counter với Shafer 0.65/0.30.
- Output: [output/backups/counter/spec-fastgs_v3_new_architecture_20260707_152830][R011-output]
- Config: scene=counter, images=images_8, iterations=30000, asg_degree=48, use_ref_score=True, ref_prior_method=shafer, sk_intensity=0.65, sk_saturation=0.30.
- Kết quả đáng chú ý: PSNR=30.8530, SSIM=0.9387, LPIPS=0.0647; Spec_PSNR=23.4116, ASG_Energy_In_Residual=0.5563, ASG_Residual_IoU=0.2175.
- Training: initial_gaussians=155767, final_gaussians=180024, peak_vram=1396.42 MiB, time=15m48s.
- Nhận xét: tăng ASG degree từ 32 lên 48 cải thiện PSNR, LPIPS, Spec_PSNR và ASG_Energy_In_Residual so với R010.
- Kết luận: ASG 48 là bước cải thiện rõ cho counter Shafer.

### R012

- Mục tiêu: tăng ASG degree lên 64 cho counter với Shafer 0.65/0.30.
- Output: [output/backups/counter/spec-fastgs_v3_new_architecture_20260707_154715][R012-output]
- Config: scene=counter, images=images_8, iterations=30000, asg_degree=64, use_ref_score=True, ref_prior_method=shafer, sk_intensity=0.65, sk_saturation=0.30.
- Kết quả đáng chú ý: PSNR=30.9368, SSIM=0.9401, LPIPS=0.0636; Spec_PSNR=23.4606, ASG_Energy_In_Residual=0.5087, ASG_Residual_IoU=0.2321.
- Training: initial_gaussians=155767, final_gaussians=178884, peak_vram=1475.31 MiB, time=16m19s.
- Nhận xét: tốt nhất nhóm counter hiện tại theo PSNR/SSIM/LPIPS/ASG_IoU; Spec_PSNR vẫn thấp hơn R001 Tan.
- Kết luận: nếu ưu tiên metric tổng thể cho counter, R012 đang là best run hiện tại; nếu ưu tiên Spec_PSNR thuần, R001 vẫn mạnh.

### R013

- Mục tiêu: tăng ASG degree lên 32 cho toaster với Shafer 0.65/0.30.
- Output: [output/backups/toaster/spec-fastgs_v3_new_architecture_20260708_155436][R013-output]
- Config: scene=toaster, images=images, iterations=30000, asg_degree=32, use_ref_score=True, ref_prior_method=shafer, sk_intensity=0.65, sk_saturation=0.30.
- Kết quả đáng chú ý: PSNR=22.1091, SSIM=0.8943, LPIPS=0.1188; Spec_PSNR=18.0103, ASG_Energy_In_Residual=0.8329, ASG_Residual_IoU=0.6108.
- Training: initial_gaussians=100000, final_gaussians=34691, peak_vram=2938.29 MiB, time=7m36s.
- Nhận xét: so với R008 ASG24, LPIPS và IoU tốt hơn nhưng Spec_PSNR giảm mạnh.
- Kết luận: ASG32 có lợi cho perceptual/IoU hơn là PSNR/spec proxy.

### R014

- Mục tiêu: tăng ASG degree lên 48 cho toaster với Shafer 0.65/0.30.
- Output: [output/backups/toaster/spec-fastgs_v3_new_architecture_20260708_161615][R014-output]
- Config: scene=toaster, images=images, iterations=30000, asg_degree=48, use_ref_score=True, ref_prior_method=shafer, sk_intensity=0.65, sk_saturation=0.30.
- Kết quả đáng chú ý: PSNR=22.0680, SSIM=0.8939, LPIPS=0.1185; Spec_PSNR=18.1737, ASG_Energy_In_Residual=0.8399, ASG_Residual_IoU=0.6175.
- Training: initial_gaussians=100000, final_gaussians=34641, peak_vram=2955.64 MiB, time=7m37s.
- Nhận xét: ASG_IoU và LPIPS tiếp tục cải thiện, nhưng PSNR thấp hơn nhóm ASG24.
- Kết luận: đáng xem visual `only_asg`; metric cho thấy ASG tập trung hơn nhưng fidelity tổng thể chưa thắng.

### R015

- Mục tiêu: tăng ASG degree lên 64 cho toaster với Shafer 0.65/0.30.
- Output: [output/backups/toaster/spec-fastgs_v3_new_architecture_20260708_162950][R015-output]
- Config: scene=toaster, images=images, iterations=30000, asg_degree=64, use_ref_score=True, ref_prior_method=shafer, sk_intensity=0.65, sk_saturation=0.30.
- Kết quả đáng chú ý: PSNR=22.0471, SSIM=0.8951, LPIPS=0.1183; Spec_PSNR=18.1298, ASG_Energy_In_Residual=0.8459, ASG_Residual_IoU=0.6343.
- Training: initial_gaussians=100000, final_gaussians=34505, peak_vram=2973.25 MiB, time=7m43s.
- Nhận xét: best toaster hiện tại theo SSIM, LPIPS, ASG_Energy và ASG_IoU, nhưng PSNR/Spec_PSNR thấp hơn R016/R008.
- Kết luận: nếu mục tiêu là ASG alignment/perceptual, R015 rất đáng xem visual; nếu mục tiêu là PSNR thì chưa phải best.

### R016

- Mục tiêu: rerun ASG degree 24 cho toaster với Shafer 0.65/0.30.
- Output: [output/backups/toaster/spec-fastgs_v3_new_architecture_20260708_164439][R016-output]
- Config: scene=toaster, images=images, iterations=30000, asg_degree=24, use_ref_score=True, ref_prior_method=shafer, sk_intensity=0.65, sk_saturation=0.30.
- Kết quả đáng chú ý: PSNR=22.1508, SSIM=0.8939, LPIPS=0.1197; Spec_PSNR=18.3542, ASG_Energy_In_Residual=0.8222, ASG_Residual_IoU=0.5866.
- Training: initial_gaussians=100000, final_gaussians=35018, peak_vram=2927.04 MiB, time=7m21s.
- Nhận xét: best toaster hiện tại theo PSNR, và gần R008 về Spec_PSNR; thấp hơn nhóm ASG cao về LPIPS/IoU.
- Kết luận: ASG24 vẫn tốt nhất nếu ưu tiên PSNR; ASG64 tốt hơn nếu ưu tiên visual/perceptual proxy.

### R017

- Mục tiêu: tạo mốc `use_adaptive_prior=False` cho toaster sau khi thêm các chỉnh sửa Geometry Coverage: scene-relative RefScore budget, soft RefScore decay và f_rest sparse schedule.
- Output: [output/backups/toaster/spec-fastgs_v3_new_architecture_20260709_072529][R017-output]
- Config: scene=toaster, images=images, iterations=30000, asg_degree=24, use_ref_score=True, use_adaptive_prior=False, ref_prior_method=shafer, sk_intensity=0.65, sk_saturation=0.30, max_refscore_gaussians=1000000, refscore_threshold=0.50->0.90, refscore_min_strength=0.15, f_rest_interval=16/32/64.
- Kết quả đáng chú ý: PSNR=22.0872, SSIM=0.8940, LPIPS=0.1196; Spec_PSNR=18.1198, ASG_Energy_In_Residual=0.8324, ASG_Residual_IoU=0.6010.
- Training: initial_gaussians=100000, final_gaussians=34884, peak_vram=3196.40 MiB, time=7m03s.
- Nhận xét: so với R016 cùng `asg=24`, metric fidelity giảm; đây là dấu hiệu các chỉnh sửa budget/decay/schedule tự thân chưa đủ, hoặc run-to-run variance vẫn lớn.
- Kết luận: dùng R017 làm baseline off cho cặp adaptive-prior R017/R018, không nên coi là best toaster.

### R018

- Mục tiêu: bật residual-adaptive prior trên toaster với cùng cấu hình R017 để đo riêng tác động của adaptive prior.
- Output: [output/backups/toaster/spec-fastgs_v3_new_architecture_20260709_074103][R018-output]
- Config: scene=toaster, images=images, iterations=30000, asg_degree=24, use_ref_score=True, use_adaptive_prior=True, adaptive_prior_start=5000, adaptive_prior_interval=3000, adaptive_prior_num_cameras=20, adaptive_prior_ema=0.70, ref_prior_method=shafer, sk_intensity=0.65, sk_saturation=0.30.
- Kết quả đáng chú ý: PSNR=22.1166, SSIM=0.8931, LPIPS=0.1192; Spec_PSNR=18.2814, ASG_Energy_In_Residual=0.8330, ASG_Residual_IoU=0.6104.
- Training: initial_gaussians=100000, final_gaussians=35142, peak_vram=3195.47 MiB, time=7m04s.
- Nhận xét: so với R017, adaptive prior tăng PSNR +0.0294, giảm LPIPS -0.0004, tăng Spec_PSNR +0.1616 và tăng ASG_IoU +0.0094; SSIM giảm nhẹ.
- Kết luận: trên toaster/asg24, residual-adaptive prior là chỉnh sửa có lợi nhất quán cho specular proxy và alignment, dù chưa vượt R016/R024 về metric tổng thể.

### R019

- Mục tiêu: tạo mốc `use_adaptive_prior=False` cho counter với ASG64, auto RefScore budget và soft decay.
- Output: [output/backups/counter/spec-fastgs_v3_new_architecture_20260709_075930][R019-output]
- Config: scene=counter, images=images_8, iterations=30000, asg_degree=64, use_ref_score=True, use_adaptive_prior=False, ref_prior_method=shafer, sk_intensity=0.65, sk_saturation=0.30, max_refscore_gaussians=1000000, f_rest_interval=16/32/64.
- Kết quả đáng chú ý: PSNR=30.8666, SSIM=0.9394, LPIPS=0.0633; Spec_PSNR=23.3781, ASG_Energy_In_Residual=0.5238, ASG_Residual_IoU=0.2369.
- Training: initial_gaussians=155767, final_gaussians=175572, peak_vram=1544.43 MiB, time=15m57s.
- Nhận xét: so với R012, IoU cao hơn nhưng PSNR/SSIM/Spec_PSNR thấp hơn; các chỉnh sửa mới có thể đang tạo coverage rộng hơn nhưng chưa đủ đúng vị trí nếu không có adaptive prior.
- Kết luận: R019 là baseline off quan trọng cho counter; metric cho thấy auto-budget/decay không nên đánh giá tách rời adaptive prior.

### R020

- Mục tiêu: bật residual-adaptive prior cho counter với cùng thiết lập R019.
- Output: [output/counter][R020-output]
- Config: scene=counter, images=images_8, iterations=30000, asg_degree=64, use_ref_score=True, use_adaptive_prior=True, adaptive_prior_start=5000, adaptive_prior_interval=3000, adaptive_prior_num_cameras=20, adaptive_prior_ema=0.70, ref_prior_method=shafer, sk_intensity=0.65, sk_saturation=0.30.
- Kết quả đáng chú ý: PSNR=30.9569, SSIM=0.9403, LPIPS=0.0632; Spec_PSNR=23.4845, ASG_Energy_In_Residual=0.5150, ASG_Residual_IoU=0.2361.
- Training: initial_gaussians=155767, final_gaussians=178464, peak_vram=1550.84 MiB, time=16m12s.
- Nhận xét: so với R019, adaptive prior tăng PSNR +0.0903, SSIM +0.0009, giảm LPIPS -0.0002 và tăng Spec_PSNR +0.1064; IoU giảm rất nhẹ -0.0008 nhưng vẫn cao hơn R012.
- Kết luận: R020 là best counter hiện tại theo PSNR, SSIM, LPIPS và Spec_PSNR. Đây là bằng chứng tốt nhất rằng residual-adaptive prior giải quyết được phần Geometry Coverage trên counter.

### R021

- Mục tiêu: kiểm tra toaster với adaptive prior và ASG64.
- Output: [output/backups/toaster/spec-fastgs_v3_new_architecture_20260709_083322][R021-output]
- Config: scene=toaster, images=images, iterations=30000, asg_degree=64, use_ref_score=True, use_adaptive_prior=True, ref_prior_method=shafer, sk_intensity=0.65, sk_saturation=0.30.
- Kết quả đáng chú ý: PSNR=22.0331, SSIM=0.8953, LPIPS=0.1171; Spec_PSNR=18.1242, ASG_Energy_In_Residual=0.8184, ASG_Residual_IoU=0.5851.
- Training: initial_gaussians=100000, final_gaussians=34638, peak_vram=3241.24 MiB, time=7m21s.
- Nhận xét: R021 tốt nhất nhóm Shafer-adaptive theo SSIM/LPIPS nhưng thấp về PSNR, Spec_PSNR và IoU; khác với R015, adaptive prior làm LPIPS tốt hơn nhưng IoU không còn tăng theo ASG degree.
- Kết luận: ASG64 + adaptive prior có lợi perceptual, nhưng không phải lựa chọn tốt nếu mục tiêu là specular localization proxy.

### R022

- Mục tiêu: kiểm tra lower-capacity ASG12 với adaptive prior để xem ASG capacity tối thiểu có đủ không.
- Output: [output/backups/toaster/spec-fastgs_v3_new_architecture_20260709_084600][R022-output]
- Config: scene=toaster, images=images, iterations=30000, asg_degree=12, use_ref_score=True, use_adaptive_prior=True, ref_prior_method=shafer, sk_intensity=0.65, sk_saturation=0.30.
- Kết quả đáng chú ý: PSNR=21.8971, SSIM=0.8917, LPIPS=0.1208; Spec_PSNR=18.1607, ASG_Energy_In_Residual=0.7925, ASG_Residual_IoU=0.5367.
- Training: initial_gaussians=100000, final_gaussians=34435, peak_vram=3185.15 MiB, time=7m01s.
- Nhận xét: ASG12 giảm mạnh PSNR, SSIM, LPIPS và IoU; adaptive prior không bù được thiếu capacity biểu diễn.
- Kết luận: không nên chọn ASG12 cho toaster; ASG capacity quá thấp làm mất khả năng giải thích highlight.

### R023

- Mục tiêu: kiểm tra điểm giữa ASG32 với adaptive prior và Shafer 0.65/0.30.
- Output: [output/backups/toaster/spec-fastgs_v3_new_architecture_20260709_090531][R023-output]
- Config: scene=toaster, images=images, iterations=30000, asg_degree=32, use_ref_score=True, use_adaptive_prior=True, ref_prior_method=shafer, sk_intensity=0.65, sk_saturation=0.30.
- Kết quả đáng chú ý: PSNR=22.0221, SSIM=0.8935, LPIPS=0.1190; Spec_PSNR=18.3092, ASG_Energy_In_Residual=0.8186, ASG_Residual_IoU=0.5881.
- Training: initial_gaussians=100000, final_gaussians=34553, peak_vram=3199.92 MiB, time=7m03s.
- Nhận xét: R023 có Spec_PSNR cao nhất trong nhóm Shafer-adaptive toaster, nhưng PSNR/SSIM/LPIPS thấp hơn R018 và thấp hơn rõ R024.
- Kết luận: ASG32 + Shafer-adaptive đáng giữ nếu ưu tiên Spec_PSNR, nhưng chưa phải cấu hình toaster tốt nhất.

### R024

- Mục tiêu: thử Tan-Ikeuchi với adaptive prior và ASG32 trên toaster, sau các run Shafer-adaptive.
- Output: [output/toaster][R024-output]
- Config: scene=toaster, images=images, iterations=30000, asg_degree=32, use_ref_score=True, use_adaptive_prior=True, ref_prior_method=tan, ti_thresh=0.35, ti_bright=0.60.
- Kết quả đáng chú ý: PSNR=22.1927, SSIM=0.8957, LPIPS=0.1162; Spec_PSNR=18.2631, ASG_Energy_In_Residual=0.8362, ASG_Residual_IoU=0.6138.
- Training: initial_gaussians=100000, final_gaussians=35240, peak_vram=3205.86 MiB, time=7m05s.
- Nhận xét: đây là best toaster hiện tại theo PSNR, SSIM, LPIPS và giữ IoU cao. Spec_PSNR thấp hơn R016/R008/R023 một chút, nhưng tổng thể cân bằng hơn.
- Kết luận: Tan + adaptive prior + ASG32 là cấu hình toaster mạnh nhất hiện tại nếu cần một lựa chọn chung cho thesis.

## Conclusion

- Thời điểm tổng kết: 2026-07-09 10:55 ICT.
- Runs được dùng: R001-R024.
- Dataset: Mip-NeRF360/counter và Ref-NeRF/toaster.
- Metric chính: PSNR, SSIM, LPIPS, Spec_PSNR, ASG_Residual_IoU, ASG_Energy_In_Residual.
- Specular error contribution được ước lượng bằng mask proxy `residual_real > 0.02`, với công thức `L1 error trong spec_mask / L1 error toàn ảnh`.

### Tổng quan specular error

| Dataset | Specular mask area | Specular error contribution | Nhận xét |
| --- | ---: | ---: | --- |
| Mip-NeRF360/counter | 10-11% pixel | 30-32% tổng L1 error | Vùng specular nhỏ nhưng tạo khoảng 1/3 lỗi reconstruction. |
| Ref-NeRF/toaster | 19-20% pixel | 46-50% tổng L1 error | Gần một nửa lỗi đến từ vùng specular. |

Kết luận chính vẫn giữ nguyên: bài toán hiện tại thật sự là specular-dominated. Dù vùng specular không chiếm đa số pixel, nó đóng góp sai số lớn bất cân xứng so với diện tích. Vì vậy các ablation về reflection prior, ref-score guided densification, adaptive prior và ASG capacity là trực tiếp liên quan tới phần lỗi quan trọng nhất.

### Impact của từng ablation

- `use_ref_score`: trên Mip-NeRF360/counter, bật ref-score có lợi rõ. R001 tốt hơn R002 về PSNR, LPIPS, Spec_PSNR và ASG_IoU. Trên Ref-NeRF/toaster, hiệu ứng không ổn định; một số run `use_ref_score=False` vẫn tốt hơn run `True` theo PSNR/Spec_PSNR.
- Tan vs Shafer/Klinker: trên counter, R020 Shafer + adaptive prior hiện vượt R001 Tan về PSNR/SSIM/LPIPS và gần Spec_PSNR nhất, nhưng R001 vẫn là mốc Tan mạnh nếu không dùng adaptive prior. Trên toaster, kết quả mới đảo chiều so với nhóm R007-R016: R024 Tan + adaptive prior + ASG32 là best tổng thể, dù Shafer vẫn có vài run mạnh theo Spec_PSNR.
- Shafer/Klinker threshold: `0.65/0.30` là cân bằng nhất. `0.70/0.20` cho ASG_IoU cao nhưng PSNR thấp, còn `0.60/0.30` rộng hơn nhưng không vượt `0.65/0.30`.
- ASG degree trên counter: tăng ASG degree từ 32 -> 48 -> 64 cải thiện rõ trong nhóm Shafer cũ, và ASG64 vẫn là lựa chọn tốt nhất cho counter sau khi thêm adaptive prior.
- ASG degree trên toaster: tăng ASG degree tạo trade-off. Không adaptive thì R016 `asg=24` tốt nhất theo PSNR, R015 `asg=64` tốt theo LPIPS/IoU. Có adaptive thì ASG12 là quá thấp, ASG64 tốt LPIPS nhưng giảm IoU, còn ASG32 với Tan ở R024 cân bằng nhất.
- Scene-relative RefScore budget và soft RefScore decay: hai chỉnh sửa này làm budget tự scale theo số Gaussian ban đầu và giảm độ mạnh của prior khi scene đã nhiều Gaussian hơn. Tuy nhiên, R017/R019 cho thấy khi tắt adaptive prior, chúng chưa tự tạo ra cải thiện chắc chắn; chúng nên được xem là hạ tầng để RefScore ổn định hơn, không phải nguồn gain chính.
- Residual-adaptive prior: đây là chỉnh sửa mới có bằng chứng tích cực nhất. R018 tốt hơn R017 trên toaster/asg24 về PSNR, LPIPS, Spec_PSNR và ASG_IoU; R020 tốt hơn R019 trên counter/asg64 về PSNR, SSIM, LPIPS và Spec_PSNR. Tác động chính là cập nhật prior theo residual còn lại của model, nên nó bám sát lỗi geometric/specular thực tế hơn static prior.
- Sparse `f_rest` schedule: các run R017-R024 đều dùng `f_rest_interval=16/32/64`, giúp giảm cập nhật SH residual khi training đi xa hơn và nhường vai trò giải thích phần view-dependent cho ASG. Hiện chưa có ablation off riêng cho schedule này, nên chỉ nên kết luận là nó tương thích với cấu hình best mới, chưa khẳng định nó là nguồn gain độc lập.

### Best runs hiện tại

| Mục tiêu | Run | Lý do |
| --- | --- | --- |
| Counter - metric tổng thể | R020 | Best PSNR/SSIM/LPIPS/Spec_PSNR trong nhóm counter; IoU gần best. |
| Counter - ASG alignment proxy | R019/R020 | R019 nhỉnh hơn IoU rất nhẹ, nhưng R020 cân bằng hơn rõ theo fidelity. |
| Toaster - metric tổng thể | R024 | Best PSNR/SSIM/LPIPS trong toàn bộ nhóm toaster và IoU cao. |
| Toaster - Shafer Spec_PSNR | R016/R023 | R016 cao nhất trong nhóm Shafer cũ; R023 cao nhất trong nhóm Shafer-adaptive. |
| Toaster - low-capacity negative case | R022 | ASG12 tụt rõ, hữu ích để chứng minh capacity quá thấp không đủ. |

### Nhận xét về chỉnh sửa Geometry Coverage mới

Các chỉnh sửa mới đi đúng hướng, nhưng gain không đến từ một công tắc đơn lẻ. Auto budget và soft decay giúp RefScore bớt phụ thuộc scene size, còn adaptive prior mới là phần làm metric nhảy rõ nhất vì nó tái căn chỉnh prior theo residual của model trong quá trình train. Trên counter, R020 là bằng chứng rất mạnh: vừa vượt R012 cũ về PSNR/SSIM/LPIPS/Spec_PSNR, vừa giữ ASG_IoU cao hơn R012. Trên toaster, R024 cho thấy adaptive prior kết hợp Tan-Ikeuchi và ASG32 tạo cấu hình cân bằng hơn Shafer-adaptive.

Điểm cần cẩn thận khi viết thesis: Geometry Coverage không đơn giản là "tăng IoU là tốt". R019 có IoU nhỉnh hơn R020 nhưng fidelity thấp hơn; R021 có LPIPS tốt nhưng IoU giảm; R024 thắng tổng thể dù Spec_PSNR không cao nhất. Vì vậy phần kết quả nên trình bày theo trade-off: coverage tốt phải đi kèm PSNR/LPIPS/Spec_PSNR, không chỉ một proxy mask.

### Kết luận cho thesis

Các kết quả cho thấy cải thiện specular reconstruction là hướng hợp lý vì specular chiếm khoảng 30-50% tổng sai số L1 tùy dataset. Sau các run mới, kết luận nên mạnh hơn: residual-adaptive prior là bổ sung đáng giữ trong kiến trúc mới vì nó cải thiện counter rõ rệt và giúp toaster đạt best tổng thể khi kết hợp với prior phù hợp. Tuy nhiên, không có một cấu hình thắng tuyệt đối cho mọi metric. Counter hợp với ASG64 + Shafer + adaptive prior, trong khi toaster hiện hợp nhất với ASG32 + Tan + adaptive prior. Luận văn nên nhấn mạnh adaptive geometry coverage là cơ chế giảm lệch giữa static reflection prior và lỗi còn lại của model, còn lựa chọn Tan/Shafer và ASG degree vẫn cần tùy scene.

[R001-output]: output/backups/counter/spec-fastgs_v3_new_architecture_20260707_105338
[R002-output]: output/backups/counter/spec-fastgs_v3_new_architecture_20260707_124037
[R003-output]: output/backups/toaster/spec-fastgs_v3_new_architecture_20260707_122528
[R004-output]: output/backups/toaster/spec-fastgs_v3_new_architecture_20260707_125351
[R005-output]: output/backups/toaster/spec-fastgs_v3_new_architecture_20260707_131110
[R006-output]: output/backups/toaster/spec-fastgs_v3_new_architecture_20260707_132623
[R007-output]: output/backups/toaster/spec-fastgs_v3_new_architecture_20260707_140203
[R008-output]: output/backups/toaster/spec-fastgs_v3_new_architecture_20260707_141652
[R009-output]: output/backups/toaster/spec-fastgs_v3_new_architecture_20260707_143226
[R010-output]: output/backups/counter/spec-fastgs_v3_new_architecture_20260707_145020
[R011-output]: output/backups/counter/spec-fastgs_v3_new_architecture_20260707_152830
[R012-output]: output/backups/counter/spec-fastgs_v3_new_architecture_20260707_154715
[R013-output]: output/backups/toaster/spec-fastgs_v3_new_architecture_20260708_155436
[R014-output]: output/backups/toaster/spec-fastgs_v3_new_architecture_20260708_161615
[R015-output]: output/backups/toaster/spec-fastgs_v3_new_architecture_20260708_162950
[R016-output]: output/backups/toaster/spec-fastgs_v3_new_architecture_20260708_164439
[R017-output]: output/backups/toaster/spec-fastgs_v3_new_architecture_20260709_072529
[R018-output]: output/backups/toaster/spec-fastgs_v3_new_architecture_20260709_074103
[R019-output]: output/backups/counter/spec-fastgs_v3_new_architecture_20260709_075930
[R020-output]: output/counter
[R021-output]: output/backups/toaster/spec-fastgs_v3_new_architecture_20260709_083322
[R022-output]: output/backups/toaster/spec-fastgs_v3_new_architecture_20260709_084600
[R023-output]: output/backups/toaster/spec-fastgs_v3_new_architecture_20260709_090531
[R024-output]: output/toaster
