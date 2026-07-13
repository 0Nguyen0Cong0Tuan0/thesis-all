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
| R020 | 2026-07-09 08:16:36 | counter | [output/backups/counter/spec-fastgs_v3_new_architecture_20260709_081711][R020-output] | shafer, adaptive=True, asg=64, auto-budget | 30.9569 | 0.9403 | 0.0632 | 23.4845 | 0.2361 | 178464 | 16m 12s | Best counter Geometry Coverage theo PSNR/SSIM/LPIPS/Spec_PSNR |
| R025 | 2026-07-10 14:01:27 | counter | [output/backups/counter/spec-fastgs_v3_new_architecture_20260710_140202][R025-output] | tan, adaptive=True, asg=64, repr-default, no SH mask | 30.9953 | 0.9403 | 0.0627 | 23.4326 | 0.2199 | 178529 | n/a | Baseline Stage 2 no-op; best counter theo PSNR/LPIPS nhưng IoU thấp hơn R020 |
| R026 | 2026-07-10 14:22:12 | counter | [output/backups/counter/spec-fastgs_v3_new_architecture_20260710_142249][R026-output] | tan, adaptive=True, asg=64, SH mask on | 30.6901 | 0.9385 | 0.0646 | 23.4960 | 0.2781 | 175854 | n/a | SH mask tăng mạnh IoU/Spec_PSNR nhưng làm giảm fidelity tổng thể |
| R031 | 2026-07-11 13:23:22 | counter | [output/backups/counter/spec-fastgs_v3_new_architecture_20260711_132401][R031-output] | tan, adaptive=True, asg=64, SH mask, spec-L1=1.0 | 30.7784 | 0.9386 | 0.0646 | 23.5796 | 0.2727 | 176705 | 17m 43s | Spec-weighted L1 cải thiện PSNR/Spec_PSNR so với R026, IoU giảm nhẹ |
| R032 | 2026-07-11 13:43:14 | counter | [output/backups/counter/spec-fastgs_v3_new_architecture_20260711_134351][R032-output] | tan, adaptive=True, asg=64, SH mask, normal-delta only | 30.7402 | 0.9387 | 0.0649 | 23.5041 | 0.2741 | 175797 | 17m 30s | Normal-delta không học trên real branch khi reflection-dir tắt |
| R033 | 2026-07-11 14:01:48 | counter | [output/backups/counter/spec-fastgs_v3_new_architecture_20260711_140224][R033-output] | tan, adaptive=True, asg=64, SH mask, spec-L1=1.0, normal-delta | 30.6800 | 0.9382 | 0.0652 | 23.5915 | 0.2753 | 176952 | 17m 44s | Spec_PSNR cao nhất counter hiện tại nhưng fidelity thấp |
| R034 | 2026-07-11 14:28:39 | counter | [output/backups/counter/spec-fastgs_v3_new_architecture_20260711_142913][R034-output] | tan, adaptive=True, asg=64, SH mask, spec-L1=1.0, spec-reg=0.1 | 30.6401 | 0.9383 | 0.0656 | 23.2331 | 0.2446 | 177584 | 17m 35s | Spec-reg=0.1 quá mạnh, làm giảm cả fidelity và spec proxy |
| R035 | 2026-07-11 14:49:33 | counter | [output/backups/counter/spec-fastgs_v3_new_architecture_20260711_145011][R035-output] | tan, adaptive=True, asg=64, SH mask, spec-L1=1.0, spec-reg=0.001, normal-delta | 30.6136 | 0.9377 | 0.0657 | 23.4578 | 0.2734 | 177469 | 17m 53s | Spec-reg=0.001 không cải thiện counter trong run này |
| R036 | 2026-07-11 15:47:21 | counter | [output/backups/counter/spec-fastgs_v3_new_architecture_20260711_154800][R036-output] | tan, adaptive=True, asg=64, real-refl, SH mask, spec-L1=1.0, spec-reg=0.0001, normal-delta | 30.5517 | 0.9376 | 0.0656 | 23.4433 | 0.2831 | 177426 | 18m 16s | Normal-delta học được khi bật real reflection, nhưng fidelity/Spec_PSNR thấp |
| R037 | 2026-07-11 16:07:18 | counter | [output/backups/counter/spec-fastgs_v3_new_architecture_20260711_160755][R037-output] | tan, adaptive=True, asg=64, real-refl, no SH mask, spec-L1=1.0, spec-reg=0.0001, no normal-delta | 30.7709 | 0.9389 | 0.0641 | 23.3843 | 0.2145 | 181545 | 17m 44s | Real reflection + weak supervision không vượt R025/R031 |
| R038 | 2026-07-11 18:50:04 | counter | [output/backups/counter/spec-fastgs_v3_new_architecture_20260711_185043][R038-output] | tan, adaptive=True, asg=64, no SH mask, no supervision/normal | 30.9733 | 0.9404 | 0.0635 | 23.6029 | 0.2383 | 179295 | 17m 6s | Rerun no-mask rất mạnh theo Spec_PSNR, nhưng LPIPS thấp hơn R025 |
| R039 | 2026-07-11 19:47:24 | counter | [output/backups/counter/spec-fastgs_v3_new_architecture_20260711_194804][R039-output] | tan, adaptive=True, asg=64, SH mask intended, start=80000, inactive | 30.9077 | 0.9399 | 0.0634 | 23.4283 | 0.2262 | 179101 | 17m 7s | Mask không active do start > 30000; không dùng để kết luận soft mask |
| R040 | 2026-07-11 20:21:26 | counter | [output/backups/counter/spec-fastgs_v3_new_architecture_20260711_202207][R040-output] | tan, adaptive=True, asg=64, soft SH mask, scale=0.5, start=8000 | 30.9363 | 0.9401 | 0.0631 | 23.4298 | 0.2430 | 178943 | 17m 28s | Soft mask active, IoU tăng nhẹ nhưng Spec_PSNR thấp hơn no-mask R038 |
| R041 | 2026-07-11 20:52:04 | counter | [output/backups/counter/spec-fastgs_v3_new_architecture_20260711_205247][R041-output] | tan, adaptive=True, asg=64, soft SH mask, scale=0.5, spec-L1=0.5 | 30.8887 | 0.9398 | 0.0636 | 23.3650 | 0.2323 | 179895 | 18m 1s | Giảm cả fidelity và spec proxy; weighted L1=0.5 không giúp khi đi cùng soft mask |
| R042 | 2026-07-11 22:03:26 | counter | [output/backups/counter/spec-fastgs_v3_new_architecture_20260711_220403][R042-output] | tan, adaptive=True, asg=64, ASG residual only, no SH mask | 30.8979 | 0.9400 | 0.0640 | 23.4321 | 0.2346 | 178522 | 17m 37s | Residual loss có chạy nhưng không vượt no-mask R038 |
| R043 | 2026-07-11 22:24:31 | counter | [output/backups/counter/spec-fastgs_v3_new_architecture_20260711_222509][R043-output] | tan, adaptive=True, asg=64, soft SH mask + ASG residual | 30.8528 | 0.9398 | 0.0630 | 23.4023 | 0.2489 | 178627 | 17m 37s | IoU tăng nhẹ nhất nhóm soft/residual, nhưng PSNR và Spec_PSNR vẫn giảm |
| R044 | 2026-07-11 23:00:56 | counter | [output/backups/counter/spec-fastgs_v3_new_architecture_20260711_230133][R044-output] | tan, adaptive=True, asg=64, normal-delta + smooth, no ref-mask | 30.8517 | 0.9397 | 0.0636 | 23.1791 | 0.2167 | 179473 | 17m 43s | Normal-delta học nhưng giảm rõ PSNR/Spec_PSNR/IoU |
| R045 | 2026-07-11 23:20:28 | counter | [output/backups/counter/spec-fastgs_v3_new_architecture_20260711_232105][R045-output] | tan, adaptive=True, asg=64, normal-delta + smooth ref-mask | 30.6906 | 0.9387 | 0.0646 | 23.0685 | 0.1920 | 179400 | 17m 56s | Ref-mask smooth làm metric tụt mạnh hơn; không nên dùng default |
| R046 | 2026-07-11 23:41:58 | counter | [output/backups/counter/spec-fastgs_v3_new_architecture_20260711_234236][R046-output] | tan, adaptive=True, asg=64, real-refl, no normal-delta, smooth global | 30.7059 | 0.9385 | 0.0651 | 23.1551 | 0.2003 | 177867 | 17m 37s | Chỉ bật real reflection đã làm fidelity giảm mạnh; smooth không cứu được |
| R047 | 2026-07-12 00:02:37 | counter | [output/backups/counter/spec-fastgs_v3_new_architecture_20260712_000315][R047-output] | tan, adaptive=True, asg=64, no real-refl, no normal-delta, smooth global inactive | 30.9625 | 0.9402 | 0.0629 | 23.4922 | 0.2215 | 179364 | 17m 37s | Quay lại no real-reflection gần baseline; metric phục hồi rõ |
| R048 | 2026-07-12 16:39:41 | counter | [output/backups/counter/spec-fastgs_v3_new_architecture_20260712_164019][R048-output] | tan, adaptive=True, asg=64, no real-refl, no normal-delta, smooth global inactive | 30.9136 | 0.9402 | 0.0626 | 23.3579 | 0.2318 | 179615 | 17m 0s | Rerun baseline-like; PSNR thấp hơn R047 nhưng LPIPS tốt |
| R049 | 2026-07-12 17:24:27 | counter | [output/backups/counter/spec-fastgs_v3_new_architecture_20260712_172509][R049-output] | tan, adaptive=True, asg=64, no real-refl, all normal/supervision off | 30.9167 | 0.9401 | 0.0635 | 23.5190 | 0.2463 | 179284 | 17m 12s | Baseline sạch sau ref-score confidence; không đạt lại R025 do variance |
| R050 | 2026-07-12 17:53:37 | counter | [output/backups/counter/spec-fastgs_v3_new_architecture_20260712_175418][R050-output] | tan, adaptive=True, asg=64, real-refl only, no normal-delta/smooth | 30.7710 | 0.9390 | 0.0641 | 23.1857 | 0.2038 | 178024 | 17m 24s | Real reflection direction riêng lẻ vẫn kéo thấp PSNR/Spec_PSNR |
| R051 | 2026-07-12 18:19:45 | counter | [output/backups/counter/spec-fastgs_v3_new_architecture_20260712_182023][R051-output] | tan, adaptive=True, asg=64, real-refl, normal-delta, smooth ref-mask nhẹ | 30.7766 | 0.9393 | 0.0646 | 23.1830 | 0.2029 | 177597 | 17m 45s | Normal-delta học nhưng không cải thiện so với real-refl only |
| R052 | 2026-07-12 18:53:55 | counter | [output/backups/counter/spec-fastgs_v3_new_architecture_20260712_185435][R052-output] | tan, adaptive=True, asg=64, SH mask scale=0.75, no ASG residual | 30.9680 | 0.9402 | 0.0629 | 23.6289 | 0.2384 | 179620 | 17m 36s | SH mask mới hoạt động tốt nhất trong nhóm gần đây theo PSNR/Spec_PSNR |
| R053 | 2026-07-12 19:21:43 | counter | [output/backups/counter/spec-fastgs_v3_new_architecture_20260712_192226][R053-output] | tan, adaptive=True, asg=64, SH mask + ASG residual 0.01/leak 0.001 | 30.8989 | 0.9400 | 0.0637 | 23.4385 | 0.2368 | 179091 | 18m 25s | ASG residual active nhưng làm giảm PSNR/Spec_PSNR so với R052 |
| R054 | 2026-07-12 19:52:03 | counter | [output/counter][R054-output] | tan, adaptive=True, asg=64, SH mask + ASG residual nhẹ 0.003/leak 0.0005 | 30.9337 | 0.9399 | 0.0637 | 23.4247 | 0.2485 | 178605 | 17m 46s | Residual nhẹ tăng IoU so với R052 nhưng vẫn giảm fidelity/spec PSNR |

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
| R024 | 2026-07-09 10:43:07 | toaster | [output/backups/toaster/spec-fastgs_v3_new_architecture_20260709_104803][R024-output] | tan, adaptive=True, asg=32, auto-budget | 22.1927 | 0.8957 | 0.1162 | 18.2631 | 0.6138 | 35240 | 7m 5s | Best toaster Geometry Coverage theo PSNR/SSIM/LPIPS và IoU tốt |
| R027 | 2026-07-10 13:24:50 | toaster | [output/backups/toaster/spec-fastgs_v3_new_architecture_20260710_133006][R027-output] | tan, adaptive=True, asg=32, repr-default, no SH mask | 22.0400 | 0.8946 | 0.1180 | 18.1001 | 0.6109 | 35080 | n/a | Baseline Stage 2 no-op sau pipeline mới; thấp hơn R024, có thể chịu dao động/rerun prior |
| R028 | 2026-07-10 13:39:16 | toaster | [output/backups/toaster/spec-fastgs_v3_new_architecture_20260710_134433][R028-output] | tan, adaptive=True, asg=32, SH mask on | 22.1220 | 0.8948 | 0.1194 | 18.3822 | 0.6413 | 34577 | n/a | SH mask cải thiện Spec_PSNR/IoU rõ so với R027 nhưng chưa vượt R024 về fidelity tổng thể |
| R029 | 2026-07-10 23:37:11 | toaster | [output/backups/toaster/spec-fastgs_v3_new_architecture_20260710_234233][R029-output] | tan, adaptive=True, asg=32, SH mask, spec-L1=1.0, spec-reg=0.001 | 22.0170 | 0.8945 | 0.1196 | 18.3851 | 0.6564 | 36458 | 8m 14s | Spec supervision tăng IoU cao nhất toaster nhưng giảm PSNR/LPIPS |
| R030 | 2026-07-10 23:51:08 | toaster | [output/toaster][R030-output] | tan, adaptive=True, asg=32, SH mask, spec-L1=1.0, spec-reg=0.001, normal-delta | 22.0036 | 0.8942 | 0.1196 | 18.4302 | 0.6359 | 36407 | 8m 11s | Normal-delta tăng Spec_PSNR nhưng giảm IoU/fidelity so với R029 |

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
- Output: [output/backups/counter/spec-fastgs_v3_new_architecture_20260709_081711][R020-output]
- Config: scene=counter, images=images_8, iterations=30000, asg_degree=64, use_ref_score=True, use_adaptive_prior=True, adaptive_prior_start=5000, adaptive_prior_interval=3000, adaptive_prior_num_cameras=20, adaptive_prior_ema=0.70, ref_prior_method=shafer, sk_intensity=0.65, sk_saturation=0.30.
- Kết quả đáng chú ý: PSNR=30.9569, SSIM=0.9403, LPIPS=0.0632; Spec_PSNR=23.4845, ASG_Energy_In_Residual=0.5150, ASG_Residual_IoU=0.2361.
- Training: initial_gaussians=155767, final_gaussians=178464, peak_vram=1550.84 MiB, time=16m12s.
- Nhận xét: so với R019, adaptive prior tăng PSNR +0.0903, SSIM +0.0009, giảm LPIPS -0.0002 và tăng Spec_PSNR +0.1064; IoU giảm rất nhẹ -0.0008 nhưng vẫn cao hơn R012.
- Kết luận: R020 là best counter của phase Geometry Coverage trước Stage 2 theo PSNR, SSIM, LPIPS và Spec_PSNR. Đây là bằng chứng tốt nhất rằng residual-adaptive prior giải quyết được phần Geometry Coverage trên counter.

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
- Output: [output/backups/toaster/spec-fastgs_v3_new_architecture_20260709_104803][R024-output]
- Config: scene=toaster, images=images, iterations=30000, asg_degree=32, use_ref_score=True, use_adaptive_prior=True, ref_prior_method=tan, ti_thresh=0.35, ti_bright=0.60.
- Kết quả đáng chú ý: PSNR=22.1927, SSIM=0.8957, LPIPS=0.1162; Spec_PSNR=18.2631, ASG_Energy_In_Residual=0.8362, ASG_Residual_IoU=0.6138.
- Training: initial_gaussians=100000, final_gaussians=35240, peak_vram=3205.86 MiB, time=7m05s.
- Nhận xét: đây là best toaster hiện tại theo PSNR, SSIM, LPIPS và giữ IoU cao. Spec_PSNR thấp hơn R016/R008/R023 một chút, nhưng tổng thể cân bằng hơn.
- Kết luận: Tan + adaptive prior + ASG32 là cấu hình toaster mạnh nhất hiện tại nếu cần một lựa chọn chung cho thesis.

### R025

- Mục tiêu: tạo baseline Stage 2 Representation Capability trên counter với các knob mới ở trạng thái no-op, dùng cấu hình Tan + adaptive prior + ASG64.
- Output: [output/backups/counter/spec-fastgs_v3_new_architecture_20260710_140202][R025-output]
- Config: scene=counter, images=images_8, iterations=30000, asg_degree=64, use_ref_score=True, use_adaptive_prior=True, ref_prior_method=tan, ti_thresh=0.35, ti_bright=0.60, use_sh_spec_mask=False, asg_num_theta/asg_num_phi/specular_hidden/specular_layers=-1, lambda_spec_l1_weight=0.0, lambda_spec_reg=0.0.
- Kết quả đáng chú ý: PSNR=30.9953, SSIM=0.9403, LPIPS=0.0627; Spec_PSNR=23.4326, NonSpec_PSNR=34.1912, ASG_Energy_In_Residual=0.5076, ASG_Residual_IoU=0.2199.
- Training: initial_gaussians=155767, final_gaussians=178529, avg_asg_eval_count=116925.17, avg_sh_spec_mask_ratio=0.0000.
- Nhận xét: so với R020, R025 tăng PSNR +0.0384 và giảm LPIPS -0.0005, nhưng Spec_PSNR giảm -0.0519 và ASG_IoU giảm -0.0162. Do R025 đổi prior về Tan nên đây là baseline Stage 2 hữu ích, nhưng không phải so sánh thuần với R020 Shafer.
- Kết luận: R025 là best counter theo PSNR/LPIPS hiện tại, nhưng ASG alignment yếu hơn R020; nên dùng làm mốc `SH mask off` cho cặp R025/R026.

### R026

- Mục tiêu: bật SH/spec role separation trên counter bằng `use_sh_spec_mask=True` để ép SH residual giảm vai trò ở vùng ref-score cao.
- Output: [output/backups/counter/spec-fastgs_v3_new_architecture_20260710_142249][R026-output]
- Config: giống R025 nhưng use_sh_spec_mask=True, sh_spec_mask_threshold=0.70, sh_spec_grad_scale=0.0, sh_spec_mask_start=3000, sh_spec_min_metric_count=1; lambda_spec_l1_weight=0.0, lambda_spec_reg=0.0.
- Kết quả đáng chú ý: PSNR=30.6901, SSIM=0.9385, LPIPS=0.0646; Spec_PSNR=23.4960, NonSpec_PSNR=34.1480, ASG_Energy_In_Residual=0.5886, ASG_Residual_IoU=0.2781.
- Training: initial_gaussians=155767, final_gaussians=175854, avg_asg_eval_count=117676.83, avg_sh_spec_mask_ratio=0.0382.
- Nhận xét: so với R025, SH mask tăng Spec_PSNR +0.0635 và ASG_IoU +0.0583, nhưng PSNR giảm -0.3052, SSIM giảm -0.0019 và LPIPS tăng +0.0019. Tín hiệu này khá rõ: role separation đẩy năng lượng specular sang ASG, nhưng đang quá mạnh với counter nếu dùng scale=0.0.
- Kết luận: R026 là bằng chứng tốt cho hypothesis Representation Capability/role separation theo proxy ASG alignment, nhưng chưa phải cấu hình nên chọn nếu ưu tiên fidelity tổng thể.

### R027

- Mục tiêu: tạo baseline Stage 2 Representation Capability trên toaster với các knob mới ở trạng thái no-op, dùng cấu hình Tan + adaptive prior + ASG32.
- Output: [output/backups/toaster/spec-fastgs_v3_new_architecture_20260710_133006][R027-output]
- Config: scene=toaster, images=images, iterations=30000, asg_degree=32, use_ref_score=True, use_adaptive_prior=True, ref_prior_method=tan, ti_thresh=0.35, ti_bright=0.60, use_sh_spec_mask=False, asg_num_theta/asg_num_phi/specular_hidden/specular_layers=-1, lambda_spec_l1_weight=0.0, lambda_spec_reg=0.0.
- Kết quả đáng chú ý: PSNR=22.0400, SSIM=0.8946, LPIPS=0.1180; Spec_PSNR=18.1001, NonSpec_PSNR=24.2159, ASG_Energy_In_Residual=0.8333, ASG_Residual_IoU=0.6109.
- Training: initial_gaussians=100000, final_gaussians=35080, avg_asg_eval_count=39723.15, avg_sh_spec_mask_ratio=0.0000.
- Nhận xét: R027 thấp hơn R024 khá rõ dù config chính giống nhau ở mức Stage 2 no-op. Đây có thể là run variance hoặc khác biệt do rerun/regenerate prior, nên không nên dùng R027 để phủ định R024.
- Kết luận: R027 chủ yếu là mốc `SH mask off` gần nhất cho cặp R027/R028.

### R028

- Mục tiêu: bật SH/spec role separation trên toaster bằng `use_sh_spec_mask=True`.
- Output: [output/backups/toaster/spec-fastgs_v3_new_architecture_20260710_134433][R028-output]
- Config: giống R027 nhưng use_sh_spec_mask=True, sh_spec_mask_threshold=0.70, sh_spec_grad_scale=0.0, sh_spec_mask_start=3000, sh_spec_min_metric_count=1; lambda_spec_l1_weight=0.0, lambda_spec_reg=0.0.
- Kết quả đáng chú ý: PSNR=22.1220, SSIM=0.8948, LPIPS=0.1194; Spec_PSNR=18.3822, NonSpec_PSNR=24.4752, ASG_Energy_In_Residual=0.8544, ASG_Residual_IoU=0.6413.
- Training: initial_gaussians=100000, final_gaussians=34577, avg_asg_eval_count=39930.72, avg_sh_spec_mask_ratio=0.3068.
- Nhận xét: so với R027, SH mask tăng PSNR +0.0820, SSIM +0.0001, Spec_PSNR +0.2821 và ASG_IoU +0.0305; LPIPS xấu hơn +0.0014. So với R024, R028 chưa đạt fidelity tổng thể nhưng vượt Spec_PSNR và ASG_IoU.
- Kết luận: trên toaster, SH mask có tín hiệu tích cực hơn counter: nó cải thiện cả PSNR lẫn specular proxy so với baseline gần nhất. Tuy nhiên LPIPS và best tổng thể vẫn nghiêng về R024.

### R029

- Mục tiêu: bật Supervision Signal trên toaster sau SH mask: `lambda_spec_l1_weight=1.0` và `lambda_spec_reg=0.001`, chưa bật normal-delta.
- Output: [output/backups/toaster/spec-fastgs_v3_new_architecture_20260710_234233][R029-output]
- Config: scene=toaster, images=images, iterations=30000, asg_degree=32, use_ref_score=True, use_adaptive_prior=True, ref_prior_method=tan, use_sh_spec_mask=True, lambda_spec_l1_weight=1.0, lambda_spec_reg=0.001, use_normal_delta=False.
- Kết quả đáng chú ý: PSNR=22.0170, SSIM=0.8945, LPIPS=0.1196; Spec_PSNR=18.3851, ASG_Energy_In_Residual=0.8502, ASG_Residual_IoU=0.6564.
- Training: initial_gaussians=100000, final_gaussians=36458, avg_sh_spec_mask_ratio=0.3091, avg_spec_reg_loss=0.1205, time=8m14s.
- Nhận xét: so với R028, IoU tăng +0.0150 và Spec_PSNR tăng rất nhẹ +0.0029, nhưng PSNR giảm -0.1050 và LPIPS xấu hơn +0.0002. Weighted L1 + spec-reg làm ASG overlap tốt hơn nhưng fidelity tổng thể giảm.
- Kết luận: Supervision Signal có ích nếu mục tiêu là ASG localization proxy, nhưng cấu hình `lambda_spec_l1_weight=1.0` + `lambda_spec_reg=0.001` chưa nên chọn làm default cho toaster vì đánh đổi PSNR/LPIPS.

### R030

- Mục tiêu: bật thêm Normal Quality learned delta trên toaster với cùng Supervision Signal của R029.
- Output: [output/toaster][R030-output]
- Config: giống R029 nhưng use_normal_delta=True, normal_delta_lr=5e-05, normal_delta_max_norm=0.1, lambda_normal_delta_reg=0.0, lambda_normal_smooth=0.0.
- Kết quả đáng chú ý: PSNR=22.0036, SSIM=0.8942, LPIPS=0.1196; Spec_PSNR=18.4302, ASG_Energy_In_Residual=0.8439, ASG_Residual_IoU=0.6359.
- Training: initial_gaussians=100000, final_gaussians=36407, normal_delta_mean_norm=0.0251, normal_delta_max_norm_observed=0.1000, time=8m11s.
- Nhận xét: normal-delta thật sự học trên toaster synthetic branch. So với R029, Spec_PSNR tăng +0.0451 nhưng PSNR giảm -0.0134 và IoU giảm -0.0205. Điều này gợi ý delta có thể giúp một phần highlight sharpness nhưng làm ASG overlap/residual alignment kém hơn.
- Kết luận: learned normal delta là tín hiệu đáng giữ để nghiên cứu, nhưng cần regularize/tune thêm (`lambda_normal_delta_reg`, delay start, hoặc bật reflection/normal-specific diagnostic) trước khi coi là cải thiện chắc chắn.

### R031

- Mục tiêu: bật riêng specular-weighted L1 trên counter sau SH mask, không bật spec-reg và normal-delta.
- Output: [output/backups/counter/spec-fastgs_v3_new_architecture_20260711_132401][R031-output]
- Config: scene=counter, images=images_8, iterations=30000, asg_degree=64, use_ref_score=True, use_adaptive_prior=True, ref_prior_method=tan, use_sh_spec_mask=True, lambda_spec_l1_weight=1.0, lambda_spec_reg=0.0, use_normal_delta=False.
- Kết quả đáng chú ý: PSNR=30.7784, SSIM=0.9386, LPIPS=0.0646; Spec_PSNR=23.5796, ASG_Energy_In_Residual=0.5866, ASG_Residual_IoU=0.2727.
- Training: initial_gaussians=155767, final_gaussians=176705, avg_sh_spec_mask_ratio=0.0381, time=17m43s.
- Nhận xét: so với R026, spec-L1 tăng PSNR +0.0883 và Spec_PSNR +0.0835, LPIPS gần như tương đương, nhưng IoU giảm -0.0054. Đây là tín hiệu Supervision Signal tích cực nhất trên counter.
- Kết luận: `lambda_spec_l1_weight=1.0` đáng giữ để ablate tiếp trên counter, nhất là khi dùng cùng SH mask.

### R032

- Mục tiêu: bật riêng learned normal delta trên counter để kiểm tra Normal Quality khi không dùng spec-L1.
- Output: [output/backups/counter/spec-fastgs_v3_new_architecture_20260711_134351][R032-output]
- Config: giống R026 nhưng lambda_spec_l1_weight=0.0, lambda_spec_reg=0.0, use_normal_delta=True, real_use_reflection_dir=False.
- Kết quả đáng chú ý: PSNR=30.7402, SSIM=0.9387, LPIPS=0.0649; Spec_PSNR=23.5041, ASG_Energy_In_Residual=0.5773, ASG_Residual_IoU=0.2741.
- Training: initial_gaussians=155767, final_gaussians=175797, normal_delta_mean_norm=0.0000, normal_delta_max_norm_observed=0.0000, time=17m30s.
- Nhận xét: delta không học vì counter dùng real branch với `real_use_reflection_dir=False`; normal không nằm trên đường gradient specular. Metric khác R026 chủ yếu là run variance, không phải bằng chứng normal-delta có tác động.
- Kết luận: R032 là sanity check quan trọng: muốn đánh giá Normal Quality trên counter cần bật `real_use_reflection_dir=True` hoặc thiết kế loss normal riêng.

### R033

- Mục tiêu: kết hợp spec-L1 và normal-delta trên counter.
- Output: [output/backups/counter/spec-fastgs_v3_new_architecture_20260711_140224][R033-output]
- Config: giống R031 nhưng use_normal_delta=True, real_use_reflection_dir=False.
- Kết quả đáng chú ý: PSNR=30.6800, SSIM=0.9382, LPIPS=0.0652; Spec_PSNR=23.5915, ASG_Energy_In_Residual=0.5719, ASG_Residual_IoU=0.2753.
- Training: initial_gaussians=155767, final_gaussians=176952, normal_delta_mean_norm=0.0000, normal_delta_max_norm_observed=0.0000, time=17m44s.
- Nhận xét: Spec_PSNR cao nhất counter hiện tại, nhưng normal-delta vẫn không học (`norm=0`). So với R031, Spec_PSNR tăng +0.0120 nhưng PSNR/SSIM/LPIPS đều xấu hơn.
- Kết luận: nếu chỉ nhìn Spec_PSNR thì R033 đáng chú ý, nhưng không nên quy gain này cho Normal Quality; cần coi là biến thể Supervision/run variance.

### R034

- Mục tiêu: stress-test specular regularization mạnh trên counter với `lambda_spec_reg=0.1`.
- Output: [output/backups/counter/spec-fastgs_v3_new_architecture_20260711_142913][R034-output]
- Config: giống R033 nhưng lambda_spec_reg=0.1.
- Kết quả đáng chú ý: PSNR=30.6401, SSIM=0.9383, LPIPS=0.0656; Spec_PSNR=23.2331, ASG_Energy_In_Residual=0.5714, ASG_Residual_IoU=0.2446.
- Training: initial_gaussians=155767, final_gaussians=177584, avg_spec_reg_loss=0.0061, normal_delta_mean_norm=0.0000, time=17m35s.
- Nhận xét: so với R033, spec-reg=0.1 làm Spec_PSNR giảm -0.3584 và IoU giảm -0.0307, PSNR cũng giảm. Regularization mạnh phạt ASG quá nhiều, làm yếu phần specular.
- Kết luận: không dùng `lambda_spec_reg=0.1`; nếu ablate spec-reg thì chỉ nên thử rất nhỏ như 0.001 hoặc thấp hơn.

### R035

- Mục tiêu: thử spec-reg nhỏ hơn trên counter với `lambda_spec_reg=0.001` trong cấu hình spec-L1 + normal-delta.
- Output: [output/backups/counter/spec-fastgs_v3_new_architecture_20260711_145011][R035-output]
- Config: giống R033 nhưng lambda_spec_reg=0.001.
- Kết quả đáng chú ý: PSNR=30.6136, SSIM=0.9377, LPIPS=0.0657; Spec_PSNR=23.4578, ASG_Energy_In_Residual=0.5877, ASG_Residual_IoU=0.2734.
- Training: initial_gaussians=155767, final_gaussians=177469, avg_spec_reg_loss=0.0069, normal_delta_mean_norm=0.0000, time=17m53s.
- Nhận xét: so với R031/R033, spec-reg=0.001 không cải thiện counter trong run này; PSNR/SSIM/LPIPS đều thấp hơn, Spec_PSNR cũng không vượt R033.
- Kết luận: trên counter, Supervision Signal có vẻ nên bắt đầu từ weighted L1 không kèm spec-reg; spec-reg cần grid nhỏ hơn hoặc chỉ dùng khi visual cho thấy ASG leak.

### R036

- Mục tiêu: chạy ablation Normal Quality hợp lệ trên counter bằng cách bật `real_use_reflection_dir=True`, đồng thời dùng spec-L1, spec-reg rất nhỏ và SH mask.
- Output: [output/backups/counter/spec-fastgs_v3_new_architecture_20260711_154800][R036-output]
- Config: scene=counter, images=images_8, iterations=30000, asg_degree=64, use_ref_score=True, use_adaptive_prior=True, ref_prior_method=tan, real_use_reflection_dir=True, use_sh_spec_mask=True, lambda_spec_l1_weight=1.0, lambda_spec_reg=0.0001, use_normal_delta=True, normal_delta_max_norm=0.05, lambda_normal_delta_reg=0.001.
- Kết quả đáng chú ý: PSNR=30.5517, SSIM=0.9376, LPIPS=0.0656; Spec_PSNR=23.4433, ASG_Energy_In_Residual=0.6244, ASG_Residual_IoU=0.2831.
- Training: initial_gaussians=155767, final_gaussians=177426, avg_sh_spec_mask_ratio=0.0393, avg_spec_reg_loss=0.0148, normal_delta_mean_norm=0.0148, normal_delta_max_norm_observed=0.0500, time=18m16s.
- Nhận xét: khác R032-R035, normal-delta đã thật sự học khi real branch dùng reflection direction. Tuy nhiên so với R031/R033/R035, PSNR và Spec_PSNR đều thấp hơn; điểm sáng duy nhất là ASG_IoU cao nhất nhóm counter hiện tại.
- Kết luận: R036 xác nhận Normal Quality implementation có tác động trên counter khi bật `real_use_reflection_dir=True`, nhưng cấu hình này chưa tốt cho visual fidelity; hiện nó chỉ hữu ích như bằng chứng tác động/ASG alignment proxy.

### R037

- Mục tiêu: kiểm tra cấu hình real reflection + Supervision Signal nhẹ khi tắt SH mask và tắt normal-delta.
- Output: [output/backups/counter/spec-fastgs_v3_new_architecture_20260711_160755][R037-output]
- Config: scene=counter, images=images_8, iterations=30000, asg_degree=64, use_ref_score=True, use_adaptive_prior=True, ref_prior_method=tan, real_use_reflection_dir=True, use_sh_spec_mask=False, lambda_spec_l1_weight=1.0, lambda_spec_reg=0.0001, use_normal_delta=False.
- Kết quả đáng chú ý: PSNR=30.7709, SSIM=0.9389, LPIPS=0.0641; Spec_PSNR=23.3843, ASG_Energy_In_Residual=0.5203, ASG_Residual_IoU=0.2145.
- Training: initial_gaussians=155767, final_gaussians=181545, avg_spec_reg_loss=0.0095, time=17m44s.
- Nhận xét: so với R025 baseline no-mask, R037 thấp hơn rõ về PSNR/SSIM/LPIPS/Spec_PSNR và IoU; so với R031, việc bỏ SH mask và bật real reflection làm giảm spec proxy.
- Kết luận: `real_use_reflection_dir=True` + spec-L1/spec-reg nhẹ không nên dùng làm best-fidelity default cho counter; R025-like vẫn là cấu hình tổng thể tốt hơn.

### R038

- Mục tiêu: rerun counter với cấu hình R025-like sau khi tắt toàn bộ Supervision/Normal và SH mask để tạo mốc sạch trước soft-mask.
- Output: [output/backups/counter/spec-fastgs_v3_new_architecture_20260711_185043][R038-output]
- Config: scene=counter, images=images_8, iterations=30000, asg_degree=64, use_ref_score=True, use_adaptive_prior=True, ref_prior_method=tan, real_use_reflection_dir=False, use_sh_spec_mask=False, lambda_spec_l1_weight=0.0, lambda_spec_reg=0.0, use_normal_delta=False.
- Kết quả đáng chú ý: PSNR=30.9733, SSIM=0.9404, LPIPS=0.0635; Spec_PSNR=23.6029, ASG_Energy_In_Residual=0.5168, ASG_Residual_IoU=0.2383.
- Training: initial_gaussians=155767, final_gaussians=179295, avg_sh_spec_mask_ratio=0.0000, time=17m06s.
- Nhận xét: R038 thấp hơn R025 nhẹ về PSNR/LPIPS nhưng cao hơn rõ về Spec_PSNR và ASG_IoU, dù SH mask tắt. Điều này cho thấy no-mask rerun có variance/prior-rerun đáng kể, nên các ablation Representation cần so với baseline gần thời điểm như R038, không chỉ R025.
- Kết luận: R038 là mốc no-mask mới tốt cho counter; chưa phải bằng chứng Representation Capability vì `use_sh_spec_mask=False`.

### R039

- Mục tiêu: thử soft SH mask trên counter với `sh_spec_grad_scale=0.5`, threshold cao hơn và supervision/normal tắt.
- Output: [output/backups/counter/spec-fastgs_v3_new_architecture_20260711_194804][R039-output]
- Config: scene=counter, images=images_8, iterations=30000, asg_degree=64, use_ref_score=True, use_adaptive_prior=True, ref_prior_method=tan, real_use_reflection_dir=False, use_sh_spec_mask=True, sh_spec_grad_scale=0.5, sh_spec_mask_threshold=0.75, sh_spec_mask_start=80000, sh_spec_min_metric_count=3, lambda_spec_l1_weight=0.0, lambda_spec_reg=0.0, use_normal_delta=False.
- Kết quả đáng chú ý: PSNR=30.9077, SSIM=0.9399, LPIPS=0.0634; Spec_PSNR=23.4283, ASG_Energy_In_Residual=0.5068, ASG_Residual_IoU=0.2262.
- Training: initial_gaussians=155767, final_gaussians=179101, avg_sh_spec_mask_ratio=0.0000, time=17m07s.
- Nhận xét: dù `use_sh_spec_mask=True`, mask không active vì `sh_spec_mask_start=80000` lớn hơn tổng 30000 iterations. `avg_sh_spec_mask_ratio=0.0000` xác nhận không Gaussian nào bị mask. Metric thấp hơn R038 chủ yếu là rerun variance / prior state, không phải tác động soft mask.
- Kết luận: R039 là sanity check/failed-config quan trọng; không dùng để kết luận `SH_SPEC_GRAD_SCALE=0.5` tốt hay xấu. Cần rerun với `sh_spec_mask_start=8000` hoặc `10000`.

### R040

- Mục tiêu: chạy soft SH mask hợp lệ trên counter với `sh_spec_grad_scale=0.5`, `sh_spec_mask_start=8000`, threshold cao hơn và không trộn Supervision/Normal.
- Output: [output/backups/counter/spec-fastgs_v3_new_architecture_20260711_202207][R040-output]
- Config: scene=counter, images=images_8, iterations=30000, asg_degree=64, use_ref_score=True, use_adaptive_prior=True, ref_prior_method=tan, real_use_reflection_dir=False, use_sh_spec_mask=True, sh_spec_grad_scale=0.5, sh_spec_mask_threshold=0.75, sh_spec_mask_start=8000, sh_spec_min_metric_count=2, lambda_spec_l1_weight=0.0, lambda_spec_reg=0.0, use_normal_delta=False.
- Kết quả đáng chú ý: PSNR=30.9363, SSIM=0.9401, LPIPS=0.0631; Spec_PSNR=23.4298, ASG_Energy_In_Residual=0.5266, ASG_Residual_IoU=0.2430.
- Training: initial_gaussians=155767, final_gaussians=178943, avg_sh_spec_mask_ratio=0.0180, time=17m28s.
- Nhận xét: so với R038 no-mask gần nhất, R040 giảm PSNR -0.0370 và Spec_PSNR -0.1731, nhưng LPIPS tốt hơn -0.0004 và ASG_IoU tăng +0.0047. So với R039 inactive-mask, R040 tốt hơn về PSNR/LPIPS/IoU, xác nhận mask đã active nhưng tác động còn yếu và chưa cải thiện specular reconstruction.
- Kết luận: soft mask `scale=0.5/start=8000/threshold=0.75/count=2` làm role separation tăng rất nhẹ, nhưng chưa đủ để vượt no-mask baseline. Đây là bằng chứng rằng chỉ giảm SH gradient chưa đủ; Representation Capacity cần thêm signal tích cực cho ASG hoặc mask confidence tốt hơn.

### R041

- Mục tiêu: thử lại Supervision Signal nhẹ hơn sau soft SH mask bằng `lambda_spec_l1_weight=0.5`, không dùng spec-reg/normal-delta.
- Output: [output/backups/counter/spec-fastgs_v3_new_architecture_20260711_205247][R041-output]
- Config: scene=counter, images=images_8, iterations=30000, asg_degree=64, use_ref_score=True, use_adaptive_prior=True, ref_prior_method=tan, use_sh_spec_mask=True, sh_spec_grad_scale=0.5, sh_spec_mask_threshold=0.75, sh_spec_mask_start=8000, sh_spec_min_metric_count=2, lambda_spec_l1_weight=0.5, lambda_spec_reg=0.0, use_normal_delta=False.
- Kết quả đáng chú ý: PSNR=30.8887, SSIM=0.9398, LPIPS=0.0636; Spec_PSNR=23.3650, ASG_Energy_In_Residual=0.5262, ASG_Residual_IoU=0.2323.
- Training: initial_gaussians=155767, final_gaussians=179895, avg_sh_spec_mask_ratio=0.0182, time=18m01s.
- Nhận xét: so với R040, weighted L1=0.5 làm PSNR giảm -0.0476, LPIPS xấu hơn +0.0005, Spec_PSNR giảm -0.0649 và ASG_IoU giảm -0.0107. Vì loss này vẫn tác động lên final render chứ không tách riêng ASG component, nó không giải quyết đúng bottleneck Representation Capacity.
- Kết luận: không nên tiếp tục tăng/giảm quanh `lambda_spec_l1_weight` như hướng chính cho Representation Capacity; nếu dùng thì chỉ là ablation Supervision phụ, không phải solution core.

### R042

- Mục tiêu: kiểm tra ASG residual supervision riêng lẻ, tắt SH mask để xem loss mới có tự giúp ASG học residual specular hay không.
- Output: [output/backups/counter/spec-fastgs_v3_new_architecture_20260711_220403][R042-output]
- Config: scene=counter, images=images_8, iterations=30000, asg_degree=64, use_ref_score=True, use_adaptive_prior=True, ref_prior_method=tan, use_sh_spec_mask=False, use_asg_residual_supervision=True, lambda_asg_residual=0.05, lambda_asg_leak=0.005, asg_residual_start=8000, asg_residual_interval=16, asg_residual_ref_threshold=0.75, lambda_spec_l1_weight=0.0, lambda_spec_reg=0.0, use_normal_delta=False.
- Kết quả đáng chú ý: PSNR=30.8979, SSIM=0.9400, LPIPS=0.0640; Spec_PSNR=23.4321, ASG_Energy_In_Residual=0.5190, ASG_Residual_IoU=0.2346.
- Training: initial_gaussians=155767, final_gaussians=178522, avg_asg_residual_loss=0.0736, avg_asg_leak_loss=0.0051, avg_sh_spec_mask_ratio=0.0000, time=17m37s.
- Nhận xét: loss mới thật sự active, nhưng so với R038 no-mask gần nhất, R042 giảm PSNR -0.0754, LPIPS xấu hơn +0.0005, Spec_PSNR giảm -0.1708 và ASG_IoU giảm -0.0037. Điều này cho thấy residual target hiện tại chưa đủ tốt hoặc đang tối ưu lệch với metric render cuối.
- Kết luận: ASG residual supervision bản hiện tại chưa cho tín hiệu tích cực khi chạy một mình. Nó chứng minh code/loss chạy được, nhưng chưa chứng minh cải thiện Representation Capacity.

### R043

- Mục tiêu: kết hợp soft SH mask với ASG residual supervision để vừa giảm vai trò SH ở vùng specular vừa dạy ASG reconstruct residual.
- Output: [output/backups/counter/spec-fastgs_v3_new_architecture_20260711_222509][R043-output]
- Config: scene=counter, images=images_8, iterations=30000, asg_degree=64, use_ref_score=True, use_adaptive_prior=True, ref_prior_method=tan, use_sh_spec_mask=True, sh_spec_grad_scale=0.75, sh_spec_mask_threshold=0.75, sh_spec_mask_start=8000, sh_spec_min_metric_count=2, use_asg_residual_supervision=True, lambda_asg_residual=0.05, lambda_asg_leak=0.005, asg_residual_start=8000, asg_residual_interval=16, asg_residual_ref_threshold=0.75.
- Kết quả đáng chú ý: PSNR=30.8528, SSIM=0.9398, LPIPS=0.0630; Spec_PSNR=23.4023, ASG_Energy_In_Residual=0.5361, ASG_Residual_IoU=0.2489.
- Training: initial_gaussians=155767, final_gaussians=178627, avg_sh_spec_mask_ratio=0.0179, avg_asg_residual_loss=0.0736, avg_asg_leak_loss=0.0052, time=17m37s.
- Nhận xét: so với R040 soft-mask only, R043 tăng ASG_IoU +0.0058 và LPIPS tốt hơn rất nhẹ -0.0001, nhưng PSNR giảm -0.0835 và Spec_PSNR giảm -0.0276. So với R038 no-mask, IoU tăng +0.0105 nhưng PSNR giảm -0.1206 và Spec_PSNR giảm -0.2007.
- Kết luận: kết hợp soft mask + ASG residual chỉ cải thiện proxy overlap, không cải thiện reconstruction metric. Đây là bằng chứng mạnh rằng bottleneck không còn là thiếu loss đơn giản cho ASG, mà nằm ở thiết kế decomposition/mask target hoặc cách render ASG-only/residual target được định nghĩa.

### R044

- Mục tiêu: kiểm tra Normal Quality sạch hơn trên counter sau khi tắt Representation/Supervision: bật `use_normal_delta=True`, `real_use_reflection_dir=True`, delta regularization và smoothness toàn cục.
- Output: [output/backups/counter/spec-fastgs_v3_new_architecture_20260711_230133][R044-output]
- Config: scene=counter, images=images_8, iterations=30000, asg_degree=64, use_ref_score=True, use_adaptive_prior=True, ref_prior_method=tan, use_sh_spec_mask=False, use_asg_residual_supervision=False, lambda_spec_l1_weight=0.0, lambda_spec_reg=0.0, use_normal_delta=True, real_use_reflection_dir=True, normal_delta_lr=5e-05, normal_delta_max_norm=0.05, lambda_normal_delta_reg=0.001, lambda_normal_smooth=0.0001, normal_smooth_start_iter=8000, normal_smooth_use_ref_mask=False.
- Kết quả đáng chú ý: PSNR=30.8517, SSIM=0.9397, LPIPS=0.0636; Spec_PSNR=23.1791, ASG_Energy_In_Residual=0.5449, ASG_Residual_IoU=0.2167.
- Training: initial_gaussians=155767, final_gaussians=179473, normal_delta_mean_norm=0.0138, normal_delta_max_norm_observed=0.0500, avg_normal_delta_loss=0.000074, avg_normal_smooth_loss=0.7172, time=17m43s.
- Nhận xét: normal-delta thật sự học, nhưng so với R038 no-mask/no-normal baseline, R044 giảm PSNR -0.1216, Spec_PSNR -0.4238 và ASG_IoU -0.0216; LPIPS cũng xấu hơn nhẹ. Smoothness toàn cục không giúp ổn định reconstruction trong run này.
- Kết luận: R044 là negative evidence khá rõ cho Normal Quality hiện tại: có gradient/tác động, nhưng không chuyển thành metric tốt hơn.

### R045

- Mục tiêu: thử cùng Normal Quality như R044 nhưng chỉ áp smoothness theo ref-mask để xem giới hạn regularization vào vùng specular có tốt hơn không.
- Output: [output/backups/counter/spec-fastgs_v3_new_architecture_20260711_232105][R045-output]
- Config: giống R044 nhưng normal_smooth_use_ref_mask=True.
- Kết quả đáng chú ý: PSNR=30.6906, SSIM=0.9387, LPIPS=0.0646; Spec_PSNR=23.0685, ASG_Energy_In_Residual=0.5051, ASG_Residual_IoU=0.1920.
- Training: initial_gaussians=155767, final_gaussians=179400, normal_delta_mean_norm=0.0141, normal_delta_max_norm_observed=0.0500, avg_normal_delta_loss=0.000075, avg_normal_smooth_loss=0.7170, time=17m56s.
- Nhận xét: so với R044, ref-mask smooth làm PSNR giảm thêm -0.1611, LPIPS xấu hơn +0.0010, Spec_PSNR giảm -0.1106 và ASG_IoU giảm -0.0247. So với R038, mức giảm rất lớn: PSNR -0.2827, Spec_PSNR -0.5344, ASG_IoU -0.0463.
- Kết luận: `normal_smooth_use_ref_mask=True` không những không cứu Normal Quality mà còn làm metric tụt mạnh hơn. Không nên bật Normal Quality trong default pipeline hiện tại.

### R046

- Mục tiêu: tách riêng tác động của `real_use_reflection_dir=True` khi không bật normal-delta, nhưng vẫn còn global normal smoothness.
- Output: [output/backups/counter/spec-fastgs_v3_new_architecture_20260711_234236][R046-output]
- Config: scene=counter, images=images_8, iterations=30000, asg_degree=64, use_ref_score=True, use_adaptive_prior=True, ref_prior_method=tan, real_use_reflection_dir=True, use_normal_delta=False, lambda_normal_delta_reg=0.001, lambda_normal_smooth=0.0001, normal_smooth_use_ref_mask=False, use_sh_spec_mask=False, lambda_spec_l1_weight=0.0, lambda_spec_reg=0.0.
- Kết quả đáng chú ý: PSNR=30.7059, SSIM=0.9385, LPIPS=0.0651; Spec_PSNR=23.1551, ASG_Energy_In_Residual=0.5507, ASG_Residual_IoU=0.2003.
- Training: initial_gaussians=155767, final_gaussians=177867, normal_delta_mean_norm=0.0000, avg_normal_smooth_loss=0.7156, time=17m37s.
- Nhận xét: so với R038 no real-ref/no normal baseline, R046 giảm PSNR -0.2673, Spec_PSNR -0.4478 và ASG_IoU -0.0380. Vì `use_normal_delta=False`, mức giảm này cho thấy riêng việc đổi ASG sang reflection direction đã làm tối ưu khó hơn khi normal gốc chưa đủ tốt.
- Kết luận: `real_use_reflection_dir=True` không nên bật mặc định nếu chưa có normal objective tốt hơn. Nó hợp lý về mặt vật lý nhưng đang làm fidelity giảm.

### R047

- Mục tiêu: kiểm tra lại baseline khi tắt `real_use_reflection_dir` và không dùng normal-delta, trong khi các weight normal còn được truyền nhưng không active.
- Output: [output/backups/counter/spec-fastgs_v3_new_architecture_20260712_000315][R047-output]
- Config: scene=counter, images=images_8, iterations=30000, asg_degree=64, use_ref_score=True, use_adaptive_prior=True, ref_prior_method=tan, real_use_reflection_dir=False, use_normal_delta=False, lambda_normal_delta_reg=0.001, lambda_normal_smooth=0.0001, normal_smooth_use_ref_mask=False, use_sh_spec_mask=False, lambda_spec_l1_weight=0.0, lambda_spec_reg=0.0.
- Kết quả đáng chú ý: PSNR=30.9625, SSIM=0.9402, LPIPS=0.0629; Spec_PSNR=23.4922, ASG_Energy_In_Residual=0.4867, ASG_Residual_IoU=0.2215.
- Training: initial_gaussians=155767, final_gaussians=179364, normal_delta_mean_norm=0.0000, avg_normal_smooth_loss=0.7146, time=17m37s.
- Nhận xét: so với R046, chỉ cần tắt real reflection direction đã tăng PSNR +0.2566, LPIPS tốt hơn -0.0022 và Spec_PSNR tăng +0.3371. Điều này củng cố rằng bottleneck chính không phải weight smoothness, mà là reflection direction dựa trên normal chưa đáng tin.
- Kết luận: baseline không dùng real reflection direction vẫn là hướng metric tốt hơn.

### R048

- Mục tiêu: rerun baseline-like giống R047 để kiểm tra variance khi real reflection và normal-delta đều tắt.
- Output: [output/backups/counter/spec-fastgs_v3_new_architecture_20260712_164019][R048-output]
- Config: giống R047: real_use_reflection_dir=False, use_normal_delta=False, use_sh_spec_mask=False, lambda_spec_l1_weight=0.0, lambda_spec_reg=0.0.
- Kết quả đáng chú ý: PSNR=30.9136, SSIM=0.9402, LPIPS=0.0626; Spec_PSNR=23.3579, ASG_Energy_In_Residual=0.5386, ASG_Residual_IoU=0.2318.
- Training: initial_gaussians=155767, final_gaussians=179615, time=17m00s.
- Nhận xét: thấp hơn R047 về PSNR -0.0489 và Spec_PSNR -0.1343 nhưng LPIPS tốt hơn nhẹ. Đây là thêm một bằng chứng về run variance quanh baseline no-normal.
- Kết luận: không nên so một run normal đơn lẻ với một baseline duy nhất; cần so với dải R038/R047/R048/R049.

### R049

- Mục tiêu: baseline sạch sau khi thêm cơ chế `ref_score_conf`, tắt toàn bộ Representation/Supervision/Normal để kiểm tra trạng thái R025-like.
- Output: [output/backups/counter/spec-fastgs_v3_new_architecture_20260712_172509][R049-output]
- Config: scene=counter, images=images_8, iterations=30000, asg_degree=64, use_ref_score=True, use_adaptive_prior=True, ref_prior_method=tan, real_use_reflection_dir=False, use_sh_spec_mask=False, use_asg_residual_supervision=False, lambda_spec_l1_weight=0.0, lambda_spec_reg=0.0, use_normal_delta=False, lambda_normal_delta_reg=0.0, lambda_normal_smooth=0.0.
- Kết quả đáng chú ý: PSNR=30.9167, SSIM=0.9401, LPIPS=0.0635; Spec_PSNR=23.5190, ASG_Energy_In_Residual=0.5329, ASG_Residual_IoU=0.2463.
- Training: initial_gaussians=155767, final_gaussians=179284, ref_conf_quantile=0.0, refscore_conf_quantile=0.85, time=17m12s.
- Nhận xét: so với R038, R049 giảm PSNR -0.0565 nhưng Spec_PSNR cao hơn +0.0839 và ASG_IoU cao hơn +0.0080. So với R025, PSNR thấp hơn -0.0786. Đây là baseline no-normal mới, nhưng không phải cấu hình normal.
- Kết luận: ref-score confidence mới không gây tác động khi các nhánh dùng confidence tắt; chênh metric chủ yếu là variance/rerun.

### R050

- Mục tiêu: kiểm tra riêng `real_use_reflection_dir=True` khi đã tắt normal-delta và smoothness, để xem bản thân reflection direction có giúp không.
- Output: [output/backups/counter/spec-fastgs_v3_new_architecture_20260712_175418][R050-output]
- Config: giống R049 nhưng real_use_reflection_dir=True.
- Kết quả đáng chú ý: PSNR=30.7710, SSIM=0.9390, LPIPS=0.0641; Spec_PSNR=23.1857, ASG_Energy_In_Residual=0.5346, ASG_Residual_IoU=0.2038.
- Training: initial_gaussians=155767, final_gaussians=178024, time=17m24s.
- Nhận xét: so với R049, chỉ bật real reflection direction làm PSNR giảm -0.1458, LPIPS xấu hơn +0.0006, Spec_PSNR giảm -0.3333 và ASG_IoU giảm -0.0425. So với R046, việc tắt smoothness giúp PSNR tăng nhẹ +0.0650 nhưng vẫn thấp hơn baseline rõ.
- Kết luận: real reflection direction riêng lẻ hiện là negative evidence; hướng phản xạ vật lý chưa giúp metric khi normal gốc chưa đủ chính xác.

### R051

- Mục tiêu: thử Normal Quality mới sau khi ref-score confidence được thêm vào: bật `real_use_reflection_dir=True`, normal-delta, smoothness nhẹ hơn và ref-mask.
- Output: [output/backups/counter/spec-fastgs_v3_new_architecture_20260712_182023][R051-output]
- Config: scene=counter, images=images_8, iterations=30000, asg_degree=64, use_ref_score=True, use_adaptive_prior=True, ref_prior_method=tan, real_use_reflection_dir=True, use_normal_delta=True, normal_delta_lr=5e-05, normal_delta_max_norm=0.05, lambda_normal_delta_reg=0.001, lambda_normal_smooth=0.00002, normal_smooth_use_ref_mask=True, use_sh_spec_mask=False, lambda_spec_l1_weight=0.0, lambda_spec_reg=0.0.
- Kết quả đáng chú ý: PSNR=30.7766, SSIM=0.9393, LPIPS=0.0646; Spec_PSNR=23.1830, ASG_Energy_In_Residual=0.5096, ASG_Residual_IoU=0.2029.
- Training: initial_gaussians=155767, final_gaussians=177597, normal_delta_mean_norm=0.0154, normal_delta_max_norm_observed=0.0500, avg_normal_delta_loss=0.000085, avg_normal_smooth_loss=0.6121, time=17m45s.
- Nhận xét: normal-delta thật sự học, nhưng so với R050 real-ref only, PSNR chỉ tăng +0.0056, SSIM +0.0003, còn LPIPS xấu hơn +0.0005, Spec_PSNR giảm -0.0027 và ASG_IoU gần như không đổi -0.0009. So với R049 baseline sạch, R051 giảm PSNR -0.1402, Spec_PSNR -0.3360 và ASG_IoU -0.0434.
- Kết luận: ref-mask smooth nhẹ hơn có làm R051 đỡ xấu hơn R045, nhưng không biến Normal Quality thành cải thiện metric. Normal-delta học được nhưng không bù được tác hại của real reflection direction/normal không chính xác.

### R052

- Mục tiêu: kiểm tra cấu hình sau khi lược bỏ Normal Quality, bật lại `use_sh_spec_mask=True` với soft suppression nhẹ hơn để xem SH/ASG role separation có còn giúp metrics không.
- Output: [output/backups/counter/spec-fastgs_v3_new_architecture_20260712_185435][R052-output]
- Config: scene=counter, images=images_8, iterations=30000, asg_degree=64, use_ref_score=True, use_adaptive_prior=True, ref_prior_method=tan, real_use_reflection_dir=False, use_sh_spec_mask=True, sh_spec_grad_scale=0.75, sh_spec_mask_threshold=0.75, sh_spec_mask_start=8000, sh_spec_min_metric_count=2, use_asg_residual_supervision=False, lambda_spec_l1_weight=0.0, lambda_spec_reg=0.0.
- Kết quả đáng chú ý: PSNR=30.9680, SSIM=0.9402, LPIPS=0.0629; Spec_PSNR=23.6289, ASG_Energy_In_Residual=0.5216, ASG_Residual_IoU=0.2384.
- Training: initial_gaussians=155767, final_gaussians=179620, avg_sh_spec_mask_ratio=0.0180, time=17m36s.
- Nhận xét: so với R049 baseline sạch, R052 tăng PSNR +0.0513, Spec_PSNR +0.1099 và LPIPS tốt hơn -0.0005, nhưng ASG_IoU giảm -0.0079. So với R040 soft-mask cũ, cấu hình `grad_scale=0.75` cho fidelity tốt hơn rõ.
- Kết luận: SH spec mask bản nhẹ hiện là tín hiệu tích cực nhất của Representation Capacity: nó giúp final reconstruction và vùng specular, dù chưa làm ASG_IoU tăng ổn định.

### R053

- Mục tiêu: bật ASG residual supervision ở mức nhẹ vừa (`lambda_asg_residual=0.01`, `lambda_asg_leak=0.001`) trên nền R052 để xem loss trực tiếp cho ASG có cải thiện Spec_PSNR/IoU không.
- Output: [output/backups/counter/spec-fastgs_v3_new_architecture_20260712_192226][R053-output]
- Config: giống R052 nhưng use_asg_residual_supervision=True, lambda_asg_residual=0.01, lambda_asg_leak=0.001, asg_residual_start=8000, asg_residual_interval=16, asg_residual_ref_threshold=0.75.
- Kết quả đáng chú ý: PSNR=30.8989, SSIM=0.9400, LPIPS=0.0637; Spec_PSNR=23.4385, ASG_Energy_In_Residual=0.5316, ASG_Residual_IoU=0.2368.
- Training: initial_gaussians=155767, final_gaussians=179091, avg_sh_spec_mask_ratio=0.0178, avg_asg_residual_loss=0.0703, avg_asg_leak_loss=0.0051, time=18m25s.
- Nhận xét: so với R052, loss active nhưng PSNR giảm -0.0691, Spec_PSNR giảm -0.1904, LPIPS xấu hơn +0.0008 và ASG_IoU giảm nhẹ -0.0017. ASG_Energy_In_Residual tăng +0.0099, nghĩa là ASG tham gia nhiều hơn nhưng không đúng hơn.
- Kết luận: ASG residual supervision ở mức 0.01/0.001 không hiệu quả cho mục tiêu best metrics; target residual hiện tại có vẻ nhiễu hoặc lệch với reconstruction cuối.

### R054

- Mục tiêu: thử ASG residual supervision nhẹ hơn và muộn hơn để xem có giữ được lợi ích IoU mà bớt giảm PSNR không.
- Output: [output/counter][R054-output]
- Config: giống R052 nhưng use_asg_residual_supervision=True, lambda_asg_residual=0.003, lambda_asg_leak=0.0005, asg_residual_start=10000, asg_residual_interval=64, asg_residual_ref_threshold=0.85.
- Kết quả đáng chú ý: PSNR=30.9337, SSIM=0.9399, LPIPS=0.0637; Spec_PSNR=23.4247, ASG_Energy_In_Residual=0.5351, ASG_Residual_IoU=0.2485.
- Training: initial_gaussians=155767, final_gaussians=178605, avg_sh_spec_mask_ratio=0.0179, avg_asg_residual_loss=0.1139, avg_asg_leak_loss=0.0053, time=17m46s.
- Nhận xét: so với R052, R054 tăng ASG_IoU +0.0101 và ASG_Energy_In_Residual +0.0134, nhưng PSNR giảm -0.0343, SSIM giảm nhẹ, LPIPS xấu hơn +0.0008 và Spec_PSNR giảm -0.2042. So với R053, PSNR phục hồi +0.0348 và IoU tăng +0.0117, nhưng Spec_PSNR vẫn thấp hơn.
- Kết luận: giảm weight/start muộn hơn làm ASG residual bớt phá PSNR, nhưng vẫn không vượt SH-mask-only. Nếu ưu tiên metrics tốt nhất, không nên bật ASG residual supervision; nếu cần minh họa role separation/proxy IoU thì R054 có thể dùng như ablation phụ.

## Conclusion

- Thời điểm tổng kết: 2026-07-12 19:55 ICT.
- Runs được dùng: R001-R054.
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
- Tan vs Shafer/Klinker: trên counter, R020 Shafer + adaptive prior vượt R001 Tan về PSNR/SSIM/LPIPS và gần Spec_PSNR nhất trong nhóm Geometry Coverage, còn R025 Tan + adaptive prior + ASG64 đạt PSNR/LPIPS tốt hơn sau khi rerun pipeline mới. Trên toaster, R024 Tan + adaptive prior + ASG32 vẫn là best tổng thể, dù Shafer vẫn có vài run mạnh theo Spec_PSNR.
- Shafer/Klinker threshold: `0.65/0.30` là cân bằng nhất. `0.70/0.20` cho ASG_IoU cao nhưng PSNR thấp, còn `0.60/0.30` rộng hơn nhưng không vượt `0.65/0.30`.
- ASG degree trên counter: tăng ASG degree từ 32 -> 48 -> 64 cải thiện rõ trong nhóm Shafer cũ, và ASG64 vẫn là lựa chọn tốt nhất cho counter sau khi thêm adaptive prior.
- ASG degree trên toaster: tăng ASG degree tạo trade-off. Không adaptive thì R016 `asg=24` tốt nhất theo PSNR, R015 `asg=64` tốt theo LPIPS/IoU. Có adaptive thì ASG12 là quá thấp, ASG64 tốt LPIPS nhưng giảm IoU, còn ASG32 với Tan ở R024 cân bằng nhất.
- Scene-relative RefScore budget và soft RefScore decay: hai chỉnh sửa này làm budget tự scale theo số Gaussian ban đầu và giảm độ mạnh của prior khi scene đã nhiều Gaussian hơn. Tuy nhiên, R017/R019 cho thấy khi tắt adaptive prior, chúng chưa tự tạo ra cải thiện chắc chắn; chúng nên được xem là hạ tầng để RefScore ổn định hơn, không phải nguồn gain chính.
- Residual-adaptive prior: đây là chỉnh sửa Geometry Coverage có bằng chứng tích cực nhất. R018 tốt hơn R017 trên toaster/asg24 về PSNR, LPIPS, Spec_PSNR và ASG_IoU; R020 tốt hơn R019 trên counter/asg64 về PSNR, SSIM, LPIPS và Spec_PSNR. Tác động chính là cập nhật prior theo residual còn lại của model, nên nó bám sát lỗi geometric/specular thực tế hơn static prior.
- Sparse `f_rest` schedule: các run R017-R054 đều dùng `f_rest_interval=16/32/64`, giúp giảm cập nhật SH residual khi training đi xa hơn và nhường vai trò giải thích phần view-dependent cho ASG. Hiện chưa có ablation off riêng cho schedule này, nên chỉ nên kết luận là nó tương thích với cấu hình best mới, chưa khẳng định nó là nguồn gain độc lập.
- Representation Capability / SH spec mask: R025/R026 và R027/R028 cho thấy `use_sh_spec_mask=True` đẩy vai trò specular về ASG rõ rệt. Trên counter, R026 tăng Spec_PSNR và ASG_IoU rất mạnh so với R025 nhưng giảm PSNR/SSIM/LPIPS, nghĩa là hard mask đang quá cứng với `sh_spec_grad_scale=0.0`. R038 là no-mask rerun mới có Spec_PSNR cao, cho thấy baseline variance/prior-rerun đáng kể. R040 là soft-mask hợp lệ: mask active (`avg_sh_spec_mask_ratio=0.0180`) và tăng ASG_IoU nhẹ so với R038, nhưng Spec_PSNR/PSNR vẫn thấp hơn. R052 là soft SH mask tốt hơn sau khi gỡ Normal Quality: so với R049, PSNR tăng +0.0513 và Spec_PSNR tăng +0.1099, dù ASG_IoU giảm nhẹ. R042/R043/R053/R054 cho thấy ASG residual supervision đã chạy thật (`avg_asg_residual_loss>0`) nhưng không cải thiện best metrics; R054 tăng IoU lên 0.2485 nhưng PSNR/Spec_PSNR vẫn thấp hơn R052. Kết luận hiện tại: SH spec mask nhẹ có hiệu quả thực dụng cho reconstruction, còn ASG residual supervision chỉ cải thiện proxy trong một số cấu hình và chưa giúp metric cuối.
- Supervision Signal: weighted L1 có tín hiệu tốt hơn spec-reg khi chạy cùng SH mask. Trên counter, R031 tăng PSNR +0.0883 và Spec_PSNR +0.0835 so với R026 khi bật `lambda_spec_l1_weight=1.0` mà không dùng spec-reg. Nhưng R037 cho thấy khi tắt SH mask và bật `real_use_reflection_dir=True`, `lambda_spec_l1_weight=1.0` + `lambda_spec_reg=0.0001` vẫn không vượt R025/R031. Trên toaster, R029/R030 tăng ASG_IoU hoặc Spec_PSNR nhưng giảm PSNR/LPIPS so với R028. `lambda_spec_reg=0.1` ở R034 quá mạnh; `0.001` ở R035 và `0.0001` ở R036/R037 đều chưa tạo gain fidelity rõ.
- Normal Quality: toaster synthetic branch cho thấy `use_normal_delta=True` thật sự học được delta ở R030 (`mean_norm=0.0251`, `max_norm=0.1`) và tăng Spec_PSNR so với R029, nhưng giảm IoU và fidelity. Counter R032-R035 chưa phải ablation normal hợp lệ vì `real_use_reflection_dir=False`, khiến `normal_delta_mean_norm=0.0`. R036 là ablation hợp lệ hơn: bật `real_use_reflection_dir=True` làm normal-delta học được (`mean_norm=0.0148`, `max_norm=0.05`) và tăng ASG_IoU lên 0.2831, nhưng PSNR/SSIM/LPIPS/Spec_PSNR đều không tốt. R044/R045 là kiểm tra sạch hơn khi tắt Representation/Supervision và vẫn âm. R046/R050 cho thấy riêng `real_use_reflection_dir=True` đã kéo PSNR/Spec_PSNR xuống mạnh khi normal chưa đủ tốt; R051 cho thấy normal-delta + smooth ref-mask nhẹ chỉ tăng PSNR +0.0056 so với R050 nhưng vẫn thấp hơn R049 baseline sạch -0.1402 PSNR và -0.3360 Spec_PSNR. Vì vậy Normal Quality hiện chỉ chứng minh được tác động/gradient, không chứng minh được cải thiện visual quality.

### Best runs hiện tại

| Mục tiêu | Run | Lý do |
| --- | --- | --- |
| Counter - metric tổng thể | R025 | Best PSNR/LPIPS hiện tại; cần ghi rõ đây là Tan + adaptive rerun, không phải so sánh thuần với R020 Shafer. |
| Counter - Geometry Coverage baseline | R020 | Best Geometry Coverage trước Stage 2 theo cân bằng PSNR/SSIM/LPIPS/Spec_PSNR. |
| Counter - ASG alignment proxy | R036/R033/R054/R052 | R036 cao nhất ASG_IoU trong nhóm real-refl/normal nhưng fidelity thấp; R033 cao nhất Spec_PSNR trong nhóm hard-mask cũ; R054 tăng IoU bằng ASG residual nhẹ nhưng hy sinh Spec_PSNR; R052 là cấu hình SH-mask-only cân bằng nhất sau khi gỡ Normal Quality. |
| Toaster - metric tổng thể | R024 | Best PSNR/SSIM/LPIPS trong toàn bộ nhóm toaster và IoU cao. |
| Toaster - Representation/Supervision proxy | R029/R030 | R029 cao nhất ASG_IoU; R030 cao nhất Spec_PSNR, nhưng PSNR/LPIPS chưa vượt R024/R028. |
| Toaster - Shafer Spec_PSNR | R016/R023 | R016 cao nhất trong nhóm Shafer cũ; R023 cao nhất trong nhóm Shafer-adaptive. |
| Toaster - low-capacity negative case | R022 | ASG12 tụt rõ, hữu ích để chứng minh capacity quá thấp không đủ. |

### Nhận xét về chỉnh sửa Geometry Coverage mới

Các chỉnh sửa mới đi đúng hướng, nhưng gain không đến từ một công tắc đơn lẻ. Auto budget và soft decay giúp RefScore bớt phụ thuộc scene size, còn adaptive prior mới là phần làm metric nhảy rõ nhất vì nó tái căn chỉnh prior theo residual của model trong quá trình train. Trên counter, R020 là bằng chứng rất mạnh: vừa vượt R012 cũ về PSNR/SSIM/LPIPS/Spec_PSNR, vừa giữ ASG_IoU cao hơn R012. Trên toaster, R024 cho thấy adaptive prior kết hợp Tan-Ikeuchi và ASG32 tạo cấu hình cân bằng hơn Shafer-adaptive.

Điểm cần cẩn thận khi viết thesis: Geometry Coverage và Representation Capability đều không đơn giản là "tăng IoU là tốt". R019 có IoU nhỉnh hơn R020 nhưng fidelity thấp hơn; R026 có IoU cao nhất counter nhưng PSNR/LPIPS giảm mạnh; R028 có Spec_PSNR/IoU tốt nhất toaster nhưng vẫn chưa vượt R024 về fidelity tổng thể. Vì vậy phần kết quả nên trình bày theo trade-off: coverage/role separation tốt phải đi kèm PSNR/LPIPS/Spec_PSNR, không chỉ một proxy mask.

### Nhận xét về chỉnh sửa Representation / Supervision / Normal mới

Các run R025-R028 là proof-of-concept đầu tiên cho trục Representation Capability. Kiến trúc ASG/MLP vẫn để default bằng `asg_num_theta=-1`, `asg_num_phi=-1`, `specular_hidden=-1`, `specular_layers=-1`, nên biến số chính ở đây là SH spec mask. Khi bật hard mask, ASG nhận nhiều trách nhiệm hơn ở vùng ref-score cao: counter tăng ASG_IoU từ 0.2199 lên 0.2781, toaster tăng từ 0.6109 lên 0.6413. Điều này đúng với mục tiêu role separation. Sau khi chuyển sang soft suppression nhẹ hơn, R052 cho thấy SH mask có thể cải thiện cả PSNR và Spec_PSNR so với baseline sạch R049.

Tuy vậy, hard mask hiện tại đang là hard suppression (`sh_spec_grad_scale=0.0`), nên counter bị mất fidelity nặng. Toaster phản ứng tốt hơn vì vừa tăng PSNR vừa tăng Spec_PSNR/IoU so với R027, nhưng vẫn xấu hơn R024 về PSNR/LPIPS. R039 là failed-config vì start=80000 vượt quá 30000 iterations và `avg_sh_spec_mask_ratio=0.0`. R040 đã chạy đúng soft mask nhưng chỉ tăng IoU nhẹ, trong khi Spec_PSNR thấp hơn R038. R052 là bước cải thiện quan trọng vì `grad_scale=0.75` đủ nhẹ để không phá reconstruction. R042/R043/R053/R054 kiểm tra hướng "dạy ASG reconstruct residual" trực tiếp hơn: loss đều active, nhưng kết quả không vượt R052. Vì vậy vấn đề không chỉ là thiếu một L1 residual đơn giản; target `gt - sh_only` và mask confidence hiện vẫn chưa đủ sạch để biến ASG residual supervision thành gain metrics.

Nhóm Supervision R029-R037 cho thấy `lambda_spec_l1_weight=1.0` là knob đáng thử tiếp, đặc biệt trên counter ở R031 khi chạy cùng SH mask và không kèm spec-reg. Ngược lại, `lambda_spec_reg` chưa có bằng chứng tốt: R034 cho thấy 0.1 quá mạnh, R035 với 0.001 không cải thiện counter, còn 0.0001 trong R036/R037 cũng chưa đưa fidelity vượt baseline. Trên toaster, supervision làm tăng ASG overlap/spec proxy nhưng kéo PSNR/LPIPS xuống, nên nên tune nhỏ hơn hoặc chạy riêng `lambda_spec_l1_weight` không kèm spec-reg.

Nhóm Normal Quality mới chỉ có bằng chứng tác động, chưa có bằng chứng cải thiện fidelity. R030 trên toaster xác nhận learned normal delta có thể học và tăng Spec_PSNR, nhưng trade-off với IoU/fidelity chưa tốt. R036 trên counter là run hợp lệ hơn vì `real_use_reflection_dir=True` làm delta học được; tuy vậy metric tổng thể giảm mạnh và chỉ ASG_IoU tăng. R044/R045 loại bỏ nhiễu từ Representation/Supervision và vẫn cho kết quả âm. R046/R050 tách riêng `real_use_reflection_dir=True` và cho thấy chỉ đổi ASG sang reflection direction đã làm PSNR/Spec_PSNR giảm mạnh so với baseline no-real-reflection. R051 dùng normal-delta + smooth ref-mask nhẹ hơn, delta học được (`mean_norm=0.0154`) nhưng chỉ nhích hơn R050 về PSNR +0.0056 và vẫn thua R049 rất xa. Vì vậy normal-delta không nên đưa vào default; nếu tiếp tục nghiên cứu thì cần đổi thiết kế normal objective hoặc cách ước lượng normal/reflection direction, không nên chỉ tune nhỏ quanh các weight hiện tại.

### Kết luận về hai thành phần đã loại bỏ

`real_use_reflection_dir` đã được khảo sát như một hướng làm ASG mang tính vật lý hơn bằng cách dùng hướng phản xạ từ normal thay cho view direction. Tuy nhiên, R046 và R050 cho thấy chỉ riêng việc bật real reflection direction đã làm fidelity giảm rõ: R050 thấp hơn R049 `-0.1458` PSNR, `-0.3333` Spec_PSNR và `-0.0425` ASG_IoU. Khi kết hợp normal-delta ở R051, metric chỉ nhích nhẹ so với R050 nhưng vẫn thua xa baseline sạch R049. Kết luận: trong pipeline hiện tại, normal/geometry chưa đủ đáng tin để dùng reflection direction làm input ASG; thành phần này không góp phần cải thiện specular visual quality cuối cùng và đã được loại khỏi final pipeline.

`use_asg_residual_supervision` đã được khảo sát như một hướng dạy ASG trực tiếp reconstruct residual specular. Các run R042/R043/R053/R054 xác nhận loss này thật sự active và có thể làm ASG tham gia mạnh hơn, nhưng không tạo ra best metrics. So với R052 SH-mask-only, R053 giảm `-0.0691` PSNR và `-0.1904` Spec_PSNR; R054 nhẹ hơn có tăng ASG_IoU `+0.0101` nhưng vẫn giảm `-0.0343` PSNR và `-0.2042` Spec_PSNR. Kết luận: residual target `gt - sh_only` và ref-mask hiện chưa đủ sạch; loss này hữu ích như ablation/diagnostic cho role separation nhưng không cải thiện reconstruction/specular quality tốt nhất, nên cũng đã được loại khỏi final pipeline.

Sau khi loại bỏ hai thành phần trên, cấu hình final nên giữ phần có bằng chứng tốt nhất: Geometry Coverage với `use_ref_score=True`, `use_adaptive_prior=True`, `ref_prior_method=tan`, ASG64 cho counter, và SH spec mask nhẹ `use_sh_spec_mask=True`, `sh_spec_grad_scale=0.75`, `sh_spec_mask_start=8000`, `sh_spec_mask_threshold=0.75`, `sh_spec_min_metric_count=2`. Các loss phụ `lambda_spec_l1_weight`, `lambda_spec_reg` và ASG residual đều để 0 trong cấu hình đề xuất.

### Kết luận cho thesis

Các kết quả cho thấy cải thiện specular reconstruction là hướng hợp lý vì specular chiếm khoảng 30-50% tổng sai số L1 tùy dataset. Sau R001-R054, kết luận nên tách ba tầng: residual-adaptive prior là bổ sung đáng giữ cho Geometry Coverage; SH spec mask nhẹ ở R052 là tín hiệu Representation Capacity tốt nhất hiện tại vì cải thiện PSNR/Spec_PSNR so với baseline sạch R049; ASG residual supervision bản hiện tại chưa cho hiệu quả để đạt metrics tốt nhất dù loss đã active. Supervision Signal có tín hiệu tốt nhất ở weighted L1=1.0 trong R031, nhưng R041 với weight=0.5 lại giảm metric khi dùng cùng soft mask, nên không nên tiếp tục xem đây là hướng chính. Normal delta đã được xác nhận có tác động cả trên toaster và counter khi counter bật `real_use_reflection_dir=True`, nhưng R044-R051 cho thấy tác động đó không chuyển thành metric tốt hơn. R049 là baseline sạch sau ref-score confidence; R052 là cấu hình counter hiện tại đáng giữ nếu ưu tiên fidelity/specular PSNR; R053/R054 cho thấy thêm ASG residual chỉ tăng ASG energy/IoU trong một số trường hợp nhưng làm PSNR/Spec_PSNR thấp hơn R052. Counter hiện có R025/R038/R052 nếu ưu tiên fidelity/spec proxy, còn R036/R033/R054 chỉ nên dùng để minh họa ASG alignment/proxy trade-off; toaster vẫn hợp nhất với R024 cho metric tổng thể.

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
[R020-output]: output/backups/counter/spec-fastgs_v3_new_architecture_20260709_081711
[R021-output]: output/backups/toaster/spec-fastgs_v3_new_architecture_20260709_083322
[R022-output]: output/backups/toaster/spec-fastgs_v3_new_architecture_20260709_084600
[R023-output]: output/backups/toaster/spec-fastgs_v3_new_architecture_20260709_090531
[R024-output]: output/backups/toaster/spec-fastgs_v3_new_architecture_20260709_104803
[R025-output]: output/backups/counter/spec-fastgs_v3_new_architecture_20260710_140202
[R026-output]: output/backups/counter/spec-fastgs_v3_new_architecture_20260710_142249
[R027-output]: output/backups/toaster/spec-fastgs_v3_new_architecture_20260710_133006
[R028-output]: output/backups/toaster/spec-fastgs_v3_new_architecture_20260710_134433
[R029-output]: output/backups/toaster/spec-fastgs_v3_new_architecture_20260710_234233
[R030-output]: output/toaster
[R031-output]: output/backups/counter/spec-fastgs_v3_new_architecture_20260711_132401
[R032-output]: output/backups/counter/spec-fastgs_v3_new_architecture_20260711_134351
[R033-output]: output/backups/counter/spec-fastgs_v3_new_architecture_20260711_140224
[R034-output]: output/backups/counter/spec-fastgs_v3_new_architecture_20260711_142913
[R035-output]: output/backups/counter/spec-fastgs_v3_new_architecture_20260711_145011
[R036-output]: output/backups/counter/spec-fastgs_v3_new_architecture_20260711_154800
[R037-output]: output/backups/counter/spec-fastgs_v3_new_architecture_20260711_160755
[R038-output]: output/backups/counter/spec-fastgs_v3_new_architecture_20260711_185043
[R039-output]: output/backups/counter/spec-fastgs_v3_new_architecture_20260711_194804
[R040-output]: output/backups/counter/spec-fastgs_v3_new_architecture_20260711_202207
[R041-output]: output/backups/counter/spec-fastgs_v3_new_architecture_20260711_205247
[R042-output]: output/backups/counter/spec-fastgs_v3_new_architecture_20260711_220403
[R043-output]: output/backups/counter/spec-fastgs_v3_new_architecture_20260711_222509
[R044-output]: output/backups/counter/spec-fastgs_v3_new_architecture_20260711_230133
[R045-output]: output/backups/counter/spec-fastgs_v3_new_architecture_20260711_232105
[R046-output]: output/backups/counter/spec-fastgs_v3_new_architecture_20260711_234236
[R047-output]: output/backups/counter/spec-fastgs_v3_new_architecture_20260712_000315
[R048-output]: output/backups/counter/spec-fastgs_v3_new_architecture_20260712_164019
[R049-output]: output/backups/counter/spec-fastgs_v3_new_architecture_20260712_172509
[R050-output]: output/backups/counter/spec-fastgs_v3_new_architecture_20260712_175418
[R051-output]: output/backups/counter/spec-fastgs_v3_new_architecture_20260712_182023
[R052-output]: output/backups/counter/spec-fastgs_v3_new_architecture_20260712_185435
[R053-output]: output/backups/counter/spec-fastgs_v3_new_architecture_20260712_192226
[R054-output]: output/counter
