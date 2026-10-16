**Warm latency, ms, p50 (p95) — 8 threads on the 8 P-cores (P8)**

| workload | CPU-Max (Q8_0) | PyTorch FP32 (Sentence Transformers) | ONNX Runtime INT8 | upstream embeddinggemma.c (Q8_0) | fork vs fastest other, same budget (p50) |
|---|---:|---:|---:|---:|---:|
| text, 16 tokens | **8.6 (9.6)** | 20.4 (21.0) | 36.8 (37.4) | 17.4 (17.6) | 2.03x |
| text, 128 tokens | **29.5 (29.7)** | 77.5 (78.5) | 84.7 (91.9) | 117 (128) | 2.63x |
| text, 512 tokens | **134 (136)** | 250 (253) | 292 (295) | 556 (565) | 1.87x |
| image, 70 tokens (336x480) | **152 (154)** | 341 (342) | 1,368 (1,525) | 3,909 (4,241) | 2.24x |
| image, 280 tokens (672x960) | **942 (947)** | 1,829 (1,840) | 3,028 (3,139) | 32,082 (32,215) | 1.94x |
| image, 1120 tokens (1344x1920) | **10,140 (10,562)** | 13,646 (13,676) | 14,321 (14,327) | 552,466 (554,508) | 1.35x |
| audio, 5 s | **82.1 (83.1)** | 361 (406) | 1,297 (1,483) | 1,320 (1,330) | 4.40x |
| audio, 15 s | **241 (243)** | 654 (734) | 2,014 (2,115) | 3,935 (4,244) | 2.71x |
| audio, 30 s | **523 (524)** | 1,169 (1,490) | 2,448 (2,504) | 8,202 (8,603) | 2.24x |
| video, 4 frames | **1,482 (1,496)** | 3,265 (3,309) | 4,332 (4,428) | 42,652 (46,146) | 2.20x |
| video, 8 frames | **3,096 (3,145)** | 6,946 (6,960) | 7,465 (7,507) | 87,577 (95,909) | 2.24x |
| video, 16 frames | **6,405 (6,422)** | 16,418 (17,979) | 14,392 (14,494) | 161,199 (170,245) | 2.25x |
| text + image | **1,114 (1,178)** | 1,707 (1,716) | 3,050 (3,235) | 33,015 (34,190) | 1.53x |
| text + audio | **258 (260)** | 635 (689) | 1,997 (2,055) | 3,991 (4,317) | 2.46x |
| text + image + audio | **1,295 (1,308)** | 2,495 (2,604) | 4,454 (4,485) | 36,394 (37,071) | 1.93x |
| text + video | **3,137 (3,289)** | 7,276 (9,353) | 7,519 (7,822) | 87,103 (87,788) | 2.32x |

Same-budget geometric mean (P8, 16 workloads): 2.20x over the fastest other backend (minimum 1.35x).

**Warm latency, ms, p50 (p95) — all 28 hardware threads (ALL)**

| workload | CPU-Max (Q8_0) | PyTorch FP32 (Sentence Transformers) | ONNX Runtime INT8 | upstream embeddinggemma.c (Q8_0) | fork vs fastest other, same budget (p50) |
|---|---:|---:|---:|---:|---:|
| text, 16 tokens | **8.9 (12.7)** | 28.2 (34.0) | 68.0 (73.3) | 21.9 (24.1) | 2.47x |
| text, 128 tokens | **29.3 (29.7)** | 96.0 (1,849) | 158 (256) | 108 (111) | 3.27x |
| text, 512 tokens | **128 (130)** | 344 (1,976) | 408 (499) | 398 (434) | 2.69x |
| image, 70 tokens (336x480) | **139 (152)** | 419 (4,657) | 1,824 (2,036) | 2,548 (2,572) | 3.03x |
| image, 280 tokens (672x960) | **783 (826)** | 3,709 (5,216) | 3,294 (3,530) | – | 4.21x |
| image, 1120 tokens (1344x1920) | **7,678 (7,823)** | 25,366 (27,197) | 12,142 (13,083) | – | 1.58x |
| audio, 5 s | **84.6 (86.4)** | 1,852 (3,558) | 1,648 (2,021) | 987 (1,002) | 11.66x |
| audio, 15 s | **237 (251)** | 4,126 (8,433) | 2,470 (2,698) | 2,911 (3,032) | 10.40x |
| audio, 30 s | **441 (461)** | 18,259 (18,894) | 2,995 (3,236) | – | 6.79x |
| video, 4 frames | **1,271 (4,569)** | 12,703 (14,208) | 4,579 (5,060) | – | 3.60x |
| video, 8 frames | **2,560 (3,404)** | 18,955 (19,005) | 7,219 (7,504) | – | 2.82x |
| video, 16 frames | **5,313 (5,337)** | 23,168 (24,622) | 14,481 (15,445) | – | 2.73x |
| text + image | **849 (872)** | 10,109 (15,189) | 3,314 (3,537) | – | 3.90x |
| text + audio | **254 (271)** | 6,109 (10,606) | 2,384 (2,506) | 2,901 (2,918) | 9.37x |
| text + image + audio | **1,070 (1,098)** | 11,795 (12,890) | 4,987 (5,310) | – | 4.66x |
| text + video | **2,629 (6,952)** | 21,358 (21,682) | 7,142 (7,259) | – | 2.72x |

Same-budget geometric mean (ALL, 16 workloads): 4.02x over the fastest other backend (minimum 1.58x).

**Best thread budget per backend (P8 or ALL), p50 ms**

| workload | CPU-Max (Q8_0) | PyTorch FP32 (Sentence Transformers) | ONNX Runtime INT8 | upstream embeddinggemma.c (Q8_0) | fork vs fastest other |
|---|---:|---:|---:|---:|---:|
| text, 16 tokens | **8.6 (P8)** | 20.4 (P8) | 36.8 (P8) | 17.4 (P8) | 2.03x vs upstream |
| text, 128 tokens | **29.3 (ALL)** | 77.5 (P8) | 84.7 (P8) | 108 (ALL) | 2.64x vs PyTorch FP32 |
| text, 512 tokens | **128 (ALL)** | 250 (P8) | 292 (P8) | 398 (ALL) | 1.95x vs PyTorch FP32 |
| image, 70 tokens (336x480) | **139 (ALL)** | 341 (P8) | 1,368 (P8) | 2,548 (ALL) | 2.46x vs PyTorch FP32 |
| image, 280 tokens (672x960) | **783 (ALL)** | 1,829 (P8) | 3,028 (P8) | 32,082 (P8) | 2.34x vs PyTorch FP32 |
| image, 1120 tokens (1344x1920) | **7,678 (ALL)** | 13,646 (P8) | 12,142 (ALL) | 552,466 (P8) | 1.58x vs ONNX Runtime INT8 |
| audio, 5 s | **82.1 (P8)** | 361 (P8) | 1,297 (P8) | 987 (ALL) | 4.40x vs PyTorch FP32 |
| audio, 15 s | **237 (ALL)** | 654 (P8) | 2,014 (P8) | 2,911 (ALL) | 2.76x vs PyTorch FP32 |
| audio, 30 s | **441 (ALL)** | 1,169 (P8) | 2,448 (P8) | 8,202 (P8) | 2.65x vs PyTorch FP32 |
| video, 4 frames | **1,271 (ALL)** | 3,265 (P8) | 4,332 (P8) | 42,652 (P8) | 2.57x vs PyTorch FP32 |
| video, 8 frames | **2,560 (ALL)** | 6,946 (P8) | 7,219 (ALL) | 87,577 (P8) | 2.71x vs PyTorch FP32 |
| video, 16 frames | **5,313 (ALL)** | 16,418 (P8) | 14,392 (P8) | 161,199 (P8) | 2.71x vs ONNX Runtime INT8 |
| text + image | **849 (ALL)** | 1,707 (P8) | 3,050 (P8) | 33,015 (P8) | 2.01x vs PyTorch FP32 |
| text + audio | **254 (ALL)** | 635 (P8) | 1,997 (P8) | 2,901 (ALL) | 2.50x vs PyTorch FP32 |
| text + image + audio | **1,070 (ALL)** | 2,495 (P8) | 4,454 (P8) | 36,394 (P8) | 2.33x vs PyTorch FP32 |
| text + video | **2,629 (ALL)** | 7,276 (P8) | 7,142 (ALL) | 87,103 (P8) | 2.72x vs ONNX Runtime INT8 |

Geometric mean over the 16 workloads: **2.46x** faster than the fastest other tested backend for each workload, each backend at its best budget (minimum 1.58x).

**Final-embedding cosine vs the official Sentence Transformers FP32 model (same inputs)**

| workload | fork | fork_28t | onnx_q8 | upstream | upstream_no_timestamps | st_fp32_run1 |
|---|---:|---:|---:|---:|---:|---:|
| text, 16 tokens | 0.99980 | 0.99980 | 0.99993 | 0.99981 | – | 1.00000 |
| text, 128 tokens | 0.99974 | 0.99974 | 0.99989 | 0.99973 | – | 1.00000 |
| text, 512 tokens | 0.99977 | 0.99977 | 0.99993 | 0.99981 | – | 1.00000 |
| image, 70 tokens (336x480) | 0.99964 | 0.99964 | 0.99850 | 0.99867 | – | 1.00000 |
| image, 280 tokens (672x960) | 0.99978 | 0.99978 | 0.99962 | 0.99954 | – | 1.00000 |
| image, 1120 tokens (1344x1920) | 0.99986 | 0.99986 | 0.99928 | 0.99963 | – | 1.00000 |
| audio, 5 s | 0.99914 | 0.99914 | 0.99984 | 0.99856 | – | 1.00000 |
| audio, 15 s | 0.99946 | 0.99946 | 0.99985 | 0.99895 | – | 1.00000 |
| audio, 30 s | 0.99932 | 0.99932 | 0.99981 | 0.99883 | – | 1.00000 |
| video, 4 frames | 0.99989 | 0.99989 | 0.99850 | 0.87563 | 0.99979 | 1.00000 |
| video, 8 frames | 0.99989 | 0.99989 | 0.99885 | 0.94145 | 0.99983 | 1.00000 |
| video, 16 frames | 0.99987 | 0.99987 | 0.99871 | 0.96294 | – | 1.00000 |
| text + image | 0.99970 | 0.99970 | 0.99932 | 0.99956 | – | 1.00000 |
| text + audio | 0.99984 | 0.99984 | 0.99992 | 0.99980 | – | 1.00000 |
| text + image + audio | 0.99973 | 0.99973 | 0.99929 | 0.99955 | – | 1.00000 |
| text + video | 0.99982 | 0.99982 | 0.99884 | 0.95552 | 0.99977 | 1.00000 |
| **min / mean** | **0.99914** / 0.99970 | **0.99914** / 0.99970 | **0.99850** / 0.99938 | **0.87563** / 0.98300 | **0.99977** / 0.99980 | **1.00000** / 1.00000 |

**Retrieval quality (higher is better; `top1 agr.` = share of queries whose top hit equals the FP32 model's)**

| task | backend | R@1 | R@5 | R@10 | nDCG@10 | top-10 overlap vs FP32 | top1 agr. vs FP32 |
|---|---|---:|---:|---:|---:|---:|---:|
| flickr (5000 q / 1000 docs) | fork_official | 0.8630 | 0.9756 | 0.9872 | 0.9307 | 0.9782 | 0.9928 |
| flickr (5000 q / 1000 docs) | fork_upstream_semantics | 0.8634 | 0.9748 | 0.9876 | 0.9307 | 0.9706 | 0.9904 |
| flickr (5000 q / 1000 docs) | st_fp32 | 0.8636 | 0.9750 | 0.9872 | 0.9309 | 1.0000 | 1.0000 |
| clotho (521 q / 120 docs) | fork_official | 0.1305 | 0.3781 | 0.5298 | 0.3032 | 0.9056 | 0.8023 |
| clotho (521 q / 120 docs) | fork_upstream_semantics | 0.1344 | 0.3724 | 0.5605 | 0.3133 | 0.8902 | 0.7850 |
| clotho (521 q / 120 docs) | quixi_upstream | 0.1305 | 0.3551 | 0.5662 | 0.3120 | 0.8885 | 0.8100 |
| clotho (521 q / 120 docs) | st_fp32 | 0.1324 | 0.3743 | 0.5336 | 0.3025 | 1.0000 | 1.0000 |
| msrvtt (200 q / 200 docs) | fork_official | 0.6350 | 0.8450 | 0.8850 | 0.7623 | 0.9760 | 0.9800 |
| msrvtt (200 q / 200 docs) | fork_upstream_semantics | 0.6100 | 0.8400 | 0.8850 | 0.7465 | 0.6745 | 0.7500 |
| msrvtt (200 q / 200 docs) | st_fp32 | 0.6200 | 0.8550 | 0.8850 | 0.7564 | 1.0000 | 1.0000 |
| scifact (300 q / 5183 docs) | fork_official | 0.7700 | 0.9100 | 0.9467 | 0.8521 | 0.9790 | 0.9833 |
| scifact (300 q / 5183 docs) | fork_upstream_semantics | 0.7700 | 0.9100 | 0.9467 | 0.8521 | 0.9790 | 0.9833 |
| scifact (300 q / 5183 docs) | quixi_upstream | 0.7733 | 0.9100 | 0.9500 | 0.8539 | 0.9780 | 0.9833 |
| scifact (300 q / 5183 docs) | st_fp32 | 0.7733 | 0.9067 | 0.9467 | 0.8514 | 1.0000 | 1.0000 |

**Peak resident memory of a fresh server per workload (MB, P8)**

| backend | t512 | i280 | i1120 | a30 | v16 | m_tia |
|---|---:|---:|---:|---:|---:|---:|
| fork_5dd7dbf | 350 | 539 | 746 | 748 | 759 | 936 |
| fork_5dd7dbf_norelease | 494 | 838 | 1,044 | 1,203 | 1,057 | 1,546 |
| fork_5dd7dbf_prepack | 850 | 875 | 1,084 | 915 | 1,099 | 941 |
| quixi_upstream | 322 | 454 | 720 | 654 | 700 | 808 |

**Throughput with concurrent clients (requests/s, distinct inputs; range over the runs)**

| backend | budget | workload | clients | req/s |
|---|---|---|---:|---:|
| fork_5dd7dbf | ALL | a15 | 2 | 2.63-4.23 |
| fork_5dd7dbf | P8 | a15 | 2 | 4.04-4.06 |
| quixi_upstream | P8 | a15 | 2 | 0.24-0.25 |
| fork_5dd7dbf | ALL | i70 | 2 | 5.90-6.95 |
| fork_5dd7dbf | P8 | i70 | 2 | 6.11-6.26 |
| quixi_upstream | P8 | i70 | 2 | 0.24-0.25 |
| fork_5dd7dbf | ALL | t128 | 4 | 34.65-35.38 |
| fork_5dd7dbf | P8 | t128 | 4 | 33.19-36.22 |
| quixi_upstream | P8 | t128 | 4 | 7.79-8.18 |
| fork_5dd7dbf | ALL | t128 | 16 | 45.52-74.95 |
| fork_5dd7dbf | P8 | t128 | 16 | 73.18-74.83 |
| quixi_upstream | P8 | t128 | 16 | 16.85-17.39 |
| fork_5dd7dbf | ALL | t16 | 4 | 168.82-168.94 |
| fork_5dd7dbf | P8 | t16 | 4 | 151.27-157.16 |
| quixi_upstream | P8 | t16 | 4 | 61.71-63.70 |
| fork_5dd7dbf | ALL | t16 | 16 | 451.54-452.90 |
| fork_5dd7dbf | P8 | t16 | 16 | 410.25-472.45 |
| quixi_upstream | P8 | t16 | 16 | 119.41-138.46 |

**Weight precision vs quality** (FP32 model with its linear weights rounded per 32-block; cosine of the final embedding against the unrounded model)

| scheme | text | vision | audio | min cosine | worst workload |
|---|---|---|---|---:|---|
| q8_all | q8 | q8 | q8 | 0.99932 | a5 |
| q4_text | q4 | q8 | q8 | 0.96436 | t128 |
| q4_vision | q8 | q4 | q8 | 0.98767 | i70 |
| q4_audio | q8 | q8 | q4 | 0.93288 | a5 |
| q4_all | q4 | q4 | q4 | 0.91563 | a15 |

