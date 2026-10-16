**Warm p50 latency, ms — 8 threads on the 8 P-cores (P8)**

| workload | this fork | Sentence Transformers FP32 | ONNX Community Q8 | QuixiAI upstream Q8 | fork vs fastest other, same budget |
|---|---:|---:|---:|---:|---:|
| text, 16 tokens | **8.6** | 20.4 | 36.8 | 17.4 | 2.03x |
| text, 128 tokens | **29.5** | 77.5 | 84.7 | 117 | 2.63x |
| text, 512 tokens | **134** | 250 | 292 | 556 | 1.87x |
| image, 70 tokens (336x480) | **152** | 341 | 1,368 | 3,909 | 2.24x |
| image, 280 tokens (672x960) | **942** | 1,829 | 3,028 | 32,082 | 1.94x |
| image, 1120 tokens (1344x1920) | **10,140** | 13,646 | 14,321 | 552,466 | 1.35x |
| audio, 5 s | **82.1** | 361 | 1,297 | 1,320 | 4.40x |
| audio, 15 s | **241** | 654 | 2,014 | 3,935 | 2.71x |
| audio, 30 s | **523** | 1,169 | 2,448 | 8,202 | 2.24x |
| video, 4 frames | **1,482** | 3,265 | 4,332 | 42,652 | 2.20x |
| video, 8 frames | **3,096** | 6,946 | 7,465 | 87,577 | 2.24x |
| video, 16 frames | **6,405** | 16,418 | 14,392 | 161,199 | 2.25x |
| text + image | **1,114** | 1,707 | 3,050 | 33,015 | 1.53x |
| text + audio | **258** | 635 | 1,997 | 3,991 | 2.46x |
| text + image + audio | **1,295** | 2,495 | 4,454 | 36,394 | 1.93x |
| text + video | **3,137** | 7,276 | 7,519 | 87,103 | 2.32x |

**Warm p50 latency, ms — all 28 hardware threads (ALL)**

| workload | this fork | Sentence Transformers FP32 | ONNX Community Q8 | QuixiAI upstream Q8 | fork vs fastest other, same budget |
|---|---:|---:|---:|---:|---:|
| text, 16 tokens | **8.9** | 28.2 | 68.0 | 21.9 | 2.47x |
| text, 128 tokens | **29.3** | 96.0 | 158 | 108 | 3.27x |
| text, 512 tokens | **128** | 344 | 408 | 398 | 2.69x |
| image, 70 tokens (336x480) | **139** | 419 | 1,824 | 2,548 | 3.03x |
| image, 280 tokens (672x960) | **783** | 3,709 | 3,294 | – | 4.21x |
| image, 1120 tokens (1344x1920) | **7,678** | 25,366 | 12,142 | – | 1.58x |
| audio, 5 s | **84.6** | 1,852 | 1,648 | 987 | 11.66x |
| audio, 15 s | **237** | 4,126 | 2,470 | 2,911 | 10.40x |
| audio, 30 s | **441** | 18,259 | 2,995 | – | 6.79x |
| video, 4 frames | **1,271** | 12,703 | 4,579 | – | 3.60x |
| video, 8 frames | **2,560** | 18,955 | 7,219 | – | 2.82x |
| video, 16 frames | **5,313** | 23,168 | 14,481 | – | 2.73x |
| text + image | **849** | 10,109 | 3,314 | – | 3.90x |
| text + audio | **254** | 6,109 | 2,384 | 2,901 | 9.37x |
| text + image + audio | **1,070** | 11,795 | 4,987 | – | 4.66x |
| text + video | **2,629** | 21,358 | 7,142 | – | 2.72x |

**Best thread budget per backend (P8 or ALL), p50 ms**

| workload | this fork | Sentence Transformers FP32 | ONNX Community Q8 | QuixiAI upstream Q8 | fork vs fastest other |
|---|---:|---:|---:|---:|---:|
| text, 16 tokens | **8.6 (P8)** | 20.4 (P8) | 36.8 (P8) | 17.4 (P8) | 2.03x vs upstream |
| text, 128 tokens | **29.3 (ALL)** | 77.5 (P8) | 84.7 (P8) | 108 (ALL) | 2.64x vs ST FP32 |
| text, 512 tokens | **128 (ALL)** | 250 (P8) | 292 (P8) | 398 (ALL) | 1.95x vs ST FP32 |
| image, 70 tokens (336x480) | **139 (ALL)** | 341 (P8) | 1,368 (P8) | 2,548 (ALL) | 2.46x vs ST FP32 |
| image, 280 tokens (672x960) | **783 (ALL)** | 1,829 (P8) | 3,028 (P8) | 32,082 (P8) | 2.34x vs ST FP32 |
| image, 1120 tokens (1344x1920) | **7,678 (ALL)** | 13,646 (P8) | 12,142 (ALL) | 552,466 (P8) | 1.58x vs ONNX Q8 |
| audio, 5 s | **82.1 (P8)** | 361 (P8) | 1,297 (P8) | 987 (ALL) | 4.40x vs ST FP32 |
| audio, 15 s | **237 (ALL)** | 654 (P8) | 2,014 (P8) | 2,911 (ALL) | 2.76x vs ST FP32 |
| audio, 30 s | **441 (ALL)** | 1,169 (P8) | 2,448 (P8) | 8,202 (P8) | 2.65x vs ST FP32 |
| video, 4 frames | **1,271 (ALL)** | 3,265 (P8) | 4,332 (P8) | 42,652 (P8) | 2.57x vs ST FP32 |
| video, 8 frames | **2,560 (ALL)** | 6,946 (P8) | 7,219 (ALL) | 87,577 (P8) | 2.71x vs ST FP32 |
| video, 16 frames | **5,313 (ALL)** | 16,418 (P8) | 14,392 (P8) | 161,199 (P8) | 2.71x vs ONNX Q8 |
| text + image | **849 (ALL)** | 1,707 (P8) | 3,050 (P8) | 33,015 (P8) | 2.01x vs ST FP32 |
| text + audio | **254 (ALL)** | 635 (P8) | 1,997 (P8) | 2,901 (ALL) | 2.50x vs ST FP32 |
| text + image + audio | **1,070 (ALL)** | 2,495 (P8) | 4,454 (P8) | 36,394 (P8) | 2.33x vs ST FP32 |
| text + video | **2,629 (ALL)** | 7,276 (P8) | 7,142 (ALL) | 87,103 (P8) | 2.72x vs ONNX Q8 |

Geometric mean over the 16 workloads: **2.46x** faster than the fastest other backend for each workload (minimum 1.58x).

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

