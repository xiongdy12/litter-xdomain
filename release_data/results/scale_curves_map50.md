# Kurva skala - map50 (yolo11m, seed 0, dilatih 1024)

Skala objek (median): TACO 0.0587, PlastOPol 0.1460, CoxPWD 0.2296, BePLi 0.0155, pLitter 0.0272, RoLID 0.0147

| Latih -> Uji | imgsz* prediksi | grid terdekat | mAP@1024 | mAP@scale-match | mAP oracle (imgsz) | in-domain uji | pulih scale-match | pulih oracle |
|---|---|---|---|---|---|---|---|---|
| TACO -> PlastOPol | 412 | 320 | 0.654 | 0.657 | 0.679 (640) | 0.794 | +2.3% | +18.0% |
| TACO -> CoxPWD | 262 | 320 | 0.311 | 0.386 | 0.386 (320) | 0.814 | +14.8% | +14.8% |
| TACO -> BePLi | 3878 | 3072 | 0.150 | 0.124 | 0.150 (1024) | 0.542 | -6.6% | +0.0% |
| TACO -> pLitter | 2211 | 2048 | 0.257 | 0.232 | 0.257 (1024) | 0.639 | -6.6% | +0.0% |
| TACO -> RoLID | 4082 | 3072 | 0.001 | 0.000 | 0.001 (1024) | 0.785 | -0.1% | +0.0% |
| PlastOPol -> TACO | 2546 | 2048 | 0.466 | 0.471 | 0.479 (1536) | 0.577 | +5.0% | +12.3% |
| PlastOPol -> CoxPWD | 651 | 640 | 0.438 | 0.446 | 0.453 (320) | 0.814 | +2.3% | +4.1% |
| PlastOPol -> BePLi | 9641 | 3072 | 0.108 | 0.073 | 0.108 (1024) | 0.542 | -8.0% | +0.0% |
| PlastOPol -> pLitter | 5497 | 3072 | 0.154 | 0.168 | 0.169 (2048) | 0.639 | +2.9% | +2.9% |
| PlastOPol -> RoLID | 10150 | 3072 | 0.000 | 0.000 | 0.000 (640) | 0.785 | -0.0% | +0.0% |
| CoxPWD -> TACO | 4004 | 2048 | 0.171 | 0.168 | 0.186 (1536) | 0.577 | -0.5% | +3.8% |
| CoxPWD -> PlastOPol | 1610 | 1536 | 0.399 | 0.364 | 0.404 (640) | 0.794 | -8.7% | +1.4% |
| CoxPWD -> BePLi | 15162 | 3072 | 0.010 | 0.011 | 0.011 (3072) | 0.542 | +0.2% | +0.2% |
| CoxPWD -> pLitter | 8644 | 3072 | 0.014 | 0.017 | 0.017 (3072) | 0.639 | +0.5% | +0.5% |
| CoxPWD -> RoLID | 15963 | 3072 | 0.000 | 0.000 | 0.000 (320) | 0.785 | +0.0% | +0.0% |
| BePLi -> TACO | 270 | 320 | 0.331 | 0.384 | 0.384 (320) | 0.577 | +21.5% | +21.5% |
| BePLi -> PlastOPol | 109 | 320 | 0.346 | 0.571 | 0.571 (320) | 0.794 | +50.1% | +50.1% |
| BePLi -> CoxPWD | 69 | 320 | 0.146 | 0.351 | 0.351 (320) | 0.814 | +30.7% | +30.7% |
| BePLi -> pLitter | 584 | 640 | 0.239 | 0.253 | 0.253 (640) | 0.639 | +3.6% | +3.6% |
| BePLi -> RoLID | 1078 | 1024 | 0.013 | 0.013 | 0.013 (1024) | 0.785 | +0.0% | +0.0% |
| pLitter -> TACO | 474 | 640 | 0.202 | 0.251 | 0.299 (320) | 0.577 | +13.1% | +26.0% |
| pLitter -> PlastOPol | 191 | 320 | 0.079 | 0.292 | 0.292 (320) | 0.794 | +29.7% | +29.7% |
| pLitter -> CoxPWD | 121 | 320 | 0.038 | 0.152 | 0.152 (320) | 0.814 | +14.7% | +14.7% |
| pLitter -> BePLi | 1796 | 2048 | 0.240 | 0.267 | 0.267 (2048) | 0.542 | +8.8% | +8.8% |
| pLitter -> RoLID | 1891 | 2048 | 0.034 | 0.020 | 0.034 (1024) | 0.785 | -1.8% | +0.0% |
| RoLID -> TACO | 257 | 320 | 0.004 | 0.050 | 0.050 (320) | 0.577 | +8.0% | +8.0% |
| RoLID -> PlastOPol | 103 | 320 | 0.002 | 0.034 | 0.034 (320) | 0.794 | +4.0% | +4.0% |
| RoLID -> CoxPWD | 66 | 320 | 0.000 | 0.047 | 0.047 (320) | 0.814 | +5.8% | +5.8% |
| RoLID -> BePLi | 973 | 1024 | 0.012 | 0.012 | 0.013 (1536) | 0.542 | +0.0% | +0.2% |
| RoLID -> pLitter | 555 | 640 | 0.068 | 0.083 | 0.083 (640) | 0.639 | +2.6% | +2.6% |

## Ringkasan pemulihan celah (median antar pasangan; celah < 0.02 diabaikan)

- semua pasangan                            : n=30 | pulih scale-match median +2.4% | oracle median +3.7% | mAP rata2 0.163 -> 0.197 (oracle 0.205, in-domain 0.692)
- sudut pandang sama                        : n=14 | pulih scale-match median +2.3% | oracle median +4.0% | mAP rata2 0.259 -> 0.294 (oracle 0.306, in-domain 0.686)
- sudut pandang beda                        : n=16 | pulih scale-match median +3.3% | oracle median +3.3% | mAP rata2 0.078 -> 0.112 (oracle 0.117, in-domain 0.697)
- target objek kecil (BePLi/pLitter/RoLID)  : n=15 | pulih scale-match median +0.0% | oracle median +0.0% | mAP rata2 0.087 -> 0.085 (oracle 0.092, in-domain 0.655)
- target RoLID                              : n= 5 | pulih scale-match median -0.0% | oracle median +0.0% | mAP rata2 0.010 -> 0.007 (oracle 0.010, in-domain 0.785)
