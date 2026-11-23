# Kurva skala - map5095 (yolo11m, seed 0, dilatih 1024)

Skala objek (median): TACO 0.0587, PlastOPol 0.1460, CoxPWD 0.2296, BePLi 0.0155, pLitter 0.0272, RoLID 0.0147

| Latih -> Uji | imgsz* prediksi | grid terdekat | mAP@1024 | mAP@scale-match | mAP oracle (imgsz) | in-domain uji | pulih scale-match | pulih oracle |
|---|---|---|---|---|---|---|---|---|
| TACO -> PlastOPol | 412 | 320 | 0.485 | 0.481 | 0.513 (640) | 0.643 | -2.7% | +17.4% |
| TACO -> CoxPWD | 262 | 320 | 0.086 | 0.121 | 0.121 (320) | 0.470 | +9.2% | +9.2% |
| TACO -> BePLi | 3878 | 3072 | 0.065 | 0.053 | 0.065 (1024) | 0.260 | -6.3% | +0.0% |
| TACO -> pLitter | 2211 | 2048 | 0.134 | 0.131 | 0.135 (3072) | 0.399 | -1.4% | +0.2% |
| TACO -> RoLID | 4082 | 3072 | 0.000 | 0.000 | 0.000 (1024) | 0.326 | -0.1% | +0.0% |
| PlastOPol -> TACO | 2546 | 2048 | 0.342 | 0.342 | 0.348 (1536) | 0.432 | +0.6% | +6.9% |
| PlastOPol -> CoxPWD | 651 | 640 | 0.135 | 0.151 | 0.162 (320) | 0.470 | +5.0% | +8.0% |
| PlastOPol -> BePLi | 9641 | 3072 | 0.050 | 0.036 | 0.050 (1024) | 0.260 | -6.7% | +0.0% |
| PlastOPol -> pLitter | 5497 | 3072 | 0.083 | 0.102 | 0.102 (3072) | 0.399 | +6.3% | +6.3% |
| PlastOPol -> RoLID | 10150 | 3072 | 0.000 | 0.000 | 0.000 (320) | 0.326 | +0.0% | +0.0% |
| CoxPWD -> TACO | 4004 | 2048 | 0.052 | 0.052 | 0.056 (1536) | 0.432 | +0.1% | +1.1% |
| CoxPWD -> PlastOPol | 1610 | 1536 | 0.143 | 0.131 | 0.143 (1024) | 0.643 | -2.5% | +0.0% |
| CoxPWD -> BePLi | 15162 | 3072 | 0.004 | 0.005 | 0.005 (3072) | 0.260 | +0.5% | +0.5% |
| CoxPWD -> pLitter | 8644 | 3072 | 0.006 | 0.003 | 0.006 (1024) | 0.399 | -0.7% | +0.0% |
| CoxPWD -> RoLID | 15963 | 3072 | 0.000 | 0.000 | 0.000 (320) | 0.326 | +0.0% | +0.0% |
| BePLi -> TACO | 270 | 320 | 0.250 | 0.267 | 0.275 (640) | 0.432 | +9.3% | +13.7% |
| BePLi -> PlastOPol | 109 | 320 | 0.260 | 0.426 | 0.426 (320) | 0.643 | +43.5% | +43.5% |
| BePLi -> CoxPWD | 69 | 320 | 0.038 | 0.102 | 0.102 (320) | 0.470 | +14.9% | +14.9% |
| BePLi -> pLitter | 584 | 640 | 0.131 | 0.133 | 0.133 (640) | 0.399 | +0.6% | +0.6% |
| BePLi -> RoLID | 1078 | 1024 | 0.003 | 0.003 | 0.003 (1024) | 0.326 | +0.0% | +0.0% |
| pLitter -> TACO | 474 | 640 | 0.145 | 0.174 | 0.200 (320) | 0.432 | +10.1% | +19.0% |
| pLitter -> PlastOPol | 191 | 320 | 0.056 | 0.208 | 0.208 (320) | 0.643 | +25.9% | +25.9% |
| pLitter -> CoxPWD | 121 | 320 | 0.009 | 0.041 | 0.041 (320) | 0.470 | +6.9% | +6.9% |
| pLitter -> BePLi | 1796 | 2048 | 0.108 | 0.121 | 0.121 (2048) | 0.260 | +8.5% | +8.5% |
| pLitter -> RoLID | 1891 | 2048 | 0.007 | 0.004 | 0.007 (1024) | 0.326 | -1.0% | +0.0% |
| RoLID -> TACO | 257 | 320 | 0.001 | 0.015 | 0.015 (320) | 0.432 | +3.3% | +3.3% |
| RoLID -> PlastOPol | 103 | 320 | 0.001 | 0.013 | 0.013 (320) | 0.643 | +1.9% | +1.9% |
| RoLID -> CoxPWD | 66 | 320 | 0.000 | 0.011 | 0.011 (320) | 0.470 | +2.4% | +2.4% |
| RoLID -> BePLi | 973 | 1024 | 0.004 | 0.004 | 0.005 (1536) | 0.260 | +0.0% | +0.4% |
| RoLID -> pLitter | 555 | 640 | 0.025 | 0.027 | 0.027 (640) | 0.399 | +0.7% | +0.7% |

## Ringkasan pemulihan celah (median antar pasangan; celah < 0.02 diabaikan)

- semua pasangan                            : n=30 | pulih scale-match median +0.6% | oracle median +1.5% | mAP rata2 0.087 -> 0.105 (oracle 0.110, in-domain 0.422)
- sudut pandang sama                        : n=14 | pulih scale-match median +0.5% | oracle median +4.0% | mAP rata2 0.139 -> 0.157 (oracle 0.164, in-domain 0.439)
- sudut pandang beda                        : n=16 | pulih scale-match median +1.2% | oracle median +1.2% | mAP rata2 0.043 -> 0.060 (oracle 0.062, in-domain 0.407)
- target objek kecil (BePLi/pLitter/RoLID)  : n=15 | pulih scale-match median +0.0% | oracle median +0.0% | mAP rata2 0.041 -> 0.041 (oracle 0.044, in-domain 0.328)
- target RoLID                              : n= 5 | pulih scale-match median +0.0% | oracle median +0.0% | mAP rata2 0.002 -> 0.001 (oracle 0.002, in-domain 0.326)
