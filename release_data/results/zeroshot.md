# Zero-shot vs supervised (mAP50, test split baru)

| Model / prompt | TACO | PlastOPol | CoxPWD | BePLi | pLitter | RoLID |
|---|---|---|---|---|---|---|
| yoloworld p1 | 0.278 | 0.463 | 0.274 | 0.024 | 0.043 | 0.000 |
| yoloworld p2 | 0.297 | 0.513 | 0.375 | 0.038 | 0.023 | 0.000 |
| yoloworld p3 | 0.427 | 0.530 | 0.264 | 0.099 | 0.058 | 0.000 |
| yoloe p1 | 0.279 | 0.302 | 0.111 | 0.132 | 0.200 | 0.009 |
| yoloe p2 | 0.351 | 0.370 | 0.244 | 0.138 | 0.163 | 0.003 |
| yoloe p3 | 0.421 | 0.360 | 0.179 | 0.128 | 0.122 | 0.001 |
| gdino p1 | 0.557 | 0.780 | 0.458 | 0.217 | 0.314 | 0.006 |
| gdino p2 | 0.526 | 0.727 | 0.450 | 0.247 | 0.282 | 0.001 |
| gdino p3 | 0.460 | 0.618 | 0.404 | 0.116 | 0.089 | 0.007 |
| gdino_t2 p1 | 0.364 | 0.421 | 0.277 | 0.218 | 0.277 | 0.015 |
| gdino_t3 p1 | 0.228 | 0.237 | 0.226 | 0.175 | 0.208 | 0.011 |
| supervised lintas dataset terbaik (YOLO11m) | 0.478 (PlastOPol) | 0.640 (TACO) | 0.439 (PlastOPol) | 0.245 (pLitter) | 0.266 (TACO) | 0.037 (pLitter) |
| supervised in-domain (YOLO11m) | 0.573 | 0.786 | 0.815 | 0.542 | 0.643 | 0.797 |
