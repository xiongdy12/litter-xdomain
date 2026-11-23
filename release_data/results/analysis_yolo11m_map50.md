# Matriks lintas dataset - map50 (yolo11m, imgsz 1024)

| Latih \ Uji | TACO | PlastOPol | CoxPWD | BePLi | pLitter | RoLID |
|---|---|---|---|---|---|---|
| TACO | **0.573 ± 0.005 (n=3)** | 0.640 ± 0.013 (n=3) | 0.293 ± 0.019 (n=3) | 0.164 ± 0.014 (n=3) | 0.266 ± 0.008 (n=3) | 0.001 ± 0.000 (n=3) |
| PlastOPol | 0.478 ± 0.011 (n=3) | **0.786 ± 0.008 (n=3)** | 0.439 ± 0.003 (n=3) | 0.107 ± 0.006 (n=3) | 0.134 ± 0.020 (n=3) | 0.000 ± 0.000 (n=3) |
| CoxPWD | 0.184 ± 0.012 (n=3) | 0.391 ± 0.014 (n=3) | **0.815 ± 0.008 (n=3)** | 0.011 ± 0.001 (n=3) | 0.010 ± 0.004 (n=3) | 0.000 ± 0.000 (n=3) |
| BePLi | 0.334 ± 0.019 (n=3) | 0.310 ± 0.054 (n=3) | 0.135 ± 0.010 (n=3) | **0.542 ± 0.002 (n=3)** | 0.236 ± 0.017 (n=3) | 0.011 ± 0.002 (n=3) |
| pLitter | 0.179 ± 0.029 (n=3) | 0.075 ± 0.028 (n=3) | 0.039 ± 0.008 (n=3) | 0.245 ± 0.021 (n=3) | **0.643 ± 0.006 (n=3)** | 0.037 ± 0.009 (n=3) |
| RoLID | 0.004 ± 0.003 (n=3) | 0.005 ± 0.002 (n=3) | 0.000 ± 0.000 (n=3) | 0.017 ± 0.004 (n=3) | 0.065 ± 0.018 (n=3) | **0.797 ± 0.011 (n=3)** |

## Skala objek (median sqrt(w*h) ternormalisasi, label train)

- TACO: 0.0587  (3422 box)
- PlastOPol: 0.1460  (3719 box)
- CoxPWD: 0.2296  (5226 box)
- BePLi: 0.0155  (84176 box)
- pLitter: 0.0272  (63537 box)
- RoLID: 0.0147  (15236 box)

## Uji hipotesis skala (30 pasangan lintas dataset)

- Spearman(|gap skala|, rasio transfer) = -0.514, p = 0.0036
- Spearman(log(size_latih/size_uji), rasio transfer) = -0.133, p = 0.48  (rho negatif = latih di objek lebih BESAR dari target -> transfer lebih buruk; uji asimetri)
- Regresi log(R+0.01) ~ |gap| + sama_sudut_pandang + sama_wilayah:  R² = 0.478
    intersep             koef = -1.882
    |gap skala|          koef = -0.734   p(perm) = 0.0282
    sama sudut pandang   koef = +1.592   p(perm) = 0.0044
    sama wilayah         koef = -0.026   p(perm) = 0.9660
- R² hanya |gap skala| = 0.191 | R² hanya sudut pandang + wilayah = 0.317

| Latih -> Uji | rasio transfer | gap skala | sama view | sama wilayah |
|---|---|---|---|---|
| PlastOPol -> TACO | 0.834 | 0.91 | 1 | 0 |
| TACO -> PlastOPol | 0.814 | 0.91 | 1 | 0 |
| BePLi -> TACO | 0.582 | 1.33 | 1 | 0 |
| PlastOPol -> CoxPWD | 0.539 | 0.45 | 1 | 0 |
| CoxPWD -> PlastOPol | 0.497 | 0.45 | 1 | 0 |
| pLitter -> BePLi | 0.452 | 0.56 | 0 | 1 |
| TACO -> pLitter | 0.415 | 0.77 | 0 | 0 |
| BePLi -> PlastOPol | 0.395 | 2.24 | 1 | 0 |
| BePLi -> pLitter | 0.368 | 0.56 | 0 | 1 |
| TACO -> CoxPWD | 0.359 | 1.36 | 1 | 0 |
| CoxPWD -> TACO | 0.321 | 1.36 | 1 | 0 |
| pLitter -> TACO | 0.312 | 0.77 | 0 | 0 |
| TACO -> BePLi | 0.303 | 1.33 | 1 | 0 |
| PlastOPol -> pLitter | 0.209 | 1.68 | 0 | 0 |
| PlastOPol -> BePLi | 0.198 | 2.24 | 1 | 0 |
| BePLi -> CoxPWD | 0.166 | 2.70 | 1 | 1 |
| RoLID -> pLitter | 0.100 | 0.61 | 1 | 0 |
| pLitter -> PlastOPol | 0.096 | 1.68 | 0 | 0 |
| pLitter -> CoxPWD | 0.048 | 2.13 | 0 | 1 |
| pLitter -> RoLID | 0.047 | 0.61 | 1 | 0 |
| RoLID -> BePLi | 0.031 | 0.05 | 0 | 0 |
| CoxPWD -> BePLi | 0.020 | 2.70 | 1 | 1 |
| CoxPWD -> pLitter | 0.015 | 2.13 | 0 | 1 |
| BePLi -> RoLID | 0.014 | 0.05 | 0 | 0 |
| RoLID -> TACO | 0.008 | 1.38 | 0 | 1 |
| RoLID -> PlastOPol | 0.006 | 2.29 | 0 | 0 |
| TACO -> RoLID | 0.001 | 1.38 | 0 | 1 |
| RoLID -> CoxPWD | 0.001 | 2.75 | 0 | 0 |
| PlastOPol -> RoLID | 0.000 | 2.29 | 0 | 0 |
| CoxPWD -> RoLID | 0.000 | 2.75 | 0 | 0 |
