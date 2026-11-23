# Matriks lintas dataset - map5095 (yolo11m, imgsz 1024)

| Latih \ Uji | TACO | PlastOPol | CoxPWD | BePLi | pLitter | RoLID |
|---|---|---|---|---|---|---|
| TACO | **0.429 ± 0.003 (n=3)** | 0.478 ± 0.006 (n=3) | 0.080 ± 0.006 (n=3) | 0.072 ± 0.006 (n=3) | 0.138 ± 0.004 (n=3) | 0.000 ± 0.000 (n=3) |
| PlastOPol | 0.351 ± 0.008 (n=3) | **0.640 ± 0.003 (n=3)** | 0.135 ± 0.001 (n=3) | 0.050 ± 0.002 (n=3) | 0.072 ± 0.011 (n=3) | 0.000 ± 0.000 (n=3) |
| CoxPWD | 0.058 ± 0.005 (n=3) | 0.146 ± 0.005 (n=3) | **0.473 ± 0.004 (n=3)** | 0.004 ± 0.000 (n=3) | 0.003 ± 0.003 (n=3) | 0.000 ± 0.000 (n=3) |
| BePLi | 0.253 ± 0.013 (n=3) | 0.232 ± 0.044 (n=3) | 0.034 ± 0.003 (n=3) | **0.260 ± 0.000 (n=3)** | 0.132 ± 0.009 (n=3) | 0.003 ± 0.000 (n=3) |
| pLitter | 0.124 ± 0.023 (n=3) | 0.052 ± 0.020 (n=3) | 0.010 ± 0.002 (n=3) | 0.110 ± 0.009 (n=3) | **0.400 ± 0.003 (n=3)** | 0.008 ± 0.002 (n=3) |
| RoLID | 0.002 ± 0.002 (n=3) | 0.002 ± 0.001 (n=3) | 0.000 ± 0.000 (n=3) | 0.005 ± 0.002 (n=3) | 0.023 ± 0.007 (n=3) | **0.333 ± 0.006 (n=3)** |

## Skala objek (median sqrt(w*h) ternormalisasi, label train)

- TACO: 0.0587  (3422 box)
- PlastOPol: 0.1460  (3719 box)
- CoxPWD: 0.2296  (5226 box)
- BePLi: 0.0155  (84176 box)
- pLitter: 0.0272  (63537 box)
- RoLID: 0.0147  (15236 box)

## Uji hipotesis skala (30 pasangan lintas dataset)

- Spearman(|gap skala|, rasio transfer) = -0.473, p = 0.0082
- Spearman(log(size_latih/size_uji), rasio transfer) = -0.127, p = 0.5  (rho negatif = latih di objek lebih BESAR dari target -> transfer lebih buruk; uji asimetri)
- Regresi log(R+0.01) ~ |gap| + sama_sudut_pandang + sama_wilayah:  R² = 0.400
    intersep             koef = -2.104
    |gap skala|          koef = -0.667   p(perm) = 0.0433
    sama sudut pandang   koef = +1.372   p(perm) = 0.0145
    sama wilayah         koef = -0.171   p(perm) = 0.7949
- R² hanya |gap skala| = 0.168 | R² hanya sudut pandang + wilayah = 0.264

| Latih -> Uji | rasio transfer | gap skala | sama view | sama wilayah |
|---|---|---|---|---|
| PlastOPol -> TACO | 0.819 | 0.91 | 1 | 0 |
| TACO -> PlastOPol | 0.748 | 0.91 | 1 | 0 |
| BePLi -> TACO | 0.590 | 1.33 | 1 | 0 |
| pLitter -> BePLi | 0.421 | 0.56 | 0 | 1 |
| BePLi -> PlastOPol | 0.363 | 2.24 | 1 | 0 |
| TACO -> pLitter | 0.346 | 0.77 | 0 | 0 |
| BePLi -> pLitter | 0.329 | 0.56 | 0 | 1 |
| pLitter -> TACO | 0.290 | 0.77 | 0 | 0 |
| PlastOPol -> CoxPWD | 0.284 | 0.45 | 1 | 0 |
| TACO -> BePLi | 0.276 | 1.33 | 1 | 0 |
| CoxPWD -> PlastOPol | 0.228 | 0.45 | 1 | 0 |
| PlastOPol -> BePLi | 0.192 | 2.24 | 1 | 0 |
| PlastOPol -> pLitter | 0.179 | 1.68 | 0 | 0 |
| TACO -> CoxPWD | 0.169 | 1.36 | 1 | 0 |
| CoxPWD -> TACO | 0.134 | 1.36 | 1 | 0 |
| pLitter -> PlastOPol | 0.082 | 1.68 | 0 | 0 |
| BePLi -> CoxPWD | 0.072 | 2.70 | 1 | 1 |
| RoLID -> pLitter | 0.057 | 0.61 | 1 | 0 |
| pLitter -> RoLID | 0.024 | 0.61 | 1 | 0 |
| pLitter -> CoxPWD | 0.020 | 2.13 | 0 | 1 |
| RoLID -> BePLi | 0.020 | 0.05 | 0 | 0 |
| CoxPWD -> BePLi | 0.015 | 2.70 | 1 | 1 |
| BePLi -> RoLID | 0.008 | 0.05 | 0 | 0 |
| CoxPWD -> pLitter | 0.008 | 2.13 | 0 | 1 |
| RoLID -> PlastOPol | 0.004 | 2.29 | 0 | 0 |
| RoLID -> TACO | 0.004 | 1.38 | 0 | 1 |
| TACO -> RoLID | 0.001 | 1.38 | 0 | 1 |
| RoLID -> CoxPWD | 0.000 | 2.75 | 0 | 0 |
| PlastOPol -> RoLID | 0.000 | 2.29 | 0 | 0 |
| CoxPWD -> RoLID | 0.000 | 2.75 | 0 | 0 |
