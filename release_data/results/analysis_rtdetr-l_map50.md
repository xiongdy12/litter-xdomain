# Matriks lintas dataset - map50 (rtdetr-l, imgsz 1024)

| Latih \ Uji | TACO | PlastOPol | CoxPWD | BePLi | pLitter | RoLID |
|---|---|---|---|---|---|---|
| TACO | **0.503 (n=1)** | 0.661 (n=1) | 0.346 (n=1) | 0.133 (n=1) | 0.173 (n=1) | 0.001 (n=1) |
| PlastOPol | 0.506 (n=1) | **0.795 (n=1)** | 0.452 (n=1) | 0.124 (n=1) | 0.126 (n=1) | 0.000 (n=1) |
| CoxPWD | 0.154 (n=1) | 0.318 (n=1) | **0.656 (n=1)** | 0.010 (n=1) | 0.008 (n=1) | 0.000 (n=1) |
| BePLi | 0.353 (n=1) | 0.426 (n=1) | 0.174 (n=1) | **0.556 (n=1)** | 0.270 (n=1) | 0.004 (n=1) |
| pLitter | 0.151 (n=1) | 0.109 (n=1) | 0.117 (n=1) | 0.184 (n=1) | **0.598 (n=1)** | 0.061 (n=1) |
| RoLID | 0.005 (n=1) | 0.005 (n=1) | 0.006 (n=1) | 0.013 (n=1) | 0.043 (n=1) | **0.715 (n=1)** |

## Skala objek (median sqrt(w*h) ternormalisasi, label train)

- TACO: 0.0587  (3422 box)
- PlastOPol: 0.1460  (3719 box)
- CoxPWD: 0.2296  (5226 box)
- BePLi: 0.0155  (84176 box)
- pLitter: 0.0272  (63537 box)
- RoLID: 0.0147  (15236 box)

## Uji hipotesis skala (30 pasangan lintas dataset)

- Spearman(|gap skala|, rasio transfer) = -0.407, p = 0.026
- Spearman(log(size_latih/size_uji), rasio transfer) = -0.238, p = 0.21  (rho negatif = latih di objek lebih BESAR dari target -> transfer lebih buruk; uji asimetri)
- Regresi log(R+0.01) ~ |gap| + sama_sudut_pandang + sama_wilayah:  R² = 0.416
    intersep             koef = -2.090
    |gap skala|          koef = -0.575   p(perm) = 0.0945
    sama sudut pandang   koef = +1.676   p(perm) = 0.0031
    sama wilayah         koef = +0.075   p(perm) = 0.9103
- R² hanya |gap skala| = 0.115 | R² hanya sudut pandang + wilayah = 0.320

| Latih -> Uji | rasio transfer | gap skala | sama view | sama wilayah |
|---|---|---|---|---|
| PlastOPol -> TACO | 1.007 | 0.91 | 1 | 0 |
| TACO -> PlastOPol | 0.831 | 0.91 | 1 | 0 |
| BePLi -> TACO | 0.703 | 1.33 | 1 | 0 |
| PlastOPol -> CoxPWD | 0.690 | 0.45 | 1 | 0 |
| BePLi -> PlastOPol | 0.536 | 2.24 | 1 | 0 |
| TACO -> CoxPWD | 0.528 | 1.36 | 1 | 0 |
| BePLi -> pLitter | 0.451 | 0.56 | 0 | 1 |
| CoxPWD -> PlastOPol | 0.400 | 0.45 | 1 | 0 |
| pLitter -> BePLi | 0.332 | 0.56 | 0 | 1 |
| CoxPWD -> TACO | 0.307 | 1.36 | 1 | 0 |
| pLitter -> TACO | 0.301 | 0.77 | 0 | 0 |
| TACO -> pLitter | 0.289 | 0.77 | 0 | 0 |
| BePLi -> CoxPWD | 0.265 | 2.70 | 1 | 1 |
| TACO -> BePLi | 0.239 | 1.33 | 1 | 0 |
| PlastOPol -> BePLi | 0.223 | 2.24 | 1 | 0 |
| PlastOPol -> pLitter | 0.210 | 1.68 | 0 | 0 |
| pLitter -> CoxPWD | 0.178 | 2.13 | 0 | 1 |
| pLitter -> PlastOPol | 0.137 | 1.68 | 0 | 0 |
| pLitter -> RoLID | 0.086 | 0.61 | 1 | 0 |
| RoLID -> pLitter | 0.073 | 0.61 | 1 | 0 |
| RoLID -> BePLi | 0.023 | 0.05 | 0 | 0 |
| CoxPWD -> BePLi | 0.017 | 2.70 | 1 | 1 |
| CoxPWD -> pLitter | 0.013 | 2.13 | 0 | 1 |
| RoLID -> TACO | 0.011 | 1.38 | 0 | 1 |
| RoLID -> CoxPWD | 0.009 | 2.75 | 0 | 0 |
| RoLID -> PlastOPol | 0.007 | 2.29 | 0 | 0 |
| BePLi -> RoLID | 0.005 | 0.05 | 0 | 0 |
| TACO -> RoLID | 0.001 | 1.38 | 0 | 1 |
| PlastOPol -> RoLID | 0.000 | 2.29 | 0 | 0 |
| CoxPWD -> RoLID | 0.000 | 2.75 | 0 | 0 |
