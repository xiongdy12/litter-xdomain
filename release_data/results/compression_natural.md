# Eksperimen alami CoxPWD: foto lewat WhatsApp vs foto kamera asli (test split baru)

citra uji: WhatsApp 150, kamera asli 479

| Model latih | AP50 WA | AP50 asli | selisih (WA-asli) [95% CI] | AP50-95 WA | AP50-95 asli | selisih [95% CI] |
|---|---|---|---|---|---|---|
| TACO | 0.285 | 0.301 | -0.016 [-0.099, +0.081] | 0.080 | 0.083 | -0.003 [-0.029, +0.028] |
| PlastOPol | 0.411 | 0.452 | -0.040 [-0.133, +0.066] | 0.137 | 0.138 | -0.001 [-0.038, +0.037] |
| CoxPWD (in-domain) | 0.799 | 0.816 | -0.018 [-0.096, +0.056] | 0.460 | 0.478 | -0.018 [-0.081, +0.044] |
| BePLi | 0.126 | 0.146 | -0.020 [-0.068, +0.040] | 0.034 | 0.038 | -0.004 [-0.017, +0.015] |
| pLitter | 0.068 | 0.036 | +0.032 [+0.008, +0.070] | 0.017 | 0.009 | +0.008 [+0.002, +0.020] |
| RoLID | 0.002 | 0.000 | +0.002 [+0.000, +0.008] | 0.001 | 0.000 | +0.001 [+0.000, +0.003] |

(seed: [0, 1, 2]; bootstrap 1000x per kelompok; AP dihitung evaluator sendiri, 101 titik)
