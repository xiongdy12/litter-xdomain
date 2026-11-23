#!/usr/bin/env python3
"""
08_analyze.py - Matriks lintas dataset (rata-rata +- sd antar seed) dan uji hipotesis skala objek.

  1. Matriks mAP (baris = latih, kolom = uji), mean +- sd seed.
  2. Rasio transfer R(s->t) = mAP(s->t) / mAP(t->t)   (dinormalisasi terhadap in-domain target).
  3. Ukuran objek tiap dataset = median sqrt(w*h) box ternormalisasi (dari label train).
  4. Uji: apakah R turun seiring makin jauhnya skala objek |log(size_s/size_t)|?
       - korelasi Spearman di 30 pasangan lintas dataset
       - regresi: log(R) ~ |gap skala| + sama_sudut_pandang + sama_wilayah
       - permutation test (acak y antar pasangan) untuk p-value regresi.
         CATATAN: 30 pasangan tidak independen (berbagi dataset), jadi p-value ini indikatif saja.

Pemakaian:  python 08_analyze.py [--metric map50|map5095] [--model yolo11m] [--imgsz 1024]
Output:     work/results/analysis_<metric>.md
"""
import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

D = Path(os.path.expanduser("~/data/litter"))
W = Path(os.path.expanduser("~/litter-xdomain/work"))
DS = ["taco", "plastopol", "coxpwd", "bepli", "plitterstreet", "rolid"]
SHORT = {"taco": "TACO", "plastopol": "PlastOPol", "coxpwd": "CoxPWD", "bepli": "BePLi",
         "plitterstreet": "pLitter", "rolid": "RoLID"}
VIEW = {"taco": "hand", "plastopol": "hand", "coxpwd": "hand", "bepli": "hand",
        "plitterstreet": "vehicle", "rolid": "vehicle"}
REGION = {"taco": "EU", "rolid": "EU", "coxpwd": "Asia", "bepli": "Asia", "plitterstreet": "Asia",
          "plastopol": "Global"}


def object_scale(ds):
    sizes = []
    for p in (D / "splits" / f"{ds}_train.txt").read_text().split():
        lab = D / "harmonized" / "labels" / (Path(p).stem + ".txt")
        for ln in lab.read_text().splitlines():
            v = ln.split()
            if len(v) == 5:
                sizes.append(np.sqrt(float(v[3]) * float(v[4])))
    return float(np.median(sizes)), len(sizes)


def fit(y, X):
    X1 = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    pred = X1 @ beta
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return beta, r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="map50", choices=["map50", "map5095"])
    ap.add_argument("--model", default="yolo11m")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--n-perm", type=int, default=10000)
    args = ap.parse_args()

    vals = defaultdict(list)
    for r in csv.DictReader(open(W / "results" / "results.csv")):
        if r["model"] != args.model or int(r["imgsz_eval"]) != args.imgsz or int(r["imgsz_train"]) != args.imgsz:
            continue
        vals[(r["train_ds"], r["test_ds"])].append(float(r[args.metric]))
    out = []

    def say(s=""):
        print(s)
        out.append(s)

    M = {k: np.mean(v) for k, v in vals.items()}
    say(f"# Matriks lintas dataset - {args.metric} ({args.model}, imgsz {args.imgsz})\n")
    say("| Latih \\ Uji | " + " | ".join(SHORT[d] for d in DS) + " |")
    say("|---" * (len(DS) + 1) + "|")
    for s in DS:
        cells = []
        for t in DS:
            v = vals.get((s, t), [])
            if not v:
                cells.append("-")
                continue
            c = f"{np.mean(v):.3f}" + (f" ± {np.std(v, ddof=1):.3f}" if len(v) > 1 else "") + f" (n={len(v)})"
            cells.append(f"**{c}**" if s == t else c)
        say(f"| {SHORT[s]} | " + " | ".join(cells) + " |")

    scale = {}
    say("\n## Skala objek (median sqrt(w*h) ternormalisasi, label train)\n")
    for d in DS:
        scale[d], n = object_scale(d)
        say(f"- {SHORT[d]}: {scale[d]:.4f}  ({n} box)")

    rows = []
    for s in DS:
        for t in DS:
            if s == t or (s, t) not in M or (t, t) not in M:
                continue
            R = M[(s, t)] / max(M[(t, t)], 1e-6)
            rows.append(dict(s=s, t=t, R=R, gap=abs(np.log(scale[s] / scale[t])),
                             signed=np.log(scale[s] / scale[t]),
                             view=int(VIEW[s] == VIEW[t]), region=int(REGION[s] == REGION[t])))
    if len(rows) < 6:
        say("\n(belum cukup pasangan untuk analisis)")
    else:
        R = np.array([r["R"] for r in rows])
        gap = np.array([r["gap"] for r in rows])
        sg = np.array([r["signed"] for r in rows])
        view = np.array([r["view"] for r in rows])
        reg = np.array([r["region"] for r in rows])
        y = np.log(R + 0.01)
        say(f"\n## Uji hipotesis skala ({len(rows)} pasangan lintas dataset)\n")
        rho, p = stats.spearmanr(gap, R)
        say(f"- Spearman(|gap skala|, rasio transfer) = {rho:.3f}, p = {p:.2g}")
        rho2, p2 = stats.spearmanr(sg, R)
        say(f"- Spearman(log(size_latih/size_uji), rasio transfer) = {rho2:.3f}, p = {p2:.2g}"
            "  (rho negatif = latih di objek lebih BESAR dari target -> transfer lebih buruk; uji asimetri)")
        X = np.column_stack([gap, view, reg])
        beta, r2 = fit(y, X)
        names = ["intersep", "|gap skala|", "sama sudut pandang", "sama wilayah"]
        # permutation test per koefisien (acak baris y)
        rng = np.random.default_rng(0)
        perm = np.array([fit(rng.permutation(y), X)[0] for _ in range(args.n_perm)])
        say(f"- Regresi log(R+0.01) ~ |gap| + sama_sudut_pandang + sama_wilayah:  R² = {r2:.3f}")
        for i, n in enumerate(names):
            pv = (np.abs(perm[:, i]) >= abs(beta[i])).mean() if i > 0 else float("nan")
            say(f"    {n:20s} koef = {beta[i]:+.3f}" + (f"   p(perm) = {pv:.4f}" if i > 0 else ""))
        b_gap, r2_gap = fit(y, gap[:, None])
        b_vr, r2_vr = fit(y, np.column_stack([view, reg]))
        say(f"- R² hanya |gap skala| = {r2_gap:.3f} | R² hanya sudut pandang + wilayah = {r2_vr:.3f}")
        say("\n| Latih -> Uji | rasio transfer | gap skala | sama view | sama wilayah |\n|---|---|---|---|---|")
        for r in sorted(rows, key=lambda r: -r["R"]):
            say(f"| {SHORT[r['s']]} -> {SHORT[r['t']]} | {r['R']:.3f} | {r['gap']:.2f} | {r['view']} | {r['region']} |")

    path = W / "results" / f"analysis_{args.model}_{args.metric}.md"
    path.write_text("\n".join(out) + "\n")
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
