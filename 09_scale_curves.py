#!/usr/bin/env python3
"""
09_scale_curves.py - Seberapa banyak penurunan lintas dataset yang pulih hanya dengan menyesuaikan skala uji?

Untuk setiap pasangan (latih s -> uji t), seed 0, kurva mAP vs resolusi uji (imgsz_eval).
  baseline   : mAP di imgsz 1024 (sama dengan training)
  scale-match: resolusi diprediksi dari skala objek  imgsz* = 1024 * size_s / size_t,
               dibulatkan ke nilai grid terdekat (skala log)  -> tidak melihat hasil uji
  oracle     : resolusi terbaik di grid (batas atas; melihat hasil uji, hanya pembanding)
  pemulihan  = (mAP_metode - mAP_baseline) / (mAP_in-domain(t) - mAP_baseline)

Output: work/results/scale_curves_<metric>.md  dan  scale_curves_<metric>.png
Pemakaian: python 09_scale_curves.py [--metric map50|map5095]
"""
import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

D = Path(os.path.expanduser("~/data/litter"))
W = Path(os.path.expanduser("~/litter-xdomain/work"))
DS = ["taco", "plastopol", "coxpwd", "bepli", "plitterstreet", "rolid"]
SHORT = {"taco": "TACO", "plastopol": "PlastOPol", "coxpwd": "CoxPWD", "bepli": "BePLi",
         "plitterstreet": "pLitter", "rolid": "RoLID"}
VIEW = {"taco": "hand", "plastopol": "hand", "coxpwd": "hand", "bepli": "hand",
        "plitterstreet": "vehicle", "rolid": "vehicle"}
# palet kategorikal referensi (urutan tetap, warna mengikuti dataset LATIH)
COLOR = dict(zip(DS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
MARKER = dict(zip(DS, ["o", "s", "^", "D", "v", "P"]))


def object_scale(ds):
    sizes = []
    for p in (D / "splits" / f"{ds}_train.txt").read_text().split():
        for ln in (D / "harmonized" / "labels" / (Path(p).stem + ".txt")).read_text().splitlines():
            v = ln.split()
            if len(v) == 5:
                sizes.append(np.sqrt(float(v[3]) * float(v[4])))
    return float(np.median(sizes))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="map50", choices=["map50", "map5095"])
    ap.add_argument("--model", default="yolo11m")
    ap.add_argument("--seed", default="0")
    ap.add_argument("--base", type=int, default=1024)
    args = ap.parse_args()

    curve = defaultdict(dict)  # (s,t) -> {imgsz: mAP}
    for r in csv.DictReader(open(W / "results" / "results.csv")):
        if r["model"] != args.model or r["seed"] != args.seed or int(r["imgsz_train"]) != args.base:
            continue
        curve[(r["train_ds"], r["test_ds"])][int(r["imgsz_eval"])] = float(r[args.metric])
    scale = {d: object_scale(d) for d in DS}

    out = []

    def say(s=""):
        print(s)
        out.append(s)

    say(f"# Kurva skala - {args.metric} ({args.model}, seed {args.seed}, dilatih {args.base})\n")
    say("Skala objek (median): " + ", ".join(f"{SHORT[d]} {scale[d]:.4f}" for d in DS) + "\n")
    say("| Latih -> Uji | imgsz* prediksi | grid terdekat | mAP@1024 | mAP@scale-match | mAP oracle (imgsz) "
        "| in-domain uji | pulih scale-match | pulih oracle |")
    say("|---|---|---|---|---|---|---|---|---|")
    rec = []
    for s in DS:
        for t in DS:
            if s == t or (s, t) not in curve or args.base not in curve[(s, t)] or args.base not in curve[(t, t)]:
                continue
            c = curve[(s, t)]
            grid = sorted(c)
            pred = args.base * scale[s] / scale[t]
            near = min(grid, key=lambda g: abs(np.log(g / pred)))
            base, sm = c[args.base], c[near]
            best_sz = max(grid, key=lambda g: c[g])
            ind = curve[(t, t)][args.base]
            gap = ind - base
            r_sm = (sm - base) / gap if gap > 0.02 else np.nan
            r_or = (c[best_sz] - base) / gap if gap > 0.02 else np.nan
            rec.append(dict(s=s, t=t, base=base, sm=sm, oracle=c[best_sz], ind=ind, r_sm=r_sm, r_or=r_or,
                            near=near, pred=pred, view=VIEW[s] == VIEW[t]))
            say(f"| {SHORT[s]} -> {SHORT[t]} | {pred:.0f} | {near} | {base:.3f} | {sm:.3f} | {c[best_sz]:.3f} ({best_sz}) "
                f"| {ind:.3f} | {r_sm:+.1%} | {r_or:+.1%} |")

    if rec:
        say("\n## Ringkasan pemulihan celah (median antar pasangan; celah < 0.02 diabaikan)\n")
        for name, sel in [("semua pasangan", lambda r: True),
                          ("sudut pandang sama", lambda r: r["view"]),
                          ("sudut pandang beda", lambda r: not r["view"]),
                          ("arah PERKECIL (objek latih < objek uji)", lambda r: r["pred"] < args.base),
                          ("arah PERBESAR (objek latih > objek uji)", lambda r: r["pred"] > args.base),
                          ("arah PERKECIL, tanpa sumber RoLID", lambda r: r["pred"] < args.base and r["s"] != "rolid"),
                          ("target objek kecil (BePLi/pLitter/RoLID)", lambda r: r["t"] in ("bepli", "plitterstreet", "rolid")),
                          ("target RoLID", lambda r: r["t"] == "rolid")]:
            L = [r for r in rec if sel(r) and not np.isnan(r["r_sm"])]
            if L:
                say(f"- {name:42s}: n={len(L):2d} | pulih scale-match median {np.median([r['r_sm'] for r in L]):+.1%} "
                    f"| oracle median {np.median([r['r_or'] for r in L]):+.1%} "
                    f"| mAP rata2 {np.mean([r['base'] for r in L]):.3f} -> {np.mean([r['sm'] for r in L]):.3f} "
                    f"(oracle {np.mean([r['oracle'] for r in L]):.3f}, in-domain {np.mean([r['ind'] for r in L]):.3f})")

    # ---------------- gambar: small multiples per dataset UJI, garis per dataset LATIH
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#8a8984", "axes.labelcolor": "#52514e",
                         "xtick.color": "#52514e", "ytick.color": "#52514e", "text.color": "#0b0b0b"})
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.4), sharey=False)
    for ax, t in zip(axes.flat, DS):
        for s in DS:
            c = curve.get((s, t))
            if not c:
                continue
            xs = sorted(c)
            ys = [c[x] for x in xs]
            lw = 2.6 if s == t else 1.6
            ax.plot(xs, ys, color=COLOR[s], marker=MARKER[s], ms=5, lw=lw, label=SHORT[s] + (" (in-domain)" if s == t else ""),
                    zorder=3 if s == t else 2)
            if s != t:  # titik scale-match: penanda berongga
                pred = args.base * scale[s] / scale[t]
                near = min(xs, key=lambda g: abs(np.log(g / pred)))
                ax.plot([near], [c[near]], marker=MARKER[s], ms=10, mfc="none", mec=COLOR[s], mew=1.6, zorder=4)
        ax.set_xscale("log", base=2)
        ax.set_xticks([320, 640, 1024, 2048, 3072])
        ax.set_xticklabels(["320", "640", "1024", "2048", "3072"])
        ax.axvline(args.base, color="#c3c2b7", lw=1, ls="--", zorder=1)
        ax.grid(True, color="#e6e5e0", lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_title(f"Uji: {SHORT[t]}  (skala objek {scale[t]:.3f})", fontsize=10, loc="left")
        ax.set_xlabel("resolusi uji (px)")
        ax.set_ylabel(args.metric.replace("map", "mAP").replace("5095", "50-95"))
        ax.set_ylim(bottom=0)
    h, l = axes.flat[0].get_legend_handles_labels()
    handles = [plt.Line2D([], [], color=COLOR[d], marker=MARKER[d], lw=1.8, ms=5) for d in DS]
    handles.append(plt.Line2D([], [], color="#52514e", marker="o", ms=9, mfc="none", lw=0))
    fig.legend(handles, [f"latih: {SHORT[d]}" for d in DS] + ["resolusi scale-match"],
               loc="lower center", ncol=7, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    png = W / "results" / f"scale_curves_{args.metric}.png"
    fig.savefig(png, dpi=200)
    (W / "results" / f"scale_curves_{args.metric}.md").write_text("\n".join(out) + "\n")
    print(f"\n-> {png}\n-> {png.with_suffix('.md')}")


if __name__ == "__main__":
    main()
