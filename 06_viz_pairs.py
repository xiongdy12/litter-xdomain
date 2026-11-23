#!/usr/bin/env python3
"""
06_viz_pairs.py - Mosaik contoh pasangan duplikat untuk dicek mata (dan bahan gambar paper).

Per dataset dibuat satu JPG berisi N pasangan acak (kiri-kanan) dari kategori:
  * lintas_split_resmi : duplikat terverifikasi yang split resminya berbeda
  * lintas_grup        : duplikat terverifikasi antar grup berbeda (mis. RoLID antar rekaman)
Setiap pasangan diberi teks: cosine, jumlah inlier ORB (-1 = diterima otomatis tanpa ORB),
split resmi, dan grup asal.

Output: ~/data/litter/audit/viz/<dataset>_<kategori>.jpg
Pemakaian: python 06_viz_pairs.py [--n 8] [--seed 0]
"""
import argparse
import csv
import os
import random
from pathlib import Path

from PIL import Image, ImageDraw

D = Path(os.path.expanduser("~/data/litter"))
H, A = D / "harmonized", D / "audit"


def tile(uid, h=300):
    im = Image.open(H / "images" / f"{uid}.jpg").convert("RGB")
    return im.resize((max(1, round(im.width * h / im.height)), h))


def mosaic(pairs, meta, out, h=300):
    rows = []
    for p in pairs:
        a, b = tile(p["uid_a"], h), tile(p["uid_b"], h)
        ma, mb = meta[p["uid_a"]], meta[p["uid_b"]]
        bar = 46
        canvas = Image.new("RGB", (a.width + b.width + 10, h + bar), "white")
        canvas.paste(a, (0, bar))
        canvas.paste(b, (a.width + 10, bar))
        d = ImageDraw.Draw(canvas)
        d.text((4, 2), f"cos {float(p['cos']):.3f} | ORB inlier {p['inliers']} | "
                       f"split resmi: {ma['official_split']} vs {mb['official_split']}", fill="black")
        d.text((4, 22), f"grup: {ma['group0'].split(':', 1)[1][:40]}  vs  {mb['group0'].split(':', 1)[1][:40]}", fill="black")
        rows.append(canvas)
    W = max(r.width for r in rows)
    out_im = Image.new("RGB", (W, sum(r.height + 8 for r in rows)), (200, 200, 200))
    y = 0
    for r in rows:
        out_im.paste(r, (0, y))
        y += r.height + 8
    s = min(1.0, 1800 / out_im.width)
    if s < 1:
        out_im = out_im.resize((round(out_im.width * s), round(out_im.height * s)))
    out_im.save(out, quality=85)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    meta = {r["uid"]: r for r in csv.DictReader(open(A / "meta_split.csv"))}
    pairs = [p for p in csv.DictReader(open(A / "pairs.csv")) if p["dup"] == "1" and p["ds_a"] == p["ds_b"]]
    (A / "viz").mkdir(exist_ok=True)
    rnd = random.Random(args.seed)
    for ds in sorted({p["ds_a"] for p in pairs}):
        P = [p for p in pairs if p["ds_a"] == ds]
        cats = {
            "lintas_split_resmi": [p for p in P if meta[p["uid_a"]]["official_split"] != meta[p["uid_b"]]["official_split"]],
            "lintas_grup": [p for p in P if meta[p["uid_a"]]["group0"] != meta[p["uid_b"]]["group0"]],
        }
        for cat, L in cats.items():
            if not L:
                continue
            auto = sum(1 for p in L if p["inliers"] == "-1")
            print(f"{ds:14s} {cat:20s}: {len(L):6d} pasangan (diterima otomatis tanpa ORB: {auto})")
            out = A / "viz" / f"{ds}_{cat}.jpg"
            mosaic(rnd.sample(L, min(args.n, len(L))), meta, out)
    print(f"\nmosaik -> {A / 'viz'}")


if __name__ == "__main__":
    main()
