#!/usr/bin/env python3
"""
10_compression.py - Apakah pengiriman foto (WhatsApp-like) merusak deteksi sampah?

Bagian A  natural  : eksperimen alami di CoxPWD - AP pada foto yang PERNAH lewat WhatsApp vs foto
                     kamera asli (Pixel / IMG), dari predictions.json yang sudah ada (tanpa GPU).
                     Selisih AP + 95% CI bootstrap (resampling citra, rata-rata antar seed).
                     CATATAN: foto WA bisa beda HP/waktu/lokasi -> perbedaan tidak murni kompresi.
Bagian B  simulate : simulasi rantai kirim pada test set yang SAMA (terkontrol):
                     L0  asli
                     WA1 sisi panjang <= 1600, JPEG q75          (mirip kualitas standar WhatsApp)
                     WA2 WA1 diulang 2x (diteruskan)
                     WA3 WA1 diulang 3x
                     LOW sisi panjang <= 800, JPEG q50           (mode hemat data / kirim ulang buruk)
                     Lalu evaluasi semua model baseline (6 dataset x seed) di tiap level.

Pemakaian:
  python 10_compression.py natural
  python 10_compression.py simulate --device 7 [--targets taco plastopol coxpwd bepli] [--seeds 0 1 2]
Output: work/results/compression_natural.md, work/results/compression_sim.csv (+ .md ringkasan)
"""
import argparse
import csv
import io
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

D = Path(os.path.expanduser("~/data/litter"))
W = Path(os.path.expanduser("~/litter-xdomain/work"))
DS = ["taco", "plastopol", "coxpwd", "bepli", "plitterstreet", "rolid"]
SHORT = {"taco": "TACO", "plastopol": "PlastOPol", "coxpwd": "CoxPWD", "bepli": "BePLi",
         "plitterstreet": "pLitter", "rolid": "RoLID"}
IOUS = np.arange(0.5, 0.96, 0.05)
CHAINS = {"L0": [], "WA1": [(1600, 75)], "WA2": [(1600, 75)] * 2, "WA3": [(1600, 75)] * 3, "LOW": [(800, 50)]}


# ------------------------------------------------------------------ evaluator AP (1 kelas, gaya COCO 101 titik)
def iou_matrix(a, b):  # a: (n,4) xyxy, b: (m,4)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0]); iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2]); iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-9)


def match_image(pred, gt):
    """pred: (n,5) xyxy+score, gt: (m,4). Kembalikan scores (n,), tp (n, len(IOUS)), n_gt."""
    if len(pred) == 0:
        return np.zeros(0), np.zeros((0, len(IOUS)), bool), len(gt)
    order = np.argsort(-pred[:, 4])
    pred = pred[order]
    tp = np.zeros((len(pred), len(IOUS)), bool)
    if len(gt):
        iou = iou_matrix(pred[:, :4], gt)
        for k, thr in enumerate(IOUS):
            used = np.zeros(len(gt), bool)
            for i in range(len(pred)):
                cand = np.where(~used & (iou[i] >= thr))[0]
                if len(cand):
                    j = cand[np.argmax(iou[i, cand])]
                    used[j] = True
                    tp[i, k] = True
    return pred[:, 4], tp, len(gt)


def ap_from(scores, tp, n_gt):
    if n_gt == 0:
        return np.full(len(IOUS), np.nan)
    o = np.argsort(-scores, kind="mergesort")
    tp = tp[o]
    ctp = np.cumsum(tp, 0); cfp = np.cumsum(~tp, 0)
    rec = ctp / n_gt
    prec = ctp / np.maximum(ctp + cfp, 1e-9)
    out = []
    rs = np.linspace(0, 1, 101)
    for k in range(len(IOUS)):
        p = prec[:, k].copy(); r = rec[:, k]
        p = np.maximum.accumulate(p[::-1])[::-1] if len(p) else p
        idx = np.searchsorted(r, rs, side="left")
        q = np.array([p[i] if i < len(p) else 0.0 for i in idx])
        out.append(q.mean())
    return np.array(out)


def load_gt(uid, w, h):
    lab = D / "harmonized" / "labels" / f"{uid}.txt"
    b = []
    for ln in lab.read_text().splitlines():
        v = ln.split()
        if len(v) == 5:
            cx, cy, bw, bh = (float(x) for x in v[1:])
            b.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return np.array(b).reshape(-1, 4)


# ------------------------------------------------------------------ A. eksperimen alami
def natural(args):
    meta = {r["uid"]: r for r in csv.DictReader(open(D / "audit" / "meta_split.csv"))}
    test = [u for u, r in meta.items() if r["dataset"] == "coxpwd" and r["split"] == "test"]
    group = {u: ("whatsapp" if meta[u]["camera"] == "whatsapp" else "kamera_asli") for u in test}
    gts = {u: load_gt(u, int(meta[u]["w"]), int(meta[u]["h"])) for u in test}
    out = []

    def say(s=""):
        print(s)
        out.append(s)

    say("# Eksperimen alami CoxPWD: foto lewat WhatsApp vs foto kamera asli (test split baru)\n")
    say(f"citra uji: WhatsApp {sum(g == 'whatsapp' for g in group.values())}, "
        f"kamera asli {sum(g == 'kamera_asli' for g in group.values())}\n")
    say("| Model latih | AP50 WA | AP50 asli | selisih (WA-asli) [95% CI] | AP50-95 WA | AP50-95 asli | selisih [95% CI] |")
    say("|---|---|---|---|---|---|---|")
    rng = np.random.default_rng(0)
    for src in DS:
        per_seed = []
        for seed in args.seeds:
            pj = W / "evals" / f"yolo11m_{src}_s{seed}" / "coxpwd_1024" / "predictions.json"
            if not pj.exists():
                continue
            preds = defaultdict(list)
            for p in json.load(open(pj)):
                x, y, w, h = p["bbox"]
                preds[str(p["image_id"])].append([x, y, x + w, y + h, p["score"]])
            per_seed.append({u: match_image(np.array(preds.get(u, [])).reshape(-1, 5), gts[u]) for u in test})
        if not per_seed:
            continue
        sub = {g: [u for u in test if group[u] == g] for g in ("whatsapp", "kamera_asli")}

        def ap_mean(idx_by_group):
            res = {}
            for g, ids in idx_by_group.items():
                aps = []
                for m in per_seed:
                    s = np.concatenate([m[u][0] for u in ids]); t = np.concatenate([m[u][1] for u in ids])
                    aps.append(ap_from(s, t, sum(m[u][2] for u in ids)))
                res[g] = np.nanmean(aps, axis=0)
            return res

        base = ap_mean(sub)
        diffs50, diffs = [], []
        for _ in range(args.n_boot):
            bs = {g: list(rng.choice(ids, len(ids))) for g, ids in sub.items()}
            r = ap_mean(bs)
            diffs50.append(r["whatsapp"][0] - r["kamera_asli"][0])
            diffs.append(r["whatsapp"].mean() - r["kamera_asli"].mean())
        d50 = base["whatsapp"][0] - base["kamera_asli"][0]
        d = base["whatsapp"].mean() - base["kamera_asli"].mean()
        lo50, hi50 = np.percentile(diffs50, [2.5, 97.5]); lo, hi = np.percentile(diffs, [2.5, 97.5])
        say(f"| {SHORT[src]}{' (in-domain)' if src == 'coxpwd' else ''} | {base['whatsapp'][0]:.3f} | {base['kamera_asli'][0]:.3f} | "
            f"{d50:+.3f} [{lo50:+.3f}, {hi50:+.3f}] | {base['whatsapp'].mean():.3f} | {base['kamera_asli'].mean():.3f} | "
            f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}] |")
    say(f"\n(seed: {args.seeds}; bootstrap {args.n_boot}x per kelompok; AP dihitung evaluator sendiri, 101 titik)")
    (W / "results" / "compression_natural.md").write_text("\n".join(out) + "\n")


# ------------------------------------------------------------------ B. simulasi rantai kirim
def degrade(src_path, steps):
    from PIL import Image
    im = Image.open(src_path).convert("RGB")
    data = None
    for long_side, q in steps:
        if max(im.size) > long_side:
            s = long_side / max(im.size)
            im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=q)
        data = buf.getvalue()
        im = Image.open(io.BytesIO(data)).convert("RGB")
    return data


def build_chain_set(ds, chain):
    base = W / "data_comp" / f"{ds}_{chain}"
    di, dl = base / "test" / "images", base / "test" / "labels"
    src_imgs = sorted((W / "data" / ds / "test" / "images").iterdir())
    if not (di.exists() and len(list(di.iterdir())) == len(src_imgs)):
        shutil.rmtree(base, ignore_errors=True)
        di.mkdir(parents=True); dl.mkdir(parents=True)
        for p in src_imgs:
            if CHAINS[chain]:
                (di / p.name).write_bytes(degrade(p.resolve(), CHAINS[chain]))
            else:
                (di / p.name).symlink_to(p.resolve())
            (dl / (p.stem + ".txt")).symlink_to((W / "data" / ds / "test" / "labels" / (p.stem + ".txt")).resolve())
    y = base / "data.yaml"
    y.write_text(f"path: {base}\ntrain: test/images\nval: test/images\ntest: test/images\nnames:\n  0: litter\n")
    return y


def simulate(args):
    from ultralytics import YOLO
    out = W / "results" / "compression_sim.csv"
    done = set()
    if out.exists():
        done = {(r["run"], r["test_ds"], r["chain"]) for r in csv.DictReader(open(out))}
    yamls = {}
    for t in args.targets:
        for c in CHAINS:
            yamls[(t, c)] = build_chain_set(t, c)
            print(f"set {t}/{c} siap", flush=True)
    new = not out.exists()
    with open(out, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["run", "train_ds", "seed", "test_ds", "chain", "map50", "map5095"])
        for seed in args.seeds:
            for s in DS:
                run = f"yolo11m_{s}_s{seed}"
                wts = W / "runs" / run / "weights" / "best.pt"
                if not wts.exists():
                    continue
                m = YOLO(str(wts))
                for t in args.targets:
                    for c in CHAINS:
                        if (run, t, c) in done:
                            continue
                        r = m.val(data=str(yamls[(t, c)]), split="test", imgsz=1024, batch=16, device=args.device,
                                  conf=0.001, plots=False, verbose=False, project=str(W / "tmp_comp"),
                                  name="v", exist_ok=True, workers=8)
                        w.writerow([run, s, seed, t, c, f"{r.box.map50:.4f}", f"{r.box.map:.4f}"])
                        fh.flush()
                        print(f"[{run}] {t:10s} {c:4s} mAP50 {r.box.map50:.3f}", flush=True)
    shutil.rmtree(W / "tmp_comp", ignore_errors=True)
    summarize()


def summarize():
    rows = list(csv.DictReader(open(W / "results" / "compression_sim.csv")))
    agg = defaultdict(list)
    for r in rows:
        agg[(r["train_ds"], r["test_ds"], r["chain"])].append(float(r["map50"]))
    lines = ["# Simulasi rantai kirim - mAP50 (rata-rata seed) dan perubahan relatif terhadap L0\n",
             "| Latih -> Uji | " + " | ".join(CHAINS) + " |", "|---" * (len(CHAINS) + 1) + "|"]
    for (s, t) in sorted({(k[0], k[1]) for k in agg}):
        l0 = np.mean(agg.get((s, t, "L0"), [np.nan]))
        cells = []
        for c in CHAINS:
            v = np.mean(agg.get((s, t, c), [np.nan]))
            cells.append(f"{v:.3f}" if c == "L0" else f"{v:.3f} ({(v - l0) / l0:+.1%})" if l0 > 0 else f"{v:.3f}")
        lines.append(f"| {SHORT[s]} -> {SHORT[t]}{' *' if s == t else ''} | " + " | ".join(cells) + " |")
    lines.append("\n(* = in-domain)")
    (W / "results" / "compression_sim.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("natural")
    a.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    a.add_argument("--n-boot", type=int, default=1000)
    b = sub.add_parser("simulate")
    b.add_argument("--device", default="0")
    b.add_argument("--targets", nargs="+", choices=DS, default=["taco", "plastopol", "coxpwd", "bepli"])
    b.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = ap.parse_args()
    {"natural": natural, "simulate": simulate}[args.cmd](args)


if __name__ == "__main__":
    main()
