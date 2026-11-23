#!/usr/bin/env python3
"""
11_zeroshot.py - Detektor open-vocabulary TANPA training di 6 test set (split baru).

Model:
  yoloworld : YOLO-World v2-X (Ultralytics, yolov8x-worldv2.pt)
  yoloe     : YOLOE-11L (Ultralytics, yoloe-11l-seg.pt, prompt teks)
  gdino     : Grounding DINO base (Hugging Face transformers, IDEA-Research/grounding-dino-base)
  gdino_t2  : Grounding DINO + tiling grid 2x2 (tumpang tindih 20%) + citra utuh, digabung NMS
  gdino_t3  : sama, grid 3x3   -> menguji apakah objek kecil tertolong bila tampak lebih besar
Prompt (semua hasil digabung menjadi satu kelas "litter"):
  p1  : litter
  p2  : litter, trash, garbage, plastic waste
  p3  : daftar objek umum: plastic bottle, plastic bag, plastic wrapper, can, cup, bottle cap,
        cigarette butt, straw, food container, paper, styrofoam, face mask
CATATAN: prompt terbaik dipilih DI TEST SET = batas atas optimistis; laporkan ketiganya.
AP dihitung dengan evaluator sendiri (gaya COCO 101 titik, sudah dicocokkan dengan pycocotools).

Pemakaian:
  python 11_zeroshot.py --models yoloworld yoloe --device 7
  python 11_zeroshot.py --models gdino --device 7
  python 11_zeroshot.py --summary
Output: work/results/zeroshot.csv dan zeroshot.md ; prediksi mentah di work/zeroshot/<model>_<prompt>_<ds>.json
"""
import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

D = Path(os.path.expanduser("~/data/litter"))
W = Path(os.path.expanduser("~/litter-xdomain/work"))
DS = ["taco", "plastopol", "coxpwd", "bepli", "plitterstreet", "rolid"]
SHORT = {"taco": "TACO", "plastopol": "PlastOPol", "coxpwd": "CoxPWD", "bepli": "BePLi",
         "plitterstreet": "pLitter", "rolid": "RoLID"}
PROMPTS = {
    "p1": ["litter"],
    "p2": ["litter", "trash", "garbage", "plastic waste"],
    "p3": ["plastic bottle", "plastic bag", "plastic wrapper", "can", "cup", "bottle cap", "cigarette butt",
           "straw", "food container", "paper", "styrofoam", "face mask"],
}
IOUS = np.arange(0.5, 0.96, 0.05)


# ------------------------------------------------------------------ evaluator (sama dengan 10_compression.py)
def iou_matrix(a, b):
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0]); iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2]); iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-9)


def match_image(pred, gt):
    if len(pred) == 0:
        return np.zeros(0), np.zeros((0, len(IOUS)), bool), len(gt)
    pred = pred[np.argsort(-pred[:, 4])]
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
    rec = ctp / n_gt; prec = ctp / np.maximum(ctp + cfp, 1e-9)
    rs = np.linspace(0, 1, 101); out = []
    for k in range(len(IOUS)):
        p = prec[:, k].copy(); r = rec[:, k]
        p = np.maximum.accumulate(p[::-1])[::-1] if len(p) else p
        idx = np.searchsorted(r, rs, side="left")
        out.append(np.array([p[i] if i < len(p) else 0.0 for i in idx]).mean())
    return np.array(out)


def nms_merge(boxes, iou_thr=0.6, max_det=300):
    """Gabungkan prediksi dari banyak prompt menjadi satu kelas (NMS agnostik kelas)."""
    if len(boxes) == 0:
        return boxes
    boxes = boxes[np.argsort(-boxes[:, 4])]
    keep = []
    alive = np.ones(len(boxes), bool)
    for i in range(len(boxes)):
        if not alive[i]:
            continue
        keep.append(i)
        if len(keep) >= max_det:
            break
        iou = iou_matrix(boxes[i:i + 1, :4], boxes[:, :4])[0]
        alive &= iou < iou_thr
        alive[i] = False
    return boxes[keep]


def test_set(ds):
    meta = {r["uid"]: r for r in csv.DictReader(open(D / "audit" / "meta_split.csv"))}
    out = []
    for p in (D / "splits" / f"{ds}_test.txt").read_text().split():
        u = Path(p).stem
        w, h = int(meta[u]["w"]), int(meta[u]["h"])
        gt = []
        for ln in (D / "harmonized" / "labels" / f"{u}.txt").read_text().splitlines():
            v = ln.split()
            if len(v) == 5:
                cx, cy, bw, bh = (float(x) for x in v[1:])
                gt.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
        out.append((u, p, np.array(gt).reshape(-1, 4)))
    return out


# ------------------------------------------------------------------ prediktor
def predict_ultra(kind, names, items, device, imgsz):
    from ultralytics import YOLO, YOLOE
    if kind == "yoloworld":
        m = YOLO("yolov8x-worldv2.pt")
        m.set_classes(names)
    else:
        m = YOLOE("yoloe-11l-seg.pt")
        m.set_classes(names, m.get_text_pe(names))
    preds = {}
    paths = [p for _, p, _ in items]
    uids = [u for u, _, _ in items]
    for k in range(0, len(paths), 8):
        res = m.predict(paths[k:k + 8], imgsz=imgsz, conf=0.001, max_det=300, device=device,
                        verbose=False, agnostic_nms=True)
        for u, r in zip(uids[k:k + 8], res):
            b = r.boxes
            arr = np.concatenate([b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None]], 1) if len(b) else np.zeros((0, 5))
            preds[u] = nms_merge(arr)
    return preds


def tile_boxes(W, H, g, ov=0.2):
    """Grid g x g dengan tumpang tindih ov; kembalikan daftar (x0, y0, x1, y1)."""
    tw = int(np.ceil(W / (g - (g - 1) * ov)))
    th = int(np.ceil(H / (g - (g - 1) * ov)))
    sx = (W - tw) / max(g - 1, 1)
    sy = (H - th) / max(g - 1, 1)
    return [(int(round(i * sx)), int(round(j * sy)), int(round(i * sx)) + tw, int(round(j * sy)) + th)
            for j in range(g) for i in range(g)]


def predict_gdino(names, items, device, box_thr=0.05, grid=0, ov=0.2):
    """grid=0: citra utuh saja. grid=g: citra utuh + g x g potongan (tiling), digabung dengan NMS."""
    import torch
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    mid = "IDEA-Research/grounding-dino-base"
    proc = AutoProcessor.from_pretrained(mid)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(mid).to(f"cuda:{device}").eval()
    text = " . ".join(names) + " ."

    def run(crops):
        inp = proc(images=crops, text=[text] * len(crops), return_tensors="pt", padding=True).to(f"cuda:{device}")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(**inp)
        out.logits = out.logits.float(); out.pred_boxes = out.pred_boxes.float()
        sizes = [c.size[::-1] for c in crops]
        try:
            rs = proc.post_process_grounded_object_detection(out, inp.input_ids, threshold=box_thr,
                                                             text_threshold=box_thr, target_sizes=sizes)
        except TypeError:
            rs = proc.post_process_grounded_object_detection(out, inp.input_ids, box_threshold=box_thr,
                                                             text_threshold=box_thr, target_sizes=sizes)
        return [(r["boxes"].cpu().numpy(), r["scores"].cpu().numpy()) for r in rs]

    preds = {}
    for n, (u, p, _) in enumerate(items):
        im = Image.open(p).convert("RGB")
        allb = []
        b, s = run([im])[0]
        if len(b):
            allb.append(np.concatenate([b, s[:, None]], 1))
        if grid:
            tb = tile_boxes(im.width, im.height, grid, ov)
            for (x0, y0, _, _), (b, s) in zip(tb, run([im.crop(t) for t in tb])):
                if len(b):
                    b = b + np.array([x0, y0, x0, y0], dtype=b.dtype)
                    allb.append(np.concatenate([b, s[:, None]], 1))
        arr = np.concatenate(allb) if allb else np.zeros((0, 5))
        preds[u] = nms_merge(arr, iou_thr=0.5 if grid else 0.6)
        if n % 500 == 0:
            print(f"    gdino grid={grid} {n}/{len(items)}", flush=True)
    return preds


def score(preds, items):
    m = [match_image(preds.get(u, np.zeros((0, 5))), gt) for u, _, gt in items]
    s = np.concatenate([x[0] for x in m]); t = np.concatenate([x[1] for x in m])
    return ap_from(s, t, sum(x[2] for x in m))


def run(args):
    out = W / "results" / "zeroshot.csv"
    (W / "zeroshot").mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {(r["model"], r["prompt"], r["test_ds"]) for r in csv.DictReader(open(out))}
    sets = {d: test_set(d) for d in args.ds}
    new = not out.exists()
    with open(out, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["model", "prompt", "test_ds", "imgsz", "map50", "map5095", "n_img"])
        for model in args.models:
            for pk in args.prompts:
                todo = [d for d in args.ds if (model, pk, d) not in done]
                if not todo:
                    continue
                items = [it for d in todo for it in sets[d]]
                print(f"[{model} {pk}] {len(items)} citra ...", flush=True)
                if model.startswith("gdino"):
                    grid = int(model.split("_t")[1]) if "_t" in model else 0
                    preds = predict_gdino(PROMPTS[pk], items, args.device, grid=grid)
                else:
                    preds = predict_ultra(model, PROMPTS[pk], items, args.device, args.imgsz)
                for d in todo:
                    ap = score(preds, sets[d])
                    w.writerow([model, pk, d, args.imgsz if not model.startswith("gdino") else "native",
                                f"{ap[0]:.4f}", f"{ap.mean():.4f}", len(sets[d])])
                    fh.flush()
                    json.dump({u: preds[u].round(2).tolist() for u, _, _ in sets[d]},
                              open(W / "zeroshot" / f"{model}_{pk}_{d}.json", "w"))
                    print(f"  [{model} {pk}] {d:14s} AP50 {ap[0]:.3f}  AP50-95 {ap.mean():.3f}", flush=True)
    summary()


def summary():
    zs = list(csv.DictReader(open(W / "results" / "zeroshot.csv")))
    sup = defaultdict(list)
    for r in csv.DictReader(open(W / "results" / "results.csv")):
        if r["model"] == "yolo11m" and r["imgsz_eval"] == "1024" and r["train_ds"] in DS:
            sup[(r["train_ds"], r["test_ds"])].append(float(r["map50"]))
    lines = ["# Zero-shot vs supervised (mAP50, test split baru)\n",
             "| Model / prompt | " + " | ".join(SHORT[d] for d in DS) + " |", "|---" * (len(DS) + 1) + "|"]
    for model in ("yoloworld", "yoloe", "gdino", "gdino_t2", "gdino_t3"):
        for pk in PROMPTS:
            row = {r["test_ds"]: float(r["map50"]) for r in zs if r["model"] == model and r["prompt"] == pk}
            if row:
                lines.append(f"| {model} {pk} | " + " | ".join(f"{row[d]:.3f}" if d in row else "-" for d in DS) + " |")
    best = []
    for d in DS:
        c = [(np.mean(v), s) for (s, t), v in sup.items() if t == d and s != d]
        best.append(f"{max(c)[0]:.3f} ({SHORT[max(c)[1]]})" if c else "-")
    lines.append("| supervised lintas dataset terbaik (YOLO11m) | " + " | ".join(best) + " |")
    lines.append("| supervised in-domain (YOLO11m) | " + " | ".join(
        f"{np.mean(sup[(d, d)]):.3f}" if (d, d) in sup else "-" for d in DS) + " |")
    (W / "results" / "zeroshot.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["yoloworld", "yoloe", "gdino"], choices=["yoloworld", "yoloe", "gdino", "gdino_t2", "gdino_t3"])
    ap.add_argument("--prompts", nargs="+", default=list(PROMPTS), choices=list(PROMPTS))
    ap.add_argument("--ds", nargs="+", default=DS, choices=DS)
    ap.add_argument("--device", default="0")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    if args.summary:
        summary()
    else:
        run(args)


if __name__ == "__main__":
    main()
