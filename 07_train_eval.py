#!/usr/bin/env python3
"""
07_train_eval.py - Baseline lintas dataset: latih di 1 dataset, uji di test set keenam dataset.

Subperintah:
  prep   : buat folder per (dataset, split) berisi symlink ke harmonized/ + data yaml,
           lalu bangun cache label Ultralytics satu per satu (supaya proses paralel tidak rebutan menulis cache)
  train  : latih model untuk dataset & seed yang dipilih, lalu evaluasi best.pt di 6 test set
  eval   : evaluasi ulang run yang sudah ada (mis. dengan imgsz lain)

Hasil:
  ~/litter-xdomain/work/runs/<model>_<ds>_s<seed>/        (output training Ultralytics)
  ~/litter-xdomain/work/evals/<run>/<test_ds>_<imgsz>/    (predictions.json untuk analisis lanjut)
  ~/litter-xdomain/work/results/results.csv               (satu baris per run x test set)

Contoh:
  python 07_train_eval.py prep --device 7
  python 07_train_eval.py train --ds taco --seeds 0 --epochs 1 --device 7          # uji singkat
  python 07_train_eval.py train --ds taco plastopol --seeds 0 1 2 --device 7
  python 07_train_eval.py train --lodo --ds rolid --seeds 0 --device 7               # latih 5 lain, tahan RoLID
  python 07_train_eval.py train --ds coxpwd --scale 0.9 --tag sc09 --device 7         # augmentasi skala lebar
  python 07_train_eval.py train --ds taco --seeds 0 --model rtdetr-l --device 5        # RT-DETR (arsitektur kedua)
Catatan: run dengan --tag dicatat di results.csv sebagai model "<model>_<tag>" (tidak tercampur baseline).
"""
import argparse
import csv
import fcntl
import os
import shutil
import time
from pathlib import Path

D = Path(os.path.expanduser("~/data/litter"))
H = D / "harmonized"
S = D / "splits"
W = Path(os.path.expanduser("~/litter-xdomain/work"))
DATASETS = ["taco", "plastopol", "coxpwd", "bepli", "plitterstreet", "rolid"]
SPLITS = ["train", "val", "test"]


def yaml_path(ds):
    return W / "data" / f"{ds}.yaml"


def prep(args):
    from ultralytics import YOLO
    for ds in DATASETS:
        base = W / "data" / ds
        for sp in SPLITS:
            lst = S / f"{ds}_{sp}.txt"
            uids = [Path(p).stem for p in lst.read_text().split()]
            di, dl = base / sp / "images", base / sp / "labels"
            if di.exists() and len(list(di.iterdir())) == len(uids):
                continue
            shutil.rmtree(base / sp, ignore_errors=True)
            di.mkdir(parents=True)
            dl.mkdir(parents=True)
            for u in uids:
                (di / f"{u}.jpg").symlink_to(H / "images" / f"{u}.jpg")
                (dl / f"{u}.txt").symlink_to(H / "labels" / f"{u}.txt")
            print(f"{ds:14s} {sp:5s}: {len(uids)} citra", flush=True)
        yaml_path(ds).write_text(
            f"path: {base}\ntrain: train/images\nval: val/images\ntest: test/images\nnames:\n  0: litter\n")
    # bangun cache label secara berurutan (val kecil-kecilan dengan model nano, resolusi 64)
    m = YOLO("yolo11n.pt")
    for ds in DATASETS:
        for sp in SPLITS:
            if (W / "data" / ds / sp / "labels.cache").exists():
                continue
            m.val(data=str(yaml_path(ds)), split=sp, imgsz=64, batch=64, device=args.device,
                  plots=False, verbose=False, project=str(W / "tmp"), name="cache", exist_ok=True, workers=8)
            print(f"cache {ds}/{sp} OK", flush=True)
    shutil.rmtree(W / "tmp", ignore_errors=True)
    print("prep selesai")


def lodo_yaml(target):
    src = [d for d in DATASETS if d != target]
    p = W / "data" / f"lodo_{target}.yaml"
    p.write_text(f"path: {W / 'data'}\n"
                 f"train: [{', '.join(f'{d}/train/images' for d in src)}]\n"
                 f"val: [{', '.join(f'{d}/val/images' for d in src)}]\n"
                 f"test: {target}/test/images\nnames:\n  0: litter\n")
    return p


def load_model(path):
    """YOLO untuk keluarga YOLO, RTDETR untuk RT-DETR (kelas berbeda di Ultralytics)."""
    from ultralytics import RTDETR, YOLO
    return RTDETR(str(path)) if "rtdetr" in Path(str(path)).as_posix().lower() else YOLO(str(path))


def append_result(row):
    out = W / "results" / "results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a+", newline="") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        fh.seek(0)
        new = fh.read() == ""
        w = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)
        fcntl.flock(fh, fcntl.LOCK_UN)


def evaluate(run_name, weights, train_ds, seed, model_name, args):
    m = load_model(weights)
    for tds in args.test_ds:
        t0 = time.time()
        r = m.val(data=str(yaml_path(tds)), split="test", imgsz=args.eval_imgsz, batch=16,
                  device=args.device, conf=0.001, save_json=True, plots=False, verbose=False,
                  project=str(W / "evals" / run_name), name=f"{tds}_{args.eval_imgsz}", exist_ok=True,
                  workers=8)
        row = dict(model=model_name, train_ds=train_ds, seed=seed, test_ds=tds,
                   imgsz_train=args.imgsz, imgsz_eval=args.eval_imgsz,
                   map50=round(float(r.box.map50), 4), map5095=round(float(r.box.map), 4),
                   precision=round(float(r.box.mp), 4), recall=round(float(r.box.mr), 4),
                   in_domain=int(tds == train_ds), sec=round(time.time() - t0, 1), run=run_name)
        append_result(row)
        print(f"  [{run_name}] -> {tds:14s} mAP50 {row['map50']:.3f}  mAP50-95 {row['map5095']:.3f}", flush=True)


def train(args):
    for seed in args.seeds:          # seed di luar: semua dataset seed 0 selesai dulu
        for ds in args.ds:
            key = f"lodo-{ds}" if args.lodo else ds          # --lodo: ds = target yang DITAHAN
            data = lodo_yaml(ds) if args.lodo else yaml_path(ds)
            name = f"{args.model}_{key}_s{seed}" + (f"_{args.tag}" if args.tag else "")
            run_dir = W / "runs" / name
            best = run_dir / "weights" / "best.pt"
            if best.exists() and not args.retrain:
                print(f"[{name}] sudah dilatih, langsung evaluasi", flush=True)
            else:
                t0 = time.time()
                extra = {} if args.scale is None else {"scale": args.scale}
                load_model(f"{args.model}.pt").train(
                    data=str(data), epochs=args.epochs, patience=args.patience, imgsz=args.imgsz,
                    batch=args.batch, device=args.device, seed=seed, deterministic=True, workers=args.workers,
                    cache=(args.cache if args.cache != "none" else False), **extra,
                    project=str(W / "runs"), name=name, exist_ok=True, plots=False, verbose=False)
                print(f"[{name}] training selesai dalam {(time.time() - t0) / 60:.1f} menit", flush=True)
            evaluate(name, best, key, seed, args.model + (f"_{args.tag}" if args.tag else ""), args)


def eval_only(args):
    for ds in args.ds:
        for seed in args.seeds:
            key = f"lodo-{ds}" if args.lodo else ds
            name = f"{args.model}_{key}_s{seed}" + (f"_{args.tag}" if args.tag else "")
            evaluate(name, W / "runs" / name / "weights" / "best.pt", key, seed,
                     args.model + (f"_{args.tag}" if args.tag else ""), args)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--device", default="0")
    for name in ("train", "eval"):
        t = sub.add_parser(name)
        t.add_argument("--ds", nargs="+", choices=DATASETS, required=True)
        t.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
        t.add_argument("--model", default="yolo11m")
        t.add_argument("--epochs", type=int, default=100)
        t.add_argument("--patience", type=int, default=30)
        t.add_argument("--imgsz", type=int, default=1024)
        t.add_argument("--eval-imgsz", type=int, default=None)
        t.add_argument("--batch", type=int, default=16)
        t.add_argument("--device", default="0")
        t.add_argument("--test-ds", nargs="+", choices=DATASETS, default=DATASETS)
        t.add_argument("--tag", default="")
        t.add_argument("--retrain", action="store_true")
        t.add_argument("--workers", type=int, default=8)
        t.add_argument("--lodo", action="store_true",
                       help="leave-one-dataset-out: latih di 5 dataset lain, --ds = dataset yang ditahan")
        t.add_argument("--scale", type=float, default=None,
                       help="augmentasi skala Ultralytics (default bawaan 0.5); mis. 0.9 = jitter skala lebar")
        t.add_argument("--cache", default="none", choices=["none", "ram", "disk"],
                       help="ram: citra yang sudah diperkecil disimpan di RAM (jauh lebih cepat bila RAM cukup)")
    args = ap.parse_args()
    if getattr(args, "eval_imgsz", None) is None and args.cmd != "prep":
        args.eval_imgsz = args.imgsz
    {"prep": prep, "train": train, "eval": eval_only}[args.cmd](args)


if __name__ == "__main__":
    main()
