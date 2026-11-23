#!/usr/bin/env python3
"""
04_harmonize.py - Ubah 6 dataset ke format YOLO satu kelas "litter" + metadata per citra.

Aturan (hasil 02_inventory & 03_probe):
  * Semua kategori sampah -> kelas 0 "litter".
  * pLitterStreet: 'Trash bin' bukan sampah -> box dibuang (citra tetap dipakai);
    citra yang memuat 'Pile' (tumpukan, granularitas berbeda) -> dikeluarkan.
  * Orientasi: pilih orientasi citra yang ukurannya cocok dengan width/height di JSON
    (TACO: ~1/3 citra butuh rotasi EXIF, ada juga yang JSON-nya mengikuti ukuran mentah).
  * Citra diperkecil hanya bila sisi panjang > --max-side (default 2048), lalu JPEG q95.
    Citra JPEG yang tidak perlu diputar/diperkecil disalin APA ADANYA (byte asli),
    supaya kompresi asli (mis. foto WhatsApp di CoxPWD) tidak berubah.
  * Box di-clip ke batas citra; box < 2 px setelah skala dibuang (dicatat).
  * Grup (untuk split anti-bocor; nanti digabung lagi dengan hasil audit duplikat):
      pLitterStreet: lokasi_tahun + nomor video GoPro (chapter GX01/GX02 = video sama)
      RoLID: ID rekaman (tanpa akhiran _Trim/_Trim_1/_Trim_2)
      lainnya: per citra.
Output:
  ~/data/litter/harmonized/images/<uid>.jpg
  ~/data/litter/harmonized/labels/<uid>.txt
  ~/data/litter/harmonized/meta.csv
Pemakaian:
  python 04_harmonize.py --workers 16            # semua
  python 04_harmonize.py --only taco --limit 50  # uji cepat
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = None
D = Path(os.path.expanduser("~/data/litter"))
OUT = D / "harmonized"

SRC = {
    "taco": dict(jsons=[("taco/extracted/TACO/TACO/data/annotations.json", "all")],
                 root="taco/extracted/TACO/TACO/data"),
    "plastopol": dict(jsons=[("plastopol/extracted/PlastOpol_oneclass/plastopol-annotations_folds/annotations/plastopol_train_val.json", "trainval"),
                             ("plastopol/extracted/PlastOpol_oneclass/plastopol-annotations_folds/annotations/plastopol_test.json", "test")],
                      root="plastopol/extracted"),
    "coxpwd": dict(jsons=[(p, Path(p).parent.name) for p in
                          sorted(glob.glob(str(D / "coxpwd/extracted/*/*/_annotations.coco.json")))],
                   root="coxpwd/extracted"),
    "bepli": dict(jsons=[(f"bepli/extracted/98753/plastic_coco/annotation/{s}.json", s) for s in ("train", "val", "test")],
                  root="bepli/extracted/98753/plastic_coco/images/original_images"),
    "plitterstreet": dict(jsons=[(f"plitterstreet/extracted/annotations/annotations/4_class_splits/annotations_{s}.json", s)
                                 for s in ("train", "val", "test")],
                          root="plitterstreet/extracted/images"),
    "rolid": dict(jsons=[(f"rolid/raw/{s}.json", s) for s in ("training", "validation", "testing")],
                  root="rolid/extracted"),
}
DROP_BOX = {"plitterstreet": {"Trash bin"}}
DROP_IMAGE = {"plitterstreet": {"Pile"}}
REGION = {"taco": "Europe", "plastopol": "Global", "coxpwd": "South Asia", "bepli": "East Asia",
          "plitterstreet": "SE/South Asia", "rolid": "Europe"}
VIEW = {"taco": "handheld", "plastopol": "handheld", "coxpwd": "handheld", "bepli": "handheld",
        "plitterstreet": "vehicle", "rolid": "vehicle"}


def group_key(ds, fname):
    b = os.path.basename(fname)
    if ds == "plitterstreet":
        m = re.match(r"^(.+?)_G[XHP]\d{2}(\d{4})_\d+\.\w+$", b)
        return f"{m.group(1)}_vid{m.group(2)}" if m else "single:" + b
    if ds == "rolid":
        m = re.match(r"^(\d{14}_\d{6})", b)
        return m.group(1) if m else "single:" + b
    return "single:" + fname  # path lengkap (nama file TACO berulang antar batch)


def camera_source(ds, fname):
    if ds != "coxpwd":
        return ""
    b = os.path.basename(fname)
    if b.startswith("WhatsApp"):
        return "whatsapp"
    if b.startswith("PXL_"):
        return "pixel"
    if b.upper().startswith("IMG"):
        return "img"
    return "other"


def build_jobs(ds, limit=None):
    cfg = SRC[ds]
    root = D / cfg["root"]
    by_name = defaultdict(list)
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            by_name[p.name].append(p)
    jobs, seen, stats = [], set(), Counter()
    for jpath, split in cfg["jsons"]:
        jp = Path(jpath) if os.path.isabs(jpath) else D / jpath
        d = json.load(open(jp))
        cats = {c["id"]: c["name"] for c in d["categories"]}
        anns = defaultdict(list)
        for a in d["annotations"]:
            anns[a["image_id"]].append(a)
        for im in d["images"]:
            fn = im["file_name"].replace("\\", "/")
            src = root / fn
            if not src.exists():
                c = by_name.get(os.path.basename(fn), [])
                if len(c) != 1:
                    stats["tidak_ditemukan_atau_ambigu"] += 1
                    continue
                src = c[0]
            key = str(src.resolve())
            if key in seen:
                stats["duplikat_entri_json"] += 1
                continue
            seen.add(key)
            names = {cats.get(a["category_id"]) for a in anns[im["id"]]}
            if names & DROP_IMAGE.get(ds, set()):
                stats["dikeluarkan_karena_pile"] += 1
                continue
            boxes, dropped = [], 0
            for a in anns[im["id"]]:
                if cats.get(a["category_id"]) in DROP_BOX.get(ds, set()):
                    dropped += 1
                    continue
                if a.get("bbox"):
                    boxes.append(a["bbox"])
            stats["box_trash_bin_dibuang"] += dropped
            rel = os.path.relpath(src, D)
            uid = f"{ds}_{hashlib.sha1(rel.encode()).hexdigest()[:12]}"
            jobs.append(dict(ds=ds, uid=uid, src=str(src), rel=rel, split=split,
                             jw=im.get("width"), jh=im.get("height"), boxes=boxes,
                             group=f"{ds}:{group_key(ds, fn)}", cam=camera_source(ds, fn)))
            if limit and len(jobs) >= limit:
                return jobs, stats
    return jobs, stats


def process(job, max_side=2048):
    src = Path(job["src"])
    flags = []
    try:
        im = Image.open(src)
        raw_size = im.size
        ori = im.getexif().get(0x0112, 1)
        tr = ImageOps.exif_transpose(im) if ori not in (1, None) else im
        jw, jh = job["jw"], job["jh"]
        if jw and jh and tr.size == (jw, jh):
            use, rotated = tr, ori not in (1, None)
        elif jw and jh and raw_size == (jw, jh):
            use, rotated = im, False
            if ori not in (1, None):
                flags.append("exif_diabaikan")
        else:
            use, rotated = tr, ori not in (1, None)
            flags.append(f"ukuran_beda_json{jw}x{jh}_disk{tr.size[0]}x{tr.size[1]}")
        W, H = use.size
        rw, rh = (W / jw, H / jh) if (jw and jh) else (1.0, 1.0)  # skala JSON -> citra
        s = min(1.0, max_side / max(W, H))
        out_img = OUT / "images" / f"{job['uid']}.jpg"
        is_jpeg = src.suffix.lower() in (".jpg", ".jpeg") and im.format == "JPEG"
        if s == 1.0 and not rotated and is_jpeg:
            if not out_img.exists():
                shutil.copyfile(src, out_img)
            mode = "salin_asli"
        else:
            if s < 1.0:
                use = use.resize((round(W * s), round(H * s)), Image.LANCZOS)
            use.convert("RGB").save(out_img, "JPEG", quality=95)
            mode = "encode_q95"
        Wn, Hn = round(W * s), round(H * s)
        lines, tiny = [], 0
        for x, y, w, h in job["boxes"]:
            x0, y0 = max(0.0, x * rw * s), max(0.0, y * rh * s)
            x1, y1 = min(Wn, (x + w) * rw * s), min(Hn, (y + h) * rh * s)
            if x1 - x0 < 2 or y1 - y0 < 2:
                tiny += 1
                continue
            lines.append(f"0 {(x0 + x1) / 2 / Wn:.6f} {(y0 + y1) / 2 / Hn:.6f} "
                         f"{(x1 - x0) / Wn:.6f} {(y1 - y0) / Hn:.6f}")
        (OUT / "labels" / f"{job['uid']}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        return dict(uid=job["uid"], dataset=job["ds"], region=REGION[job["ds"]], view=VIEW[job["ds"]],
                    group=job["group"], official_split=job["split"], camera=job["cam"],
                    src=job["rel"], orig_w=W, orig_h=H, w=Wn, h=Hn, n_boxes=len(lines),
                    tiny_dropped=tiny, write_mode=mode, flags=";".join(flags))
    except Exception as e:  # noqa: BLE001
        return dict(uid=job["uid"], dataset=job["ds"], src=job["rel"], error=f"{type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", choices=list(SRC))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-side", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=None, help="uji cepat: N citra per dataset")
    args = ap.parse_args()
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    (OUT / "labels").mkdir(parents=True, exist_ok=True)

    meta_path = OUT / "meta.csv"
    rows = {}
    if meta_path.exists():
        for r in csv.DictReader(open(meta_path)):
            rows[r["uid"]] = r
    names = args.only or list(SRC)
    for ds in names:  # buang baris lama dataset yang akan diproses ulang
        rows = {k: v for k, v in rows.items() if v.get("dataset") != ds}

    errors = []
    for ds in names:
        jobs, st = build_jobs(ds, args.limit)
        print(f"\n=== {ds}: {len(jobs)} citra akan diproses | {dict(st)}", flush=True)
        with Pool(args.workers) as pool:
            res = pool.starmap(process, [(j, args.max_side) for j in jobs], chunksize=8)
        ok = [r for r in res if "error" not in r]
        errors += [r for r in res if "error" in r]
        for r in ok:
            rows[r["uid"]] = r
        nb = sum(r["n_boxes"] for r in ok)
        print(f"  selesai: {len(ok)} citra, {nb} box, error {len(res) - len(ok)}", flush=True)
        print(f"  cara tulis: {dict(Counter(r['write_mode'] for r in ok))}")
        print(f"  box terlalu kecil dibuang: {sum(r['tiny_dropped'] for r in ok)} | "
              f"citra tanpa box (negatif): {sum(1 for r in ok if r['n_boxes'] == 0)}")
        fl = Counter(f.split('_json')[0] for r in ok for f in r['flags'].split(';') if f)
        if fl:
            print(f"  flag: {dict(fl)}")
        g = Counter(r["group"] for r in ok)
        print(f"  grup: {len(g)} | terbesar: {g.most_common(3)}")
        if ds == "coxpwd":
            print(f"  sumber kamera: {dict(Counter(r['camera'] for r in ok))}")

    fields = ["uid", "dataset", "region", "view", "group", "official_split", "camera", "src",
              "orig_w", "orig_h", "w", "h", "n_boxes", "tiny_dropped", "write_mode", "flags"]
    with open(meta_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows.values():
            w.writerow(r)
    print(f"\nmeta.csv: {len(rows)} baris -> {meta_path}")
    if errors:
        print(f"ERROR {len(errors)} citra, contoh: {errors[:3]}")
        (OUT / "errors.json").write_text(json.dumps(errors, indent=1))


if __name__ == "__main__":
    main()
