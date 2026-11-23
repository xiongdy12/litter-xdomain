#!/usr/bin/env python3
"""
02_inventory.py - Ekstrak arsip tiap dataset lalu periksa isinya.

Tidak mengubah file asli di raw/. Hasil ekstrak masuk ke extracted/.
Yang dilaporkan per dataset:
  - struktur folder (2 tingkat teratas) dan jumlah citra
  - semua file anotasi COCO (json): jumlah citra/anotasi, kategori + jumlah box
  - label YOLO (txt) dan VOC (xml) bila ada
  - statistik ukuran citra (sampel): resolusi, rasio aspek, berapa yang tepat 640x640

Pemakaian:
    python 02_inventory.py                 # semua dataset yang sudah diunduh
    python 02_inventory.py --only taco coxpwd
    python 02_inventory.py --no-extract    # periksa saja, jangan ekstrak

Hasil ringkas juga disimpan di ~/data/litter/inventory/<nama>.json
"""
import argparse
import collections
import json
import os
import random
import shutil
import tarfile
import zipfile
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None
ROOT = Path(os.path.expanduser("~/data/litter"))
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
ARCH_EXT = (".zip", ".tar", ".tar.gz", ".tgz")
NAMES = ["taco", "plastopol", "coxpwd", "bepli", "plitterstreet", "rolid"]


def is_archive(p):
    n = p.name.lower()
    return n.endswith(ARCH_EXT) and not n.startswith(".done_") and not n.startswith("._")


def safe_extract(arch, dest):
    """Ekstrak satu arsip; lewati kalau sudah pernah (ada file penanda)."""
    mark = dest / f".done_{arch.name}"
    if mark.exists():
        return False
    dest.mkdir(parents=True, exist_ok=True)
    print(f"  [ekstrak] {arch.name} -> {dest}")
    n = arch.name.lower()
    if n.endswith(".zip"):
        with zipfile.ZipFile(arch) as z:
            for m in z.infolist():
                if ".." in Path(m.filename).parts or m.filename.startswith("/"):
                    continue
                z.extract(m, dest)
    else:
        with tarfile.open(arch) as t:
            t.extractall(dest, filter="data")
    mark.touch()
    return True


def extract_all(name):
    raw, ext = ROOT / name / "raw", ROOT / name / "extracted"
    if not raw.exists():
        print("  raw/ belum ada (belum diunduh)")
        return
    archives = [p for p in raw.rglob("*") if p.is_file() and is_archive(p)]
    if not archives:
        # misalnya RoLID dari Google Drive yang sudah berupa folder biasa
        print("  tidak ada arsip di raw/, diperiksa apa adanya")
    for a in archives:
        safe_extract(a, ext / a.stem.replace(".tar", ""))
    # arsip bersarang (zip di dalam zip), maksimal 2 tingkat
    for _ in range(2):
        nested = [p for p in ext.rglob("*") if p.is_file() and is_archive(p)
                  and not (p.parent / f".done_{p.name}").exists()]
        if not nested:
            break
        for a in nested:
            safe_extract(a, a.parent / a.stem.replace(".tar", ""))
            (a.parent / f".done_{a.name}").touch()


def tree(base, depth=2, max_items=12):
    lines = []

    def walk(d, lvl):
        if lvl > depth:
            return
        try:
            kids = sorted(d.iterdir())
        except PermissionError:
            return
        dirs = [k for k in kids if k.is_dir()]
        files = [k for k in kids if k.is_file() and not k.name.startswith(".done_")]
        for k in dirs[:max_items]:
            n_img = sum(1 for f in k.rglob("*") if f.suffix.lower() in IMG_EXT)
            lines.append(f"{'  ' * lvl}{k.name}/  ({n_img} citra)")
            walk(k, lvl + 1)
        if len(dirs) > max_items:
            lines.append(f"{'  ' * lvl}... +{len(dirs) - max_items} folder lain")
        exts = collections.Counter(f.suffix.lower() for f in files)
        if exts:
            lines.append(f"{'  ' * lvl}[file: " + ", ".join(f"{e or '(tanpa)'}x{c}" for e, c in exts.most_common(6)) + "]")

    walk(base, 0)
    return lines


def coco_summary(p, img_index):
    try:
        d = json.loads(p.read_text())
    except Exception as e:  # noqa: BLE001
        return {"file": str(p), "error": str(e)[:120]}
    if not isinstance(d, dict) or "annotations" not in d or "images" not in d:
        return None
    cats = {c["id"]: c.get("name") for c in d.get("categories", [])}
    supc = {c["id"]: c.get("supercategory") for c in d.get("categories", [])}
    cnt = collections.Counter(a.get("category_id") for a in d["annotations"])
    has_bbox = sum(1 for a in d["annotations"] if a.get("bbox"))
    has_seg = sum(1 for a in d["annotations"] if a.get("segmentation"))
    imgs_with_ann = len({a["image_id"] for a in d["annotations"]})
    missing = 0
    for im in d["images"]:
        if Path(im["file_name"]).name not in img_index:
            missing += 1
    return dict(
        file=str(p), n_images=len(d["images"]), n_annotations=len(d["annotations"]),
        images_with_ann=imgs_with_ann, images_missing_on_disk=missing,
        with_bbox=has_bbox, with_segmentation=has_seg,
        categories=[dict(id=k, name=cats.get(k), super=supc.get(k), n=cnt.get(k, 0))
                    for k in sorted(cats)],
        unknown_category_ids=sorted(set(cnt) - set(cats)),
    )


def yolo_summary(base):
    txts = [p for p in base.rglob("*.txt") if "labels" in p.parts or p.parent.name == "labels"]
    if not txts:
        return None
    cls = collections.Counter()
    empty = 0
    for t in txts:
        lines = [ln.split() for ln in t.read_text(errors="ignore").splitlines() if ln.strip()]
        if not lines:
            empty += 1
        for ln in lines:
            cls[ln[0]] += 1
    yamls = [str(p) for p in base.rglob("*.yaml")] + [str(p) for p in base.rglob("*.yml")]
    return dict(n_label_files=len(txts), empty_files=empty, class_counts=dict(cls.most_common()),
                yaml_files=yamls[:5])


def image_stats(imgs, k=400, seed=0):
    random.Random(seed).shuffle(imgs)
    sizes, bad = [], 0
    for p in imgs[:k]:
        try:
            with Image.open(p) as im:
                sizes.append(im.size)
        except Exception:  # noqa: BLE001
            bad += 1
    if not sizes:
        return {}
    res = collections.Counter(f"{w}x{h}" for w, h in sizes)
    ar = [w / h for w, h in sizes]
    return dict(
        sampled=len(sizes), unreadable=bad,
        top_resolutions=res.most_common(6),
        frac_640x640=round(sum(1 for s in sizes if s == (640, 640)) / len(sizes), 3),
        frac_square=round(sum(1 for a in ar if abs(a - 1) < 0.01) / len(sizes), 3),
        aspect_min=round(min(ar), 3), aspect_median=round(sorted(ar)[len(ar) // 2], 3),
        aspect_max=round(max(ar), 3),
    )


def inspect(name):
    # periksa extracted/ DAN file lepas di raw/ (mis. anotasi json yang tidak di-zip)
    bases = [b for b in (ROOT / name / "extracted", ROOT / name / "raw") if b.exists()]
    if not bases:
        return None

    def files(pattern):
        for b in bases:
            for p in b.rglob(pattern):
                if p.is_file() and not is_archive(p) and not p.name.startswith(".done_"):
                    yield p

    imgs = [p for p in files("*") if p.suffix.lower() in IMG_EXT]
    img_index = {p.name for p in imgs}
    dup_names = len(imgs) - len(img_index)
    cocos = []
    for j in files("*.json"):
        if j.stat().st_size > 2_000_000_000:
            continue
        s = coco_summary(j, img_index)
        if s:
            cocos.append(s)
    xmls = sum(1 for _ in files("*.xml"))
    yolo = None
    for b in bases:
        yolo = yolo or yolo_summary(b)
    tr = []
    for b in bases:
        tr.append(f"[{b.name}/]")
        tr += ["  " + ln for ln in tree(b)]
    out = dict(
        dataset=name, base=", ".join(str(b) for b in bases), n_images=len(imgs),
        duplicate_filenames=dup_names,
        image_ext=dict(collections.Counter(p.suffix.lower() for p in imgs)),
        coco=cocos, yolo=yolo, n_voc_xml=xmls,
        image_stats=image_stats(list(imgs)), tree=tr,
    )
    return out


def report(s):
    print(f"\n{'=' * 70}\n{s['dataset'].upper()}   ({s['base']})")
    print(f"citra: {s['n_images']}  | ekstensi: {s['image_ext']}  | nama file kembar: {s['duplicate_filenames']}")
    print("struktur:")
    for ln in s["tree"][:40]:
        print("   ", ln)
    for c in s["coco"]:
        if "error" in c:
            print(f"  COCO {c['file']}: ERROR {c['error']}")
            continue
        print(f"  COCO {Path(c['file']).relative_to(ROOT)}")
        print(f"     citra={c['n_images']} (tidak ada di disk: {c['images_missing_on_disk']}), "
              f"anotasi={c['n_annotations']}, citra berlabel={c['images_with_ann']}, "
              f"bbox={c['with_bbox']}, segmentasi={c['with_segmentation']}")
        cats = c["categories"]
        show = ", ".join(f"{x['name']}({x['n']})" for x in cats[:25])
        print(f"     {len(cats)} kategori: {show}{' ...' if len(cats) > 25 else ''}")
        if c["unknown_category_ids"]:
            print(f"     ID kategori tak terdaftar: {c['unknown_category_ids']}")
    if s["yolo"]:
        y = s["yolo"]
        print(f"  YOLO: {y['n_label_files']} file label (kosong {y['empty_files']}), kelas: {y['class_counts']}")
        if y["yaml_files"]:
            print(f"        yaml: {y['yaml_files']}")
    if s["n_voc_xml"]:
        print(f"  VOC xml: {s['n_voc_xml']} file")
    st = s["image_stats"]
    if st:
        print(f"  ukuran (sampel {st['sampled']}, rusak {st['unreadable']}): {st['top_resolutions']}")
        print(f"     tepat 640x640: {st['frac_640x640']:.1%} | persegi: {st['frac_square']:.1%} | "
              f"rasio aspek min/median/max: {st['aspect_min']}/{st['aspect_median']}/{st['aspect_max']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", choices=NAMES)
    ap.add_argument("--no-extract", action="store_true")
    args = ap.parse_args()
    outdir = ROOT / "inventory"
    outdir.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(ROOT).free / 1e9
    print(f"ruang kosong: {free:.1f} GB")
    for name in args.only or NAMES:
        print(f"\n--- {name}")
        if not args.no_extract:
            extract_all(name)
        s = inspect(name)
        if s is None:
            print("  (belum ada data)")
            continue
        (outdir / f"{name}.json").write_text(json.dumps(s, indent=2, default=str))
        report(s)


if __name__ == "__main__":
    main()
