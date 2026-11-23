#!/usr/bin/env python3
"""
03_probe.py - Pemeriksaan detail sebelum harmonisasi (tidak mengubah file apa pun).

Yang dicek per dataset:
  A. apakah setiap file_name di JSON bisa ditemukan di disk (tanpa ambigu)
  B. orientasi EXIF: apakah width/height di JSON cocok dengan citra mentah atau citra
     yang sudah diputar sesuai EXIF (penting: kalau salah, box akan meleset)
  C. statistik box: jumlah box per citra, ukuran relatif box (granularitas anotasi)
  D. pLitterStreet: pola nama file (untuk grup video) + jumlah citra berisi Pile/Trash bin
  E. BePLi: folder mana yang berisi citra asli (ada 2 salinan per nama)
  F. citra rusak / tidak terbaca

Pemakaian:  python 03_probe.py  [--only taco bepli ...]  [--n-exif 300]
"""
import argparse
import collections
import glob
import json
import os
import random
import re
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
D = Path(os.path.expanduser("~/data/litter"))
IMG_EXT = {".jpg", ".jpeg", ".png"}

# file anotasi utama per dataset (dipilih dari hasil 02_inventory)
SRC = {
    "taco": dict(jsons=["taco/extracted/TACO/TACO/data/annotations.json"],
                 root="taco/extracted/TACO/TACO/data"),
    "plastopol": dict(jsons=["plastopol/extracted/PlastOpol_oneclass/plastopol-annotations_folds/annotations/full_plastopol.json"],
                      root="plastopol/extracted"),
    "coxpwd": dict(jsons=sorted(glob.glob(str(D / "coxpwd/extracted/*/*/_annotations.coco.json"))),
                   root="coxpwd/extracted"),
    "bepli": dict(jsons=["bepli/extracted/98753/plastic_coco/annotation/all_plastic_coco.json"],
                  root="bepli/extracted"),
    "plitterstreet": dict(jsons=[f"plitterstreet/extracted/annotations/annotations/4_class_splits/annotations_{s}.json"
                                 for s in ("train", "val", "test")],
                          root="plitterstreet/extracted/images"),
    "rolid": dict(jsons=[f"rolid/raw/{s}.json" for s in ("training", "validation", "testing")],
                  root="rolid/extracted"),
}


def load(name):
    imgs, anns, cats = [], [], {}
    for j in SRC[name]["jsons"]:
        p = Path(j) if os.path.isabs(j) else D / j
        d = json.load(open(p))
        tag = p.stem
        for im in d["images"]:
            im = dict(im, _id=f"{tag}:{im['id']}", _json=tag)
            imgs.append(im)
        for a in d["annotations"]:
            anns.append(dict(a, _img=f"{tag}:{a['image_id']}"))
        for c in d.get("categories", []):
            cats[c["id"]] = c.get("name")
    return imgs, anns, cats


def index_disk(root):
    by_name = collections.defaultdict(list)
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXT:
            by_name[p.name].append(p)
    return by_name


def resolve(im, root, by_name):
    fn = im["file_name"].replace("\\", "/")
    direct = root / fn
    if direct.exists():
        return direct, "path"
    c = by_name.get(os.path.basename(fn), [])
    if len(c) == 1:
        return c[0], "basename"
    if len(c) > 1:
        return c, "ambiguous"
    return None, "missing"


def exif_orient(p):
    try:
        with Image.open(p) as im:
            return im.size, im.getexif().get(0x0112, 1)
    except Exception:  # noqa: BLE001
        return None, None


def probe(name, n_exif):
    print(f"\n{'=' * 72}\n{name.upper()}")
    imgs, anns, cats = load(name)
    root = D / SRC[name]["root"]
    by_name = index_disk(root)
    n_disk = sum(len(v) for v in by_name.values())
    print(f"JSON: {len(imgs)} citra, {len(anns)} anotasi | disk ({root.relative_to(D)}): {n_disk} citra")

    # A. resolusi path
    how = collections.Counter()
    path_of = {}
    amb_example = None
    for im in imgs:
        p, h = resolve(im, root, by_name)
        how[h] += 1
        if h in ("path", "basename"):
            path_of[im["_id"]] = p
        elif h == "ambiguous" and amb_example is None:
            amb_example = (im["file_name"], [str(x.relative_to(D)) for x in p[:3]])
    print(f"A. file_name ditemukan: {dict(how)}")
    if amb_example:
        print(f"   contoh ambigu: {amb_example}")
    dup_ids = len(imgs) - len({(im['_json'], im['file_name']) for im in imgs})
    same_file_multi_json = len(imgs) - len({os.path.basename(im['file_name']) for im in imgs})
    print(f"   file_name sama di >1 entri JSON: {same_file_multi_json} (dalam satu JSON: {dup_ids})")
    unused = n_disk - len(set(path_of.values()))
    print(f"   citra di disk yang tidak dipakai JSON: {unused}")

    # B. EXIF
    rnd = random.Random(0)
    sample = rnd.sample(imgs, min(n_exif, len(imgs)))
    ex = collections.Counter()
    bad = []
    for im in sample:
        p = path_of.get(im["_id"])
        if p is None:
            continue
        size, ori = exif_orient(p)
        if size is None:
            bad.append(str(p.relative_to(D)))
            continue
        w, h = size
        rot = ori in (5, 6, 7, 8)
        jw, jh = im.get("width"), im.get("height")
        if jw is None:
            key = "json tanpa width/height"
        elif (jw, jh) == (w, h) and not rot:
            key = "cocok (tanpa rotasi EXIF)"
        elif (jw, jh) == (w, h) and rot:
            key = "JSON = ukuran MENTAH, tapi EXIF minta rotasi"
        elif (jw, jh) == (h, w) and rot:
            key = "JSON = ukuran SETELAH rotasi EXIF"
        elif (jw, jh) == (h, w):
            key = "JSON tertukar w/h tanpa EXIF"
        else:
            key = f"beda ukuran (json {jw}x{jh} vs disk {w}x{h})"
        ex[key] += 1
    print(f"B. EXIF (sampel {len(sample)}): {dict(ex)}")
    if bad:
        print(f"F. tidak terbaca: {len(bad)} -> {bad[:5]}")

    # C. statistik box (relatif terhadap ukuran citra di JSON)
    wh = {im["_id"]: (im.get("width"), im.get("height")) for im in imgs}
    per_img = collections.Counter(a["_img"] for a in anns)
    rel = []
    for a in anns:
        W, H = wh.get(a["_img"], (None, None))
        if not W or not a.get("bbox"):
            continue
        bw, bh = a["bbox"][2], a["bbox"][3]
        rel.append(np.sqrt(max(bw * bh, 0) / (W * H)))
    rel = np.array(rel)
    zero = sum(1 for im in imgs if per_img.get(im["_id"], 0) == 0)
    cnt = np.array([per_img.get(im["_id"], 0) for im in imgs])
    print(f"C. box/citra median {np.median(cnt):.0f} (p90 {np.percentile(cnt, 90):.0f}, max {cnt.max()}), "
          f"citra tanpa box: {zero}")
    if len(rel):
        print(f"   ukuran box relatif sqrt(area): p10 {np.percentile(rel, 10):.3f} | median {np.median(rel):.3f} | "
              f"p90 {np.percentile(rel, 90):.3f} | <2% sisi citra: {np.mean(rel < 0.02):.1%}")
    cc = collections.Counter(cats.get(a["category_id"]) for a in anns)
    print(f"   kategori: {dict(cc.most_common(12))}")

    # D. khusus pLitterStreet / RoLID: grup video & kelas non-litter
    if name in ("plitterstreet", "rolid"):
        names = [os.path.basename(im["file_name"]) for im in imgs]
        print(f"D. contoh nama: {rnd.sample(names, min(6, len(names)))}")
        grp = collections.Counter(re.sub(r"[_\-]?(frame)?\d+\.(jpe?g|png)$", "", n, flags=re.I) for n in names)
        sizes = sorted(grp.values())
        print(f"   grup (nama tanpa nomor akhir): {len(grp)} grup | ukuran min/median/max "
              f"{sizes[0]}/{sizes[len(sizes) // 2]}/{sizes[-1]} | 5 terbesar: {grp.most_common(5)}")
        if name == "plitterstreet":
            for cname in ("Pile", "Trash bin", "Face mask"):
                cid = [k for k, v in cats.items() if v == cname]
                ids = {a["_img"] for a in anns if a["category_id"] in cid}
                print(f"   citra yang memuat '{cname}': {len(ids)}")
            only_pile = 0
            by_img = collections.defaultdict(set)
            for a in anns:
                by_img[a["_img"]].add(cats.get(a["category_id"]))
            only_pile = sum(1 for s in by_img.values() if s <= {"Pile", "Trash bin"})
            print(f"   citra yang HANYA berisi Pile/Trash bin: {only_pile}")

    # E. BePLi: dua salinan per nama
    if name == "bepli":
        dirs = collections.Counter(str(p.parent.relative_to(D)) for v in by_name.values() for p in v)
        print(f"E. folder citra: {dict(dirs)}")
        k = next(iter(by_name))
        for p in by_name[k]:
            with Image.open(p) as im:
                print(f"   {p.relative_to(D)}: mode {im.mode}, ukuran {im.size}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", choices=list(SRC))
    ap.add_argument("--n-exif", type=int, default=300)
    args = ap.parse_args()
    for name in args.only or list(SRC):
        try:
            probe(name, args.n_exif)
        except Exception as e:  # noqa: BLE001
            print(f"\n{name}: GAGAL - {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
