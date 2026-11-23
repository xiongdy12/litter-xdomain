#!/usr/bin/env python3
"""
01_download.py - Unduh 6 dataset sampah publik untuk studi lintas dataset.

Semua sumber bisa diunduh langsung tanpa permohonan akses.
Unduhan memakai `wget -c`, jadi kalau putus cukup jalankan ulang (melanjutkan).

Pemakaian:
    python 01_download.py --list            # cek semua sumber, TANPA mengunduh
    python 01_download.py                   # unduh semua
    python 01_download.py --only taco       # unduh satu dataset saja
    python 01_download.py --verify          # cek checksum file yang sudah ada

Hasil: ~/data/litter/<nama>/raw/  +  ~/data/litter/manifest.json
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(os.path.expanduser("~/data/litter"))
UA = {"User-Agent": "Mozilla/5.0 (litter-xdomain research script)"}

# region / viewpoint dipakai nanti untuk desain faktorial geografi x sudut pandang
DATASETS = {
    "taco": dict(
        src="zenodo", record="3587843", latest=False,
        region="Europe", view="handheld", license="CC BY 4.0",
        cite="Proenca & Simoes 2020, arXiv:2003.06975"),
    "plastopol": dict(
        src="zenodo", record="5829156", latest=False,
        region="Global (citizen science)", view="handheld", license="CC BY 4.0",
        cite="Cordova et al. 2022, Sensors 22(2):548"),
    "coxpwd": dict(
        src="mendeley", record="bdzg4tjy63", version=3,
        region="South Asia (Bangladesh)", view="handheld", license="CC BY 4.0",
        cite="Mendeley Data 10.17632/bdzg4tjy63.3"),
    "bepli": dict(
        src="url", url="https://www.seanoe.org/data/00811/92297/data/98753.zip",
        fname="98753.zip",
        region="East Asia (Japan)", view="handheld", license="CC BY-NC-SA 4.0",
        cite="Hidaka et al. 2023, Data in Brief 48:109176"),
    "plitterstreet": dict(
        src="zenodo", record="8288500", latest=False,
        region="SE/South Asia (TH, VN, LK)", view="vehicle", license="CC BY 4.0",
        cite="pLitterStreet, arXiv:2401.14719"),
    "rolid": dict(
        src="gdrive", url="https://drive.google.com/drive/folders/1aPPW76js_JKdx_9SF0tioxjs17sLbOn-",
        region="Europe (UK)", view="vehicle", license="Apache-2.0",
        cite="Wu et al. 2026, arXiv:2601.00398"),
}


def get_json(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def head_size(url):
    req = urllib.request.Request(url, headers=UA, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, int(r.headers.get("Content-Length") or 0)
    except Exception as e:  # noqa: BLE001
        return f"ERROR {e}", 0


def resolve(name, cfg):
    """Kembalikan daftar file: [{fname, url, size, md5|sha256|None}] + metadata."""
    meta = {}
    files = []
    if cfg["src"] == "zenodo":
        url = f"https://zenodo.org/api/records/{cfg['record']}"
        if cfg.get("latest"):
            url += "/versions/latest"
        rec = get_json(url)
        meta = dict(record_id=rec["id"], doi=rec.get("doi"),
                    version=rec["metadata"].get("version"),
                    license=(rec["metadata"].get("license") or {}).get("id"))
        for f in rec["files"]:
            ck = f.get("checksum", "")
            files.append(dict(fname=f["key"], url=f["links"]["self"], size=f["size"],
                              md5=ck.split(":", 1)[1] if ck.startswith("md5:") else None))
    elif cfg["src"] == "mendeley":
        url = (f"https://data.mendeley.com/public-api/datasets/{cfg['record']}"
               f"/files?folder_id=root&version={cfg['version']}")
        for f in get_json(url):
            cd = f.get("content_details", {})
            files.append(dict(fname=f["filename"], url=cd["download_url"], size=f.get("size", 0),
                              sha256=cd.get("sha256_hash")))
        meta = dict(record_id=cfg["record"], version=cfg["version"])
    elif cfg["src"] == "url":
        status, size = head_size(cfg["url"])
        files.append(dict(fname=cfg["fname"], url=cfg["url"], size=size, http=status))
    elif cfg["src"] == "gdrive":
        files.append(dict(fname="(folder Google Drive)", url=cfg["url"], size=0))
    return meta, files


def file_hash(path, algo):
    h = hashlib.new(algo)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def check(path, f):
    if not path.exists():
        return "BELUM ADA"
    if f.get("size") and path.stat().st_size != f["size"]:
        return f"UKURAN BEDA ({path.stat().st_size} vs {f['size']}) - jalankan ulang untuk melanjutkan"
    if f.get("md5"):
        return "OK (md5)" if file_hash(path, "md5") == f["md5"] else "MD5 SALAH - hapus file lalu unduh ulang"
    if f.get("sha256"):
        return "OK (sha256)" if file_hash(path, "sha256") == f["sha256"] else "SHA256 SALAH - hapus lalu unduh ulang"
    return "ADA (tanpa checksum)"


def gb(n):
    return f"{n / 1e9:6.2f} GB" if n else "   ?    "


def download(name, cfg, files):
    raw = ROOT / name / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    if cfg["src"] == "gdrive":
        if shutil.which("gdown") is None:
            sys.exit("gdown belum terpasang. Jalankan: pip install gdown")
        subprocess.run(["gdown", "--folder", "--continue", cfg["url"], "-O", str(raw)], check=True)
        return
    use_aria = shutil.which("aria2c") is not None
    for f in files:
        out = raw / f["fname"]
        ctrl = Path(str(out) + ".aria2")  # file kontrol aria2 = unduhan belum selesai
        if f.get("size") and out.exists() and out.stat().st_size == f["size"] and not ctrl.exists():
            print(f"  [lewati] {f['fname']} sudah lengkap")
            continue
        print(f"  [unduh] {f['fname']} {gb(f.get('size', 0))} via {'aria2c x8' if use_aria else 'wget'}")
        if use_aria:
            # 8 koneksi paralel; -c melanjutkan file parsial (termasuk hasil wget)
            subprocess.run(["aria2c", "-c", "-x", "8", "-s", "8", "-k", "20M",
                            "--max-tries=0", "--retry-wait=10", "--timeout=60",
                            "--file-allocation=none", "--summary-interval=60",
                            "--console-log-level=warn", "--auto-file-renaming=false",
                            "-d", str(raw), "-o", f["fname"], f["url"]], check=True)
        else:
            subprocess.run(["wget", "-c", "--tries=0", "--retry-connrefused", "--waitretry=5",
                            "--timeout=60", "--progress=dot:giga", "-O", str(out), f["url"]], check=True)
        print(f"  [cek]   {check(out, f)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="cek sumber saja, tidak mengunduh")
    ap.add_argument("--verify", action="store_true", help="cek checksum file yang sudah diunduh")
    ap.add_argument("--only", nargs="+", choices=list(DATASETS), help="pilih dataset tertentu")
    args = ap.parse_args()

    names = args.only or list(DATASETS)
    ROOT.mkdir(parents=True, exist_ok=True)
    mpath = ROOT / "manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}

    total = 0
    for name in names:
        cfg = DATASETS[name]
        print(f"\n=== {name}  [{cfg['region']} | {cfg['view']} | {cfg['license']}]")
        try:
            meta, files = resolve(name, cfg)
        except Exception as e:  # noqa: BLE001
            print(f"  GAGAL membaca sumber: {e}")
            continue
        if meta:
            print("  meta:", meta)
        for f in files:
            total += f.get("size") or 0
            extra = f"  HTTP {f['http']}" if "http" in f else ""
            print(f"  - {f['fname']:<45} {gb(f.get('size', 0))}{extra}")
        manifest[name] = dict(cfg, meta=meta, files=files)
        if args.verify:
            for f in files:
                if cfg["src"] != "gdrive":
                    print(f"  [cek] {f['fname']}: {check(ROOT / name / 'raw' / f['fname'], f)}")
        elif not args.list:
            download(name, cfg, files)

    mpath.write_text(json.dumps(manifest, indent=2, default=str))
    free = shutil.disk_usage(ROOT).free
    print(f"\nTotal ukuran yang diketahui: {gb(total)}  |  ruang kosong di {ROOT}: {gb(free)}")
    if args.list:
        print("Mode --list: tidak ada yang diunduh.")


if __name__ == "__main__":
    main()
