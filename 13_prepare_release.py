#!/usr/bin/env python3
"""
13_prepare_release.py -- salin split, hasil audit dan hasil evaluasi ke release_data/ untuk GitHub/Zenodo.
Tidak ada citra yang disalin; path absolut server dihapus (hanya uid).
    python 13_prepare_release.py
"""
import csv
import os
import shutil
from pathlib import Path

D = Path(os.path.expanduser(os.environ.get("LITTER_DATA", "~/data/litter")))
W = Path(os.path.expanduser(os.environ.get("LITTER_WORK", "~/litter-xdomain/work")))
R = Path(__file__).resolve().parent / "release_data"
DROP = {"src"}  # kolom berisi path absolut di server

def main():
    if R.exists():
        shutil.rmtree(R)
    for sub in ("splits", "audit", "results"):
        (R / sub).mkdir(parents=True)
    n = 0
    for f in sorted((D / "splits").glob("*.txt")):
        uids = [Path(p).stem for p in f.read_text().split()]
        (R / "splits" / f.name).write_text("\n".join(uids) + "\n")
        n += len(uids)
    print(f"splits: {len(list((R / 'splits').glob('*.txt')))} file, {n} uid")
    rows = list(csv.DictReader(open(D / "audit" / "meta_split.csv")))
    fields = [k for k in rows[0] if k not in DROP]
    with open(R / "audit" / "meta_split.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"meta_split.csv: {len(rows)} baris, kolom {fields}")
    k = 0
    with open(D / "audit" / "pairs.csv") as fi, open(R / "audit" / "verified_duplicates.csv", "w", newline="") as fo:
        r = csv.DictReader(fi)
        w = csv.DictWriter(fo, fieldnames=r.fieldnames)
        w.writeheader()
        for row in r:
            if row["dup"] == "1":
                w.writerow(row)
                k += 1
    print(f"verified_duplicates.csv: {k} pasangan")
    shutil.copy(D / "audit" / "audit_report.txt", R / "audit" / "audit_report.txt")
    for f in sorted((W / "results").glob("*")):
        if f.suffix in (".csv", ".md") and f.is_file():
            shutil.copy(f, R / "results" / f.name)
    print("results:", ", ".join(p.name for p in sorted((R / "results").iterdir())))
    size = sum(p.stat().st_size for p in R.rglob("*") if p.is_file())
    print(f"\nrelease_data/ siap: {size / 1e6:.1f} MB")

if __name__ == "__main__":
    main()
