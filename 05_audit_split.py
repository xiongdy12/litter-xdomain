#!/usr/bin/env python3
"""
05_audit_split.py - Audit duplikat (DINOv2 -> ORB/RANSAC) lalu split ulang per grup.

Langkah:
  1. Grup awal: pLitterStreet per chapter video GoPro, RoLID per trim rekaman, lainnya per citra.
  2. Embedding DINOv2-S (CLS, ternormalisasi) untuk semua citra harmonized (di-cache).
  3. Kandidat: k tetangga terdekat dengan cosine >= --sim-cand.
  4. Verifikasi geometris ORB + RANSAC (homografi). Duplikat bila:
        cosine >= --sim-auto, ATAU (cosine >= --sim-cand DAN inlier >= --min-inliers)
     Dataset kendaraan (RoLID, pLitterStreet): area statis (kap mobil, overlay jam/GPS) dideteksi
     dari variasi piksel antar frame dan dikeluarkan dari ORB; tidak ada terima-otomatis.
  5. Union-find: grup yang punya duplikat terverifikasi digabung (dalam dataset yang sama).
     Duplikat ANTAR dataset ditandai -> dikeluarkan dari val/test (tetap boleh di train).
  6. Audit split RESMI: berapa % citra test/val resmi yang punya near-duplicate di train resmi.
  7. Split baru per dataset 70/15/15 berbasis grup (greedy, seed tetap).

Output (~/data/litter/audit/ dan ~/data/litter/splits/):
  emb_dinov2s.npy, pairs.csv (semua pasangan kandidat + hasil verifikasi),
  meta_split.csv, splits/<dataset>_{train,val,test}.txt, audit_report.txt

Pemakaian:
  python 05_audit_split.py --device 0
"""
import argparse
import csv
import os
import random
import re
from collections import Counter, defaultdict
import multiprocessing as mp
from pathlib import Path

import cv2
import numpy as np

D = Path(os.path.expanduser("~/data/litter"))
H = D / "harmonized"
A = D / "audit"
S = D / "splits"


# ---------------------------------------------------------------- grup awal
def base_group(row):
    ds, b = row["dataset"], os.path.basename(row["src"])
    if ds == "plitterstreet":
        m = re.match(r"^(.+?)_G[XHP](\d{2})(\d{4})_\d+\.\w+$", b)
        if m:
            return f"{ds}:{m.group(1)}_vid{m.group(3)}_ch{m.group(2)}"
    if ds == "rolid":
        m = re.match(r"^(.+?)_frame\d+\.\w+$", b)
        if m:
            return f"{ds}:{m.group(1)}"
    return f"{ds}:img:{row['uid']}"


# ---------------------------------------------------------------- embedding
def embed(uids, device, bs=256, workers=16):
    import timm
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0,
                              img_size=224).eval().to(device)
    cfg = timm.data.resolve_data_config({}, model=model)
    mean = torch.tensor(cfg["mean"]).view(3, 1, 1)
    std = torch.tensor(cfg["std"]).view(3, 1, 1)

    class DS(Dataset):
        def __len__(self):
            return len(uids)

        def __getitem__(self, i):
            im = Image.open(H / "images" / f"{uids[i]}.jpg").convert("RGB").resize((224, 224), Image.BILINEAR)
            x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1)
            return (x - mean) / std

    out = []
    with torch.no_grad():
        for k, x in enumerate(DataLoader(DS(), batch_size=bs, num_workers=workers)):
            with torch.autocast("cuda", dtype=torch.float16, enabled=device != "cpu"): f = model(x.to(device, non_blocking=True))
            out.append(torch.nn.functional.normalize(f.float(), dim=1).cpu())
            if k % 20 == 0:
                print(f"  embedding {min((k + 1) * bs, len(uids))}/{len(uids)}", flush=True)
    return torch.cat(out).numpy().astype(np.float32)


def knn_pairs(emb, k, thr, device):
    import torch
    E = torch.from_numpy(emb).to(device).half()
    pairs = {}
    for s in range(0, len(E), 4096):
        sim = E[s:s + 4096] @ E.T
        v, idx = sim.float().topk(min(k + 1, len(E)), dim=1)
        v, idx = v.cpu().numpy(), idx.cpu().numpy()
        for r in range(v.shape[0]):
            i = s + r
            for sv, j in zip(v[r], idx[r]):
                if j == i or sv < thr:
                    continue
                a, b = (i, int(j)) if i < j else (int(j), i)
                pairs[(a, b)] = max(pairs.get((a, b), 0), float(sv))
    return pairs


# ---------------------------------------------------------------- ORB/RANSAC
VEHICLE = {"rolid", "plitterstreet"}  # kamera kendaraan: bisa ada kap mobil / overlay teks statis
_orb = None
_masks = {}


def static_mask(ds, uids_ds, n=300, thr=10.0, seed=0):
    """Area statis (kap mobil, overlay jam/GPS) = piksel yang hampir tidak berubah antar frame.
    Kembalikan mask uint8 800x450: 255 = boleh dipakai ORB, 0 = statis (dibuang)."""
    rnd = random.Random(seed)
    pick = rnd.sample(uids_ds, min(n, len(uids_ds)))
    stack = []
    for u in pick:
        g = cv2.imread(str(H / "images" / f"{u}.jpg"), cv2.IMREAD_GRAYSCALE)
        if g is not None:
            stack.append(cv2.resize(g, (800, 450), interpolation=cv2.INTER_AREA).astype(np.float32))
    sd = np.std(np.stack(stack), axis=0)
    static = (sd < thr).astype(np.uint8)
    static = cv2.dilate(static, np.ones((15, 15), np.uint8))
    mask = ((1 - static) * 255).astype(np.uint8)
    cv2.imwrite(str(A / f"static_mask_{ds}.png"), mask)
    return mask, float(static.mean())


def _load_gray(uid, max_side=800):
    g = cv2.imread(str(H / "images" / f"{uid}.jpg"), cv2.IMREAD_GRAYSCALE)
    if g is None:
        return None
    s = max_side / max(g.shape)
    return cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else g


def _mask_for(ds, shape):
    if ds not in VEHICLE:
        return None
    if ds not in _masks:
        p = A / f"static_mask_{ds}.png"
        _masks[ds] = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) if p.exists() else None
    m = _masks[ds]
    return None if m is None else cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)


def orb_inliers(args):
    ua, ub, dsa, dsb = args
    global _orb
    if _orb is None:
        cv2.setNumThreads(1)
        _orb = cv2.ORB_create(nfeatures=2000)
    ga, gb = _load_gray(ua), _load_gray(ub)
    if ga is None or gb is None:
        return 0
    ka, da = _orb.detectAndCompute(ga, _mask_for(dsa, ga.shape))
    kb, db = _orb.detectAndCompute(gb, _mask_for(dsb, gb.shape))
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return 0
    m = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    good = [p[0] for p in m if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(good) < 8:
        return len(good) // 2
    src = np.float32([ka[g.queryIdx].pt for g in good])
    dst = np.float32([kb[g.trainIdx].pt for g in good])
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return int(mask.sum()) if mask is not None else 0


# ---------------------------------------------------------------- union-find
class UF:
    def __init__(self):
        self.p = {}

    def f(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def u(self, a, b):
        ra, rb = self.f(a), self.f(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


# ---------------------------------------------------------------- split
def group_split(groups_sizes, frac=(0.70, 0.15, 0.15), seed=0):
    items = list(groups_sizes.items())
    random.Random(seed).shuffle(items)          # acak dulu supaya grup sama besar tidak berurutan
    items.sort(key=lambda kv: -kv[1])           # lalu grup besar didahulukan
    total = sum(groups_sizes.values())
    names = ("train", "val", "test")
    have = dict.fromkeys(names, 0)
    assign = {}
    for g, n in items:
        deficit = {s: frac[i] * total - have[s] for i, s in enumerate(names)}
        s = max(deficit, key=deficit.get)
        assign[g] = s
        have[s] += n
    return assign, {s: have[s] / total for s in names}


OFFICIAL = {  # nama split resmi -> peran
    "plastopol": {"trainval": "train", "test": "test"},
    "coxpwd": {"train": "train", "valid": "val", "test": "test"},
    "bepli": {"train": "train", "val": "val", "test": "test"},
    "plitterstreet": {"train": "train", "val": "val", "test": "test"},
    "rolid": {"training": "train", "validation": "val", "testing": "test"},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="0")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--sim-cand", type=float, default=0.90)
    ap.add_argument("--sim-auto", type=float, default=0.97)
    ap.add_argument("--min-inliers", type=int, default=30)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = "cpu" if args.device == "cpu" else f"cuda:{args.device}"
    A.mkdir(parents=True, exist_ok=True)
    S.mkdir(parents=True, exist_ok=True)
    rep = []

    def say(s=""):
        print(s, flush=True)
        rep.append(s)

    rows = list(csv.DictReader(open(H / "meta.csv")))
    rows.sort(key=lambda r: r["uid"])
    uids = [r["uid"] for r in rows]
    for r in rows:
        r["group0"] = base_group(r)
    say(f"citra: {len(rows)}")

    # 2. embedding
    ef = A / "emb_dinov2s.npy"
    uf_ = A / "emb_uids.txt"
    if ef.exists() and uf_.exists() and uf_.read_text().split() == uids:
        emb = np.load(ef)
        print("embedding dari cache")
    else:
        emb = embed(uids, dev, workers=args.workers)
        np.save(ef, emb)
        uf_.write_text("\n".join(uids))

    # 3. kandidat
    cand = knn_pairs(emb, args.k, args.sim_cand, dev)
    say(f"pasangan kandidat (cos >= {args.sim_cand}): {len(cand)}")

    # pasangan yang perlu diverifikasi: beda grup ATAU beda split resmi
    todo, info = [], {}
    for (i, j), sv in cand.items():
        ri, rj = rows[i], rows[j]
        same_g = ri["group0"] == rj["group0"]
        same_off = ri["dataset"] == rj["dataset"] and ri["official_split"] == rj["official_split"]
        info[(i, j)] = (sv, same_g)
        veh = ri["dataset"] in VEHICLE or rj["dataset"] in VEHICLE
        if not (same_g and same_off) and (sv < args.sim_auto or veh):
            todo.append((i, j))
    for ds in sorted(VEHICLE):
        u_ds = [r["uid"] for r in rows if r["dataset"] == ds]
        if u_ds:
            _, frac = static_mask(ds, u_ds)
            say(f"mask statis {ds}: {frac:.1%} area dibuang dari ORB (kap mobil/overlay) -> audit/static_mask_{ds}.png")
    say(f"diverifikasi ORB/RANSAC: {len(todo)} pasangan (dataset kendaraan: selalu diverifikasi, tanpa terima-otomatis)")
    # 'spawn' (bukan fork): fork setelah torch/OpenMP aktif bisa macet (deadlock)
    with mp.get_context("spawn").Pool(args.workers) as pool:
        inl = pool.map(orb_inliers, [(uids[i], uids[j], rows[i]["dataset"], rows[j]["dataset"]) for i, j in todo], chunksize=64)
    inliers = dict(zip(todo, inl))

    dup = {}
    with open(A / "pairs.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["uid_a", "uid_b", "ds_a", "ds_b", "cos", "inliers", "same_group0", "dup"])
        for (i, j), (sv, same_g) in info.items():
            n = inliers.get((i, j), -1)
            veh = rows[i]["dataset"] in VEHICLE or rows[j]["dataset"] in VEHICLE
            is_dup = (sv >= args.sim_auto and not veh) or n >= args.min_inliers
            dup[(i, j)] = is_dup
            w.writerow([uids[i], uids[j], rows[i]["dataset"], rows[j]["dataset"], f"{sv:.4f}", n, int(same_g), int(is_dup)])

    # 5. union-find & duplikat antar dataset
    uf = UF()
    for r in rows:
        uf.f(r["group0"])
    xdup = set()
    xpairs = Counter()
    for (i, j), d in dup.items():
        if not d:
            continue
        if rows[i]["dataset"] == rows[j]["dataset"]:
            uf.u(rows[i]["group0"], rows[j]["group0"])
        else:
            xdup.update((i, j))
            xpairs[tuple(sorted((rows[i]["dataset"], rows[j]["dataset"])))] += 1
    for r in rows:
        r["group"] = uf.f(r["group0"])
    say(f"duplikat ANTAR dataset: {sum(xpairs.values())} pasangan {dict(xpairs)} -> {len(xdup)} citra dilarang masuk test")

    # 6. audit split resmi
    say("\n=== AUDIT SPLIT RESMI (near-duplicate terverifikasi ATAU satu video/rekaman yang sama) ===")
    nbr = defaultdict(set)
    for (i, j), d in dup.items():
        if d:
            nbr[i].add(j)
            nbr[j].add(i)
    for ds, roles in OFFICIAL.items():
        idx = [k for k, r in enumerate(rows) if r["dataset"] == ds]
        role = {k: roles.get(rows[k]["official_split"]) for k in idx}
        train_groups = {rows[k]["group0"] for k in idx if role[k] == "train"}
        for part in ("val", "test"):
            P = [k for k in idx if role[k] == part]
            if not P:
                continue
            leak_dup = sum(1 for k in P if any(role.get(j) == "train" for j in nbr[k]))
            leak_grp = sum(1 for k in P if not rows[k]["group0"].split(":")[1] == "img"
                           and rows[k]["group0"] in train_groups)
            leak_any = sum(1 for k in P if any(role.get(j) == "train" for j in nbr[k])
                           or (rows[k]["group0"].split(":")[1] != "img" and rows[k]["group0"] in train_groups))
            say(f"{ds:14s} {part:4s} resmi: {len(P):5d} citra | punya near-dup di train: {leak_dup / len(P):6.1%} | "
                f"video/rekaman sama dgn train: {leak_grp / len(P):6.1%} | total bocor: {leak_any / len(P):6.1%}")

    # 7. split baru
    say("\n=== SPLIT BARU (berbasis grup, 70/15/15) ===")
    out_rows = []
    for ds in sorted({r["dataset"] for r in rows}):
        R = [r for r in rows if r["dataset"] == ds]
        sizes = Counter(r["group"] for r in R)
        assign, got = group_split(sizes, seed=args.seed)
        n0 = len({r["group0"] for r in R})
        moved = 0
        for r in R:
            r["split"] = assign[r["group"]]
            r["xdup"] = 0
        for k in xdup:
            if rows[k]["dataset"] == ds:
                rows[k]["xdup"] = 1
                if rows[k]["split"] in ("val", "test"):
                    # jangan dipindah ke train (grup-nya bisa ada di test -> bocor); keluarkan saja
                    rows[k]["split"] = "excluded"
                    moved += 1
        big = sizes.most_common(1)[0]
        say(f"{ds:14s} grup awal {n0:6d} -> setelah gabung duplikat {len(sizes):6d} | grup terbesar {big[1]} citra "
            f"({big[1] / len(R):.1%}) | train/val/test = {got['train']:.1%}/{got['val']:.1%}/{got['test']:.1%}"
            f" | xdup dikeluarkan dari val/test: {moved}")
        for part in ("train", "val", "test"):
            paths = [str(H / "images" / f"{r['uid']}.jpg") for r in R if r["split"] == part]
            (S / f"{ds}_{part}.txt").write_text("\n".join(paths) + "\n")
        out_rows += R

    fields = list(out_rows[0].keys())
    with open(A / "meta_split.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)
    (A / "audit_report.txt").write_text("\n".join(rep) + "\n")
    print(f"\nselesai -> {A / 'audit_report.txt'}, {A / 'meta_split.csv'}, {S}/")


if __name__ == "__main__":
    main()
