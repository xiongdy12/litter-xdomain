#!/usr/bin/env python3
r"""
12_paper_assets.py -- semua TABEL (LaTeX) dan GAMBAR (PDF/JPG) untuk paper MTAP, langsung dari file hasil.

Tidak butuh GPU. Jalankan setelah semua eksperimen selesai:
    python 12_paper_assets.py
Output:
    ~/litter-xdomain/work/paper_assets/tables/*.tex     (tabular booktabs, siap \input)
    ~/litter-xdomain/work/paper_assets/figures/*.pdf|jpg
    ~/litter-xdomain/work/paper_assets/summary.md       (semua angka kunci untuk menulis teks)
    ~/litter-xdomain/work/paper_assets/errors.log       (exhibit yang gagal, kalau ada)
    ~/litter-xdomain/work/paper_assets.zip              (kirim file ini)
Setiap exhibit berdiri sendiri: kalau satu gagal, yang lain tetap dibuat.
"""
import argparse
import csv
import json
import os
import re
import traceback
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

D = Path(os.path.expanduser(os.environ.get("LITTER_DATA", "~/data/litter")))
W = Path(os.path.expanduser(os.environ.get("LITTER_WORK", "~/litter-xdomain/work")))
OUT = W / "paper_assets"
TAB, FIG = OUT / "tables", OUT / "figures"

DS = ["taco", "plastopol", "coxpwd", "bepli", "plitterstreet", "rolid"]
NAME = {"taco": "TACO", "plastopol": "PlastOPol", "coxpwd": "CoxPWD", "bepli": "BePLi",
        "plitterstreet": "pLitterStreet", "rolid": "RoLID-11K"}
SHORT = {"taco": "TACO", "plastopol": "PlastOPol", "coxpwd": "CoxPWD", "bepli": "BePLi",
         "plitterstreet": "pLitter", "rolid": "RoLID"}
VIEW = {"taco": "hand", "plastopol": "hand", "coxpwd": "hand", "bepli": "hand",
        "plitterstreet": "vehicle", "rolid": "vehicle"}
REGION = {"taco": "EU", "rolid": "EU", "coxpwd": "Asia", "bepli": "Asia", "plitterstreet": "Asia",
          "plastopol": "Global"}
REGION_TXT = {"EU": "Europe", "Asia": "Asia", "Global": "Global"}
VIEW_TXT = {"hand": "Handheld", "vehicle": "Vehicle"}
OFFICIAL = {
    "plastopol": {"trainval": "train", "test": "test"},
    "coxpwd": {"train": "train", "valid": "val", "test": "test"},
    "bepli": {"train": "train", "val": "val", "test": "test"},
    "plitterstreet": {"train": "train", "val": "val", "test": "test"},
    "rolid": {"training": "train", "validation": "val", "testing": "test"},
}
BASE = "yolo11m"
SCALE_SEED = "0"   # kurva multi-resolusi hanya lengkap untuk seed 0
IMGSZ = 1024

# palet kategorikal tervalidasi (urutan tetap, warna mengikuti DATASET)
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#0ca30c", "#6250d6", "#e34948"]
COLOR = dict(zip(DS, PAL))
MARK = dict(zip(DS, ["o", "s", "^", "D", "v", "P"]))
SEQ = ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
TW = 372 / 72.27  # \textwidth sn-jnl dalam inci

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "lines.linewidth": 1.4,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "legend.frameon": False,
})

S = {}            # angka kunci -> summary.md
LOG = []


def note(k, v):
    S[k] = v


def f3(x):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def pm(v):
    if not v:
        return "--"
    sd = np.std(v, ddof=1) if len(v) > 1 else None
    return f"{np.mean(v):.3f}" + (f"{{\\scriptsize\\,$\\pm${sd:.3f}".replace("$0.", "$.") + "}" if sd is not None else "")


def write_tab(name, header, rows, align=None, midrules=(), note_=None):
    """tabular booktabs. header: list str; rows: list of list str."""
    align = align or "l" + "c" * (len(header) - 1)
    L = [f"\\begin{{tabular}}{{{align}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    for i, r in enumerate(rows):
        if i in midrules:
            L.append("\\midrule")
        L.append(" & ".join(r) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    if note_:
        L.append(f"% {note_}")
    (TAB / f"{name}.tex").write_text("\n".join(L) + "\n")


def savefig(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ data
def load_results():
    rows = {}
    for r in csv.DictReader(open(W / "results" / "results.csv")):
        k = (r["model"], r["train_ds"], str(r["seed"]), r["test_ds"], int(r["imgsz_train"]), int(r["imgsz_eval"]))
        rows[k] = r  # baris terakhir menang (rerun)
    return rows


RES = None


def vals(model, s, t, metric="map50", ev=IMGSZ, tr=IMGSZ, seeds=None):
    out = []
    for (m, a, sd, b, it, ie), r in RES.items():
        if m == model and a == s and b == t and it == tr and ie == ev and (seeds is None or sd in seeds):
            out.append(float(r[metric]))
    return out


def models_present():
    return sorted({k[0] for k in RES})


def meta_rows():
    return list(csv.DictReader(open(D / "audit" / "meta_split.csv")))


def label_boxes(uid):
    p = D / "harmonized" / "labels" / f"{uid}.txt"
    out = []
    if p.exists():
        for ln in p.read_text().splitlines():
            v = ln.split()
            if len(v) == 5:
                out.append([float(x) for x in v[1:]])
    return out


def train_scale(ds):
    sizes = []
    for p in (D / "splits" / f"{ds}_train.txt").read_text().split():
        for b in label_boxes(Path(p).stem):
            sizes.append(np.sqrt(b[2] * b[3]))
    return float(np.median(sizes))


SCALE = {}


# ------------------------------------------------------------------ E1 dataset summary + size distribution
def e1_datasets():
    M = meta_rows()
    per = defaultdict(list)
    for r in M:
        per[r["dataset"]].append(r)
    rows, rel = [], {}
    for ds in DS:
        R = per[ds]
        nb, sizes, cls = 0, [], Counter()
        for r in R:
            W_, H_ = float(r["w"]), float(r["h"])
            s = IMGSZ / max(W_, H_)
            for b in label_boxes(r["uid"]):
                nb += 1
                sizes.append(np.sqrt(b[2] * b[3]))
                a = (b[2] * W_ * s) * (b[3] * H_ * s)
                cls["S" if a < 32 ** 2 else "M" if a < 96 ** 2 else "L"] += 1
        rel[ds] = np.array(sizes)
        sp = Counter(r["split"] for r in R)
        n = len(R)
        rows.append([NAME[ds], VIEW_TXT[VIEW[ds]], REGION_TXT[REGION[ds]], f"{n:,}", f"{nb:,}", f"{nb / n:.1f}",
                     f"{SCALE[ds]:.3f}", f"{100 * cls['S'] / nb:.0f}/{100 * cls['M'] / nb:.0f}/{100 * cls['L'] / nb:.0f}",
                     f"{sp['train']:,}/{sp['val']:,}/{sp['test']:,}"])
        note(f"ds.{ds}", dict(images=n, boxes=nb, median_rel_size_train=round(SCALE[ds], 4),
                              small_pct=round(100 * cls["S"] / nb, 1), large_pct=round(100 * cls["L"] / nb, 1),
                              split=dict(sp)))
    note("ds.total_images", sum(len(per[d]) for d in DS))
    note("ds.total_boxes", sum(int(S[f"ds.{d}"]["boxes"]) for d in DS))
    write_tab("tab_datasets",
              ["Dataset", "Viewpoint", "Region", "Images", "Boxes", "Boxes/img", "Med. size",
               "S/M/L (\\%)", "Train/val/test"],
              rows, align="lllrrrccc")

    fig, ax = plt.subplots(figsize=(TW * 0.62, 2.3))
    for ds in sorted(DS, key=lambda d: SCALE[d]):
        x = np.sort(rel[ds])
        ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=COLOR[ds], label=f"{NAME[ds]} ({VIEW_TXT[VIEW[ds]].lower()})")
    ax.set_xscale("log")
    ax.set_xlabel(r"Relative object size $\sqrt{wh}$ (fraction of image side)")
    ax.set_ylabel("Cumulative fraction of boxes")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    savefig(fig, "fig_size_ecdf")


# ------------------------------------------------------------------ E2 leakage audit tables
def e2_audit():
    rep = (D / "audit" / "audit_report.txt").read_text().splitlines()
    rx = re.compile(r"^(\w+)\s+(val|test)\s+resmi:\s+(\d+) citra \| punya near-dup di train:\s+([\d.]+)% \| "
                    r"video/rekaman sama dgn train:\s+([\d.]+)% \| total bocor:\s+([\d.]+)%")
    rows = []
    for ln in rep:
        m = rx.match(ln.strip())
        if m:
            ds, part, n, a, b, c = m.groups()
            rows.append([NAME.get(ds, ds), part, f"{int(n):,}", f"{float(a):.1f}", f"{float(b):.1f}",
                         f"\\textbf{{{float(c):.1f}}}" if float(c) >= 10 else f"{float(c):.1f}"])
            note(f"leak.{ds}.{part}", dict(n=int(n), neardup=float(a), same_recording=float(b), any=float(c)))
    order = {NAME[d]: i for i, d in enumerate(DS)}
    rows.sort(key=lambda r: (order.get(r[0], 99), r[1] != "test"))
    write_tab("tab_leakage", ["Dataset", "Part", "Images", "Near-dup.\\ (\\%)",
                              "Same recording (\\%)", "Leaked (\\%)"], rows, align="llrccc")
    rx2 = re.compile(r"^(\w+)\s+grup awal\s+(\d+) -> setelah gabung duplikat\s+(\d+) \| grup terbesar (\d+) citra "
                     r"\(([\d.]+)%\) \| train/val/test = ([\d.]+)%/([\d.]+)%/([\d.]+)% \| xdup dikeluarkan dari val/test: (\d+)")
    rows2 = []
    for ln in rep:
        m = rx2.match(ln.strip())
        if m:
            ds, g0, g1, big, bigp, tr, va, te, xd = m.groups()
            rows2.append([NAME.get(ds, ds), f"{int(g0):,}", f"{int(g1):,}", f"{bigp}", f"{tr}/{va}/{te}", xd])
            note(f"split.{ds}", dict(groups_before=int(g0), groups_after=int(g1), largest_pct=float(bigp),
                                     train=float(tr), val=float(va), test=float(te), xdup_removed=int(xd)))
    rows2.sort(key=lambda r: order.get(r[0], 99))
    write_tab("tab_newsplit", ["Dataset", "Groups, initial", "Groups, merged", "Largest (\\%)",
                               "Train/val/test (\\%)", "Cross-dup."], rows2, align="lrrccr")
    for ln in rep:
        if "duplikat ANTAR dataset" in ln:
            note("audit.cross_dataset_line", ln.strip())
    # jumlah pasangan duplikat terverifikasi per dataset
    cnt = Counter()
    pp = D / "audit" / "pairs.csv"
    if pp.exists():
        for r in csv.DictReader(open(pp)):
            if r["dup"] == "1":
                cnt[(r["ds_a"], r["ds_b"])] += 1
        note("audit.verified_pairs", {f"{a}|{b}": c for (a, b), c in cnt.items()})


# ------------------------------------------------------------------ E3 audit example figure
def e3_audit_fig():
    from PIL import Image
    meta = {r["uid"]: r for r in meta_rows()}
    role = {u: OFFICIAL.get(r["dataset"], {}).get(r["official_split"]) for u, r in meta.items()}
    cand = defaultdict(list)
    for r in csv.DictReader(open(D / "audit" / "pairs.csv")):
        if r["dup"] != "1" or r["ds_a"] != r["ds_b"]:
            continue
        a, b = r["uid_a"], r["uid_b"]
        ra, rb = role.get(a), role.get(b)
        if {ra, rb} == {"train", "test"}:
            tr, te = (a, b) if ra == "train" else (b, a)
            cand[r["ds_a"]].append((tr, te, float(r["cos"]), int(float(r["inliers"]))))
    sel = [d for d in ("plitterstreet", "rolid", "coxpwd") if cand[d]]
    if not sel:
        raise RuntimeError("tidak ada pasangan train-test resmi terverifikasi")
    rng = np.random.default_rng(1)
    fig, axs = plt.subplots(len(sel), 2, figsize=(TW, 1.75 * len(sel)))
    axs = np.atleast_2d(axs)
    for i, ds in enumerate(sel):
        c = sorted(cand[ds], key=lambda x: x[2])
        tr, te, cs, inl = c[int(rng.integers(len(c) // 4, max(len(c) // 4 + 1, 3 * len(c) // 4)))] if len(c) > 3 else c[0]
        for j, (u, lab) in enumerate([(tr, "official train"), (te, "official test")]):
            im = Image.open(D / "harmonized" / "images" / f"{u}.jpg").convert("RGB")
            im.thumbnail((900, 900))
            ax = axs[i, j]
            ax.imshow(im)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.set_title(f"{NAME[ds]} -- {lab}" + (f"   (cos {cs:.3f}, {inl} inliers)" if j == 1 else ""), loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "fig_audit_pairs.jpg", dpi=300, pil_kwargs={"quality": 88})
    plt.close(fig)
    mp = D / "audit" / "static_mask_rolid.png"
    if mp.exists():
        import shutil
        shutil.copy(mp, FIG / "static_mask_rolid.png")


# ------------------------------------------------------------------ E4 dataset example figure
def e4_examples():
    from PIL import Image, ImageDraw
    M = meta_rows()
    rng = np.random.default_rng(3)
    fig, axs = plt.subplots(2, 3, figsize=(TW, 3.3))
    for ax, ds in zip(axs.ravel(), DS):
        pool = [r for r in M if r["dataset"] == ds and r["split"] == "test" and 2 <= len(label_boxes(r["uid"])) <= 8]
        pool = pool or [r for r in M if r["dataset"] == ds and r["split"] == "test"]
        r = pool[int(rng.integers(len(pool)))]
        im = Image.open(D / "harmonized" / "images" / f"{r['uid']}.jpg").convert("RGB")
        w, h = im.size
        dr = ImageDraw.Draw(im)
        lw = max(2, int(max(w, h) / 300))
        for cx, cy, bw, bh in label_boxes(r["uid"]):
            dr.rectangle([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h],
                         outline="#eb6834", width=lw)
        im.thumbnail((900, 900))
        ax.imshow(im)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"{NAME[ds]} ({VIEW_TXT[VIEW[ds]].lower()})", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "fig_examples.jpg", dpi=300, pil_kwargs={"quality": 88})
    plt.close(fig)


# ------------------------------------------------------------------ E5 cross-dataset matrix
def matrix(model, metric, seeds=None):
    return {(s, t): vals(model, s, t, metric, seeds=seeds) for s in DS for t in DS}


def e5_matrix():
    for metric in ("map50", "map5095"):
        Mv = matrix(BASE, metric)
        rows = []
        for s in DS:
            cells = []
            for t in DS:
                c = pm(Mv[(s, t)])
                cells.append(f"\\textbf{{{c}}}" if s == t else c)
            rows.append([NAME[s]] + cells)
        write_tab(f"tab_matrix_{metric}", ["Train $\\backslash$ test"] + [SHORT[t] for t in DS],
                  rows, align="l" + "c" * len(DS))
        mean = {k: np.mean(v) for k, v in Mv.items() if v}
        diag = [mean[(d, d)] for d in DS if (d, d) in mean]
        off = [mean[(s, t)] for s in DS for t in DS if s != t and (s, t) in mean]
        ratio = [mean[(s, t)] / mean[(t, t)] for s in DS for t in DS if s != t and (s, t) in mean and (t, t) in mean]
        sds = [np.std(v, ddof=1) for v in Mv.values() if len(v) > 1]
        note(f"matrix.{metric}", dict(
            in_domain={d: round(mean[(d, d)], 3) for d in DS if (d, d) in mean},
            in_domain_mean=round(np.mean(diag), 3), cross_mean=round(np.mean(off), 3),
            ratio_median=round(float(np.median(ratio)), 3), ratio_min=round(min(ratio), 3), ratio_max=round(max(ratio), 3),
            n_seeds=sorted({len(v) for v in Mv.values()}), sd_max=round(max(sds), 3) if sds else None,
            sd_median=round(float(np.median(sds)), 3) if sds else None,
            best_source={t: SHORT[max((mean[(s, t)], s) for s in DS if s != t and (s, t) in mean)[1]] for t in DS},
            cells={f"{s}>{t}": round(v, 3) for (s, t), v in mean.items()}))
        if metric != "map50":
            continue
        # gambar: (a) mAP50, (b) rasio transfer
        A = np.array([[mean.get((s, t), np.nan) for t in DS] for s in DS])
        R = np.array([[A[i, j] / A[j, j] for j in range(len(DS))] for i in range(len(DS))])
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list("seqblue", SEQ)
        fig, axs = plt.subplots(1, 2, figsize=(TW, 3.0))
        for ax, X, ttl, vmax in [(axs[0], A, "(a) mAP$_{50}$", 1.0), (axs[1], R, "(b) Transfer ratio (cross / in-domain)", 1.0)]:
            im = ax.imshow(np.clip(X, 0, vmax), cmap=cmap, vmin=0, vmax=vmax)
            ax.grid(False)
            ax.set_xticks(range(len(DS)), [SHORT[d] for d in DS], rotation=35, ha="right")
            ax.set_yticks(range(len(DS)), [SHORT[d] for d in DS])
            ax.set_xlabel("Test dataset"); ax.set_ylabel("Training dataset")
            ax.set_title(ttl, loc="left")
            for sp in ax.spines.values():
                sp.set_visible(False)
            for i in range(len(DS)):
                for j in range(len(DS)):
                    if np.isnan(X[i, j]):
                        continue
                    v = X[i, j]
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.8,
                            color="white" if v > 0.55 else INK, fontweight="bold" if i == j else "normal")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).outline.set_visible(False)
        fig.tight_layout()
        savefig(fig, "fig_matrix")


# ------------------------------------------------------------------ E6 factor analysis
def fit(y, X):
    X1 = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    pred = X1 @ beta
    return beta, 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def pairs_for(model, metric, seeds=None):
    Mv = {k: np.mean(v) for k, v in matrix(model, metric, seeds).items() if v}
    rows = []
    for s in DS:
        for t in DS:
            if s != t and (s, t) in Mv and (t, t) in Mv:
                rows.append(dict(s=s, t=t, R=Mv[(s, t)] / max(Mv[(t, t)], 1e-6),
                                 gap=abs(np.log(SCALE[s] / SCALE[t])), signed=np.log(SCALE[s] / SCALE[t]),
                                 view=int(VIEW[s] == VIEW[t]), region=int(REGION[s] == REGION[t])))
    return rows


def regress(rows, n_perm=10000):
    y = np.log(np.array([r["R"] for r in rows]) + 0.01)
    X = np.column_stack([[r["gap"] for r in rows], [r["view"] for r in rows], [r["region"] for r in rows]])
    beta, r2 = fit(y, X)
    rng = np.random.default_rng(0)
    perm = np.array([fit(rng.permutation(y), X)[0] for _ in range(n_perm)])
    p = [(np.abs(perm[:, i]) >= abs(beta[i])).mean() for i in range(1, 4)]
    return beta, p, r2


def e6_factors():
    rows_tab = []
    specs = [(BASE, "map50", "YOLO11m", "mAP$_{50}$"), (BASE, "map5095", "YOLO11m", "mAP$_{50:95}$")]
    rt = [m for m in models_present() if "rtdetr" in m.lower()]
    if rt:
        specs.append((rt[0], "map50", "RT-DETR-l", "mAP$_{50}$"))
    for model, metric, mname, mtxt in specs:
        P = pairs_for(model, metric)
        if len(P) < 10:
            continue
        beta, p, r2 = regress(P)
        rho, prho = stats.spearmanr([r["gap"] for r in P], [r["R"] for r in P])
        # jackknife per dataset: buang semua pasangan yang melibatkan dataset d
        jk = []
        for d in DS:
            Q = [r for r in P if d not in (r["s"], r["t"])]
            if len(Q) >= 8:
                jk.append(fit(np.log(np.array([r["R"] for r in Q]) + 0.01),
                              np.column_stack([[r["gap"] for r in Q], [r["view"] for r in Q], [r["region"] for r in Q]]))[0])
        jk = np.array(jk)
        rng_ = lambda i: f"[{jk[:, i].min():+.2f}, {jk[:, i].max():+.2f}]" if len(jk) else "--"
        rows_tab.append([f"{mname}, {mtxt}", str(len(P)),
                         f"{beta[1]:+.2f} ({p[0]:.3f})", f"{beta[2]:+.2f} ({p[1]:.3f})", f"{beta[3]:+.2f} ({p[2]:.3f})",
                         f"{r2:.2f}", f"{rho:+.2f} ({prho:.3f})"])
        rows_tab.append(["\\quad jackknife range", "", rng_(1), rng_(2), rng_(3), "", ""])
        note(f"reg.{model}.{metric}", dict(n=len(P), b_gap=round(beta[1], 3), p_gap=p[0], b_view=round(beta[2], 3),
                                           p_view=p[1], b_region=round(beta[3], 3), p_region=p[2], r2=round(r2, 3),
                                           spearman_gap=round(rho, 3), p_spearman=prho,
                                           jk_gap=[round(x, 3) for x in jk[:, 1]] if len(jk) else None,
                                           jk_view=[round(x, 3) for x in jk[:, 2]] if len(jk) else None,
                                           jk_region=[round(x, 3) for x in jk[:, 3]] if len(jk) else None))
        if model == BASE and metric == "map50":
            sg = [r["signed"] for r in P]
            rho2, p2 = stats.spearmanr(sg, [r["R"] for r in P])
            note("reg.signed_scale_spearman", dict(rho=round(rho2, 3), p=p2))
            fig, ax = plt.subplots(figsize=(TW * 0.62, 2.6))
            for v, col, lab in [(1, PAL[0], "same viewpoint"), (0, PAL[1], "different viewpoint")]:
                Q = [r for r in P if r["view"] == v]
                ax.scatter([r["signed"] for r in Q], [r["R"] for r in Q], s=22, color=col, label=lab,
                           edgecolor="white", linewidth=0.8, zorder=3)
            ax.axvline(0, color=INK2, lw=0.8, ls=":")
            ax.set_xlabel("log(training object size / test object size)")
            ax.set_ylabel("Transfer ratio")
            ax.text(0.01, 0.98, "$\\leftarrow$ train smaller", transform=ax.transAxes, fontsize=6.5, color=INK2, va="top")
            ax.text(0.99, 0.98, "train larger $\\rightarrow$", transform=ax.transAxes, fontsize=6.5, color=INK2, va="top", ha="right")
            ax.set_ylim(0, 1.0)
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
            savefig(fig, "fig_factors")
    write_tab("tab_regression", ["Model, metric", "Pairs", "$|\\Delta$scale$|$", "Same viewpoint", "Same region",
                                 "$R^2$", "Spearman (scale)"], rows_tab, align="lcccccc")


# ------------------------------------------------------------------ E7 scale intervention
def e7_scale():
    curve = defaultdict(lambda: defaultdict(list))
    for (m, s, sd, t, it, ie), r in RES.items():
        if m == BASE and it == IMGSZ and not s.startswith("lodo") and sd == SCALE_SEED:
            curve[(s, t)][ie].append(float(r["map50"]))
    rec = []
    for s in DS:
        for t in DS:
            c = curve[(s, t)]
            if s == t or IMGSZ not in c or IMGSZ not in curve[(t, t)] or len(c) < 3:
                continue
            grid = sorted(c)
            pred = IMGSZ * SCALE[s] / SCALE[t]
            near = min(grid, key=lambda g: abs(np.log(g / pred)))
            mc = {g: np.mean(v) for g, v in c.items()}
            base, sm = mc[IMGSZ], mc[near]
            best = max(grid, key=lambda g: mc[g])
            ind = np.mean(curve[(t, t)][IMGSZ])
            gap = ind - base
            rec.append(dict(s=s, t=t, base=base, sm=sm, oracle=mc[best], best=best, ind=ind, near=near, pred=pred,
                            r_sm=(sm - base) / gap if gap > 0.02 else np.nan,
                            r_or=(mc[best] - base) / gap if gap > 0.02 else np.nan))
    groups = [("All pairs", lambda r: True),
              ("Shrink", lambda r: r["pred"] < IMGSZ),
              ("\\quad without RoLID source", lambda r: r["pred"] < IMGSZ and r["s"] != "rolid"),
              ("Enlarge", lambda r: r["pred"] > IMGSZ),
              ("Small-object targets", lambda r: r["t"] in ("bepli", "plitterstreet", "rolid"))]
    rows = []
    for name, f in groups:
        L = [r for r in rec if f(r) and not np.isnan(r["r_sm"])]
        if not L:
            continue
        rows.append([name, str(len(L)), f3(np.mean([r["base"] for r in L])), f3(np.mean([r["sm"] for r in L])),
                     f3(np.mean([r["oracle"] for r in L])), f3(np.mean([r["ind"] for r in L])),
                     f"{100 * np.median([r['r_sm'] for r in L]) + 0.0:+.0f}".replace("-0", "0").replace("+0", "0") if abs(100 * np.median([r['r_sm'] for r in L])) < 0.5 else f"{100 * np.median([r['r_sm'] for r in L]):+.0f}",
                     f"{100 * np.median([r['r_or'] for r in L]):+.0f}" if abs(100 * np.median([r['r_or'] for r in L])) >= 0.5 else "0"])
        note(f"scale.{re.sub(r'[^a-z]+', '_', name.lower()).strip('_')}",
             dict(n=len(L), base=round(np.mean([r['base'] for r in L]), 3), sm=round(np.mean([r['sm'] for r in L]), 3),
                  oracle=round(np.mean([r['oracle'] for r in L]), 3),
                  rec_sm_median=round(float(np.median([r['r_sm'] for r in L])), 3),
                  rec_or_median=round(float(np.median([r['r_or'] for r in L])), 3)))
    write_tab("tab_scale", ["Pair group", "$n$", "@1024", "SM", "Oracle", "In-dom.",
                            "Rec.\\ SM (\\%)", "Rec.\\ oracle (\\%)"], rows, align="lccccccc")
    note("scale.pairs", {f"{r['s']}>{r['t']}": dict(base=round(r["base"], 3), sm=round(r["sm"], 3), near=r["near"],
                                                    oracle=round(r["oracle"], 3), best=r["best"], ind=round(r["ind"], 3))
                         for r in rec})
    note("scale.max_eval_size", max(ie for k in curve.values() for ie in k))
    # gambar: 6 panel per dataset uji
    fig, axs = plt.subplots(2, 3, figsize=(TW, 3.9), sharey=True)
    for ax, t in zip(axs.ravel(), DS):
        for s in DS:
            c = curve[(s, t)]
            if len(c) < 2:
                continue
            g = sorted(c)
            ax.plot(g, [np.mean(c[x]) for x in g], color=COLOR[s], marker=MARK[s], ms=3.5,
                    lw=2.0 if s == t else 1.2, ls="-" if s != t else "--", label=SHORT[s])
            if s != t:
                pred = IMGSZ * SCALE[s] / SCALE[t]
                near = min(g, key=lambda x: abs(np.log(x / pred)))
                ax.scatter([near], [np.mean(c[near])], s=36, facecolor="none", edgecolor=COLOR[s], lw=1.0, zorder=4)
        ax.set_xscale("log", base=2)
        ax.set_xticks([320, 640, 1024, 2048, 3072], ["320", "640", "1k", "2k", "3k"])
        ax.minorticks_off()
        ax.set_title(f"Test: {NAME[t]}", loc="left")
        ax.set_ylim(0, 1)
    for ax in axs[1]:
        ax.set_xlabel("Test image size (px)")
    for ax in axs[:, 0]:
        ax.set_ylabel("mAP$_{50}$")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.03), title="Training dataset (dashed = in-domain; circle = scale-matched size)")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    savefig(fig, "fig_scale_curves")


# ------------------------------------------------------------------ E8 compression
CH = ["L0", "WA1", "WA2", "WA3", "LOW"]
MIN_BASE = 0.05  # pasangan lintas dataset dengan mAP50 asli < 0.05 tidak ditabelkan
CH_TXT = {"L0": "Original", "WA1": "1$\\times$ share", "WA2": "2$\\times$", "WA3": "3$\\times$", "LOW": "Low quality"}


def e8_compression():
    rows = list(csv.DictReader(open(W / "results" / "compression_sim.csv")))
    v = defaultdict(lambda: defaultdict(list))
    for r in rows:
        v[(r["train_ds"], r["test_ds"])][r["chain"]].append((r["seed"], float(r["map50"]), float(r["map5095"])))
    out = []
    rel = {}
    for (s, t), c in sorted(v.items(), key=lambda kv: (kv[0][0] != kv[0][1], SCALE.get(kv[0][1], 0))):
        if "L0" not in c:
            continue
        base = {sd: (a, b) for sd, a, b in c["L0"]}
        if s != t and np.mean([x[0] for x in base.values()]) < MIN_BASE:
            continue  # persentase dari baseline ~0 tidak bermakna
        line = [f"{SHORT[s]} $\\rightarrow$ {SHORT[t]}" if s != t else f"{NAME[t]} (in-domain)", f3(np.mean([x[0] for x in base.values()]))]
        rr = {}
        for ch in CH[1:]:
            d = [100 * (a - base[sd][0]) / base[sd][0] for sd, a, b in c.get(ch, []) if sd in base and base[sd][0] > 0]
            rr[ch] = d
            line.append((f"{np.mean(d):+.1f}" + (f" $\\pm$ {np.std(d, ddof=1):.1f}" if len(d) > 1 else "")) if d else "--")
        rel[(s, t)] = rr
        out.append(line)
        note(f"comp.{s}>{t}", {ch: round(float(np.mean(d)), 2) for ch, d in rr.items() if d})
    write_tab("tab_compression", ["Train $\\rightarrow$ test", "mAP$_{50}$ original"] + [CH_TXT[c] for c in CH[1:]],
              out, align="lcrrrr")
    ind = [(s, t) for (s, t) in rel if s == t]
    if ind:
        fig, ax = plt.subplots(figsize=(TW * 0.62, 2.3))
        for (s, t) in sorted(ind, key=lambda k: -SCALE[k[1]]):
            y = [0.0] + [np.mean(rel[(s, t)][c]) if rel[(s, t)].get(c) else np.nan for c in CH[1:]]
            ax.plot(range(len(CH)), y, color=COLOR[t], marker=MARK[t], ms=4, label=f"{NAME[t]} (median size {SCALE[t]:.3f})")
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks(range(len(CH)), ["Original", "Shared\n1x", "Shared\n2x", "Shared\n3x", "Low\nquality"])
        ax.set_ylabel("Change in mAP$_{50}$ (%)")
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), title="In-domain model")
        savefig(fig, "fig_compression")


def e9_natural():
    p = W / "results" / "compression_natural.md"
    L = [ln for ln in p.read_text().splitlines() if ln.startswith("|") and not ln.startswith("|---")]
    hdr, body = L[0], L[1:]
    rows = []
    for ln in body:
        c = [x.strip() for x in ln.strip("|").split("|")]
        name = c[0].replace("(in-domain)", "(in-domain)").replace("pLitter", "pLitter")
        def ci(x):  # "-0.010 [-0.050, +0.030]" -> 2 desimal di CI
            m_ = re.match(r"([+-]?[\d.]+)\s*\[([+-]?[\d.]+),\s*([+-]?[\d.]+)\]", x)
            return f"{float(m_.group(1)):+.3f} [{float(m_.group(2)):+.2f}, {float(m_.group(3)):+.2f}]" if m_ else x
        rows.append([name.replace("pLitter", "pLitter")] + [ci(x) for x in c[1:]])
    for ln in p.read_text().splitlines():
        if ln.startswith("citra uji"):
            note("natural.counts", ln)
    write_tab("tab_natural", ["Training set", "WA", "Cam.", "$\\Delta$ [95\\% CI]",
                              "WA", "Cam.", "$\\Delta$ [95\\% CI]"], rows, align="lcccccc",
              note_="kolom 2-4 = AP50, kolom 5-7 = AP50:95; WA = foto lewat WhatsApp, Cam. = kamera asli")
    note("natural.rows", rows)


# ------------------------------------------------------------------ E10 scale augmentation
def e10_scaleaug():
    aug = BASE + "_sc09"
    srcs = sorted({k[1] for k in RES if k[0] == aug}, key=lambda d: DS.index(d) if d in DS else 99)
    rows, diffs_cross, diffs_in = [], [], []
    for s in srcs:
        for t in DS:
            a, b = vals(aug, s, t), vals(BASE, s, t)
            if not a or not b:
                continue
            d = np.mean(a) - np.mean(b)
            (diffs_in if s == t else diffs_cross).append(d)
            rows.append([f"{SHORT[s]} $\\rightarrow$ {SHORT[t]}" + (" (in-dom.)" if s == t else ""), pm(b), pm(a), f"{d:+.3f}"])
    if not rows:
        raise RuntimeError("tidak ada run sc09")
    w = stats.wilcoxon(diffs_cross) if len(diffs_cross) >= 5 else None
    note("aug", dict(sources=srcs, n_cross=len(diffs_cross), mean_cross=round(float(np.mean(diffs_cross)), 4),
                     n_improved=int(sum(d > 0 for d in diffs_cross)), wilcoxon_p=float(w.pvalue) if w else None,
                     mean_in=round(float(np.mean(diffs_in)), 4) if diffs_in else None))
    rows.append(["Mean $\\Delta$, cross-dataset", "", "", f"{np.mean(diffs_cross):+.3f}"])
    if w:
        rows.append([f"Wilcoxon signed-rank $p$ ({len(diffs_cross)} pairs)", "", "", f"{w.pvalue:.3f}"])
    write_tab("tab_scaleaug", ["Train $\\rightarrow$ test", "Baseline", "Scale aug.\\ (0.9)", "$\\Delta$"], rows,
              align="lccc", midrules=(len(rows) - (2 if w else 1),))


# ------------------------------------------------------------------ E11 LODO + zero-shot
def zs_rows():
    p = W / "results" / "zeroshot.csv"
    return list(csv.DictReader(open(p))) if p.exists() else []


def e11_lodo():
    zs = {(r["model"], r["prompt"], r["test_ds"]): float(r["map50"]) for r in zs_rows()}
    rows, data = [], {}
    for t in DS:
        ind = vals(BASE, t, t)
        best = max(((np.mean(vals(BASE, s, t)), s) for s in DS if s != t and vals(BASE, s, t)), default=(np.nan, None))
        lodo = vals(BASE, f"lodo-{t}", t)
        g = zs.get(("gdino", "p1", t), np.nan)
        data[t] = dict(ind=ind, best=vals(BASE, best[1], t) if best[1] else [], best_src=best[1], lodo=lodo, gd=g)
        win = "LODO" if lodo and np.mean(lodo) > best[0] else "single"
        rows.append([NAME[t], pm(ind), (pm(vals(BASE, best[1], t)) + f" ({SHORT[best[1]]})") if best[1] else "--",
                     pm(lodo), f3(g),
                     f"{np.mean(lodo) - best[0]:+.3f}" if lodo else "--"])
        note(f"lodo.{t}", dict(in_domain=round(np.mean(ind), 3) if ind else None, best_single=round(best[0], 3),
                               best_src=best[1], lodo=round(np.mean(lodo), 3) if lodo else None,
                               lodo_sd=round(np.std(lodo, ddof=1), 3) if len(lodo) > 1 else None,
                               lodo_seeds=len(lodo), gdino=g, winner=win))
    write_tab("tab_lodo", ["Target", "In-domain", "Best single", "LODO", "G-DINO", "LODO$-$best"],
              rows, align="lccccc")
    fig, ax = plt.subplots(figsize=(TW, 2.3))
    x = np.arange(len(DS))
    w_ = 0.2
    series = [("In-domain (oracle)", "ind", PAL[6]), ("Best single source", "best", PAL[0]),
              ("Leave-one-dataset-out", "lodo", PAL[1]), ("Grounding DINO, zero-shot", "gd", PAL[2])]
    for k, (lab, key, col) in enumerate(series):
        m = [np.mean(data[t][key]) if key != "gd" and data[t][key] else (data[t]["gd"] if key == "gd" else np.nan) for t in DS]
        e = None if key == "gd" else [np.std(data[t][key], ddof=1) if len(data[t][key]) > 1 else 0 for t in DS]
        ax.bar(x + (k - 1.5) * w_, m, w_ * 0.9, color=col, label=lab, yerr=e, error_kw=dict(lw=0.8, capsize=1.5, ecolor=INK2))
    ax.set_xticks(x, [NAME[t] for t in DS])
    ax.set_ylabel("mAP$_{50}$ on target")
    ax.set_ylim(0, 1)
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.28))
    savefig(fig, "fig_lodo_zeroshot")


def e12_zeroshot():
    Z = zs_rows()
    if not Z:
        raise RuntimeError("zeroshot.csv tidak ada")
    MN = {"yoloworld": "YOLO-World v2", "yoloe": "YOLOE", "gdino": "Grounding DINO",
          "gdino_t2": "Grounding DINO, 2$\\times$2 tiles", "gdino_t3": "Grounding DINO, 3$\\times$3 tiles"}
    PR = {"p1": "``litter''", "p2": "synonyms", "p3": "object list"}
    for metric in ("map50", "map5095"):
        tab = defaultdict(dict)
        for r in Z:
            tab[(r["model"], r["prompt"])][r["test_ds"]] = float(r[metric])
        keys = [(m, p) for m in MN for p in PR if (m, p) in tab]
        best = {t: max((tab[k].get(t, -1) for k in keys), default=-1) for t in DS}
        rows = []
        for k in keys:
            rows.append([MN[k[0]], PR[k[1]]] + [("\\textbf{" + f3(tab[k][t]) + "}") if t in tab[k] and tab[k][t] == best[t]
                                                 else f3(tab[k].get(t)) for t in DS])
        mid = [len(rows)]
        sup_best = []
        for t in DS:
            c = [(np.mean(vals(BASE, s, t, metric)), s) for s in DS if s != t and vals(BASE, s, t, metric)]
            sup_best.append(f3(max(c)[0]) if c else "--")
        rows.append(["Supervised YOLO11m", "best other dataset"] + sup_best)
        rows.append(["Supervised YOLO11m", "LODO"] + [f3(np.mean(vals(BASE, f"lodo-{t}", t, metric))) if vals(BASE, f"lodo-{t}", t, metric) else "--" for t in DS])
        rows.append(["Supervised YOLO11m", "in-domain"] + [f3(np.mean(vals(BASE, t, t, metric))) if vals(BASE, t, t, metric) else "--" for t in DS])
        write_tab(f"tab_zeroshot_{metric}", ["Detector", "Prompt"] + [SHORT[t] for t in DS], rows,
                  align="ll" + "c" * len(DS), midrules=tuple(mid))
        note(f"zs.{metric}", {f"{m}|{p}": {t: round(v, 3) for t, v in tab[(m, p)].items()} for (m, p) in keys})


def e13_rtdetr():
    rt = [m for m in models_present() if "rtdetr" in m.lower()]
    if not rt:
        raise RuntimeError("tidak ada run RT-DETR")
    m = rt[0]
    Mv = matrix(m, "map50")
    rows = []
    for s in DS:
        rows.append([NAME[s]] + [("\\textbf{" + pm(Mv[(s, t)]) + "}") if s == t else pm(Mv[(s, t)]) for t in DS])
    write_tab("tab_matrix_rtdetr", ["Train $\\backslash$ test"] + [SHORT[t] for t in DS], rows,
              align="l" + "c" * len(DS))
    seeds = sorted({k[2] for k in RES if k[0] == m})
    Y = {k: np.mean(v) for k, v in matrix(BASE, "map50", seeds=seeds).items() if v}
    Rm = {k: np.mean(v) for k, v in Mv.items() if v}
    common = [k for k in Rm if k in Y]
    off = [k for k in common if k[0] != k[1]]
    rho, p = stats.spearmanr([Y[k] for k in off], [Rm[k] for k in off]) if len(off) > 3 else (np.nan, np.nan)
    note("rtdetr", dict(model=m, seeds=seeds, n_cells=len(common),
                        in_domain={k[0]: round(Rm[k], 3) for k in common if k[0] == k[1]},
                        in_domain_yolo={k[0]: round(Y[k], 3) for k in common if k[0] == k[1]},
                        spearman_offdiag=round(float(rho), 3), p=float(p),
                        cross_mean_rtdetr=round(float(np.mean([Rm[k] for k in off])), 3) if off else None,
                        cross_mean_yolo=round(float(np.mean([Y[k] for k in off])), 3) if off else None))


# ------------------------------------------------------------------ main
EXHIBITS = [e1_datasets, e2_audit, e3_audit_fig, e4_examples, e5_matrix, e6_factors, e7_scale, e8_compression,
            e9_natural, e10_scaleaug, e11_lodo, e12_zeroshot, e13_rtdetr]


def main():
    global RES
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="nama fungsi exhibit, mis. e5_matrix")
    args = ap.parse_args()
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    RES = load_results()
    for d in DS:
        SCALE[d] = train_scale(d)
    note("models_in_results", models_present())
    for fn in EXHIBITS:
        if args.only and fn.__name__ not in args.only:
            continue
        try:
            fn()
            print(f"[ok]    {fn.__name__}", flush=True)
        except Exception as e:  # noqa: BLE001
            LOG.append(f"### {fn.__name__}\n{traceback.format_exc()}")
            print(f"[GAGAL] {fn.__name__}: {e}", flush=True)
    with open(OUT / "summary.md", "w") as fh:
        fh.write("# Angka kunci untuk naskah\n\n")
        for k, v in S.items():
            fh.write(f"- **{k}**: {json.dumps(v, default=float)}\n")
    (OUT / "errors.log").write_text("\n".join(LOG) if LOG else "tidak ada error\n")
    z = W / "paper_assets.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in OUT.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(OUT))
    print(f"\n{len(LOG)} exhibit gagal (lihat errors.log)" if LOG else "\nsemua exhibit OK")
    print(f"-> {z}  ({z.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
