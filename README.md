# litter-xdomain: leakage-audited cross-dataset benchmark for litter detection

Code, data splits and audit results for the paper

> T. A. Prasetyo, *Viewpoint and Object Scale, Not Geography, Limit Cross-Dataset Litter Detection: A Leakage-Audited Benchmark*, submitted to *Multimedia Tools and Applications*.

The benchmark harmonizes six public litter datasets (37,277 images) into a single `litter` class, audits their official
splits for near-duplicates and shared recordings, and provides new group-aware 70/15/15 splits. On top of these splits we
train and evaluate detectors across datasets, analyse which dataset properties explain the transfer loss, and test
scale, transmission (messaging-app recompression), multi-source and zero-shot settings.

**No images are redistributed here.** Every dataset must be downloaded from its original source under its own licence;
the scripts below rebuild exactly the same harmonized images and identifiers (`uid`).

## Datasets

| Dataset | Viewpoint | Region | Source | Licence |
|---|---|---|---|---|
| TACO | handheld | Europe | Zenodo record 3587843 | CC BY 4.0 |
| PlastOPol | handheld | global (citizen science) | Zenodo record 5829156 | CC BY 4.0 |
| CoxPWD | handheld | Bangladesh | Mendeley Data bdzg4tjy63 v3 | CC BY 4.0 |
| BePLi v1 | handheld | Japan | SEANOE 92297 | CC BY-NC-SA 4.0 |
| pLitterStreet | vehicle | Thailand, Vietnam, Sri Lanka | Zenodo record 8288500 | CC BY 4.0 |
| RoLID-11K | vehicle | United Kingdom | authors' Google Drive (see paper) | Apache-2.0 |

Please cite the original dataset papers listed in our manuscript when you use them.

## Repository layout

```
01_download.py         download all datasets (checksums verified)
02_inventory.py        extract archives, inventory annotations and image sizes
03_probe.py            sanity checks: EXIF orientation, box statistics
04_harmonize.py        convert to single-class YOLO format, assign source groups
05_audit_split.py      DINOv2 + ORB/RANSAC near-duplicate audit, official-split leakage, new group splits
06_viz_pairs.py        mosaics of verified duplicate pairs
07_train_eval.py       train / evaluate YOLO11m and RT-DETR-l (single dataset or leave-one-dataset-out)
08_analyze.py          cross-dataset matrix and factor regression
09_scale_curves.py     multi-resolution evaluation and scale-matching analysis
10_compression.py      natural WhatsApp experiment and simulated transmission chains
11_zeroshot.py         YOLO-World v2, YOLOE and Grounding DINO zero-shot evaluation (with tiling)
12_paper_assets.py     every table (LaTeX) and figure (PDF) in the paper, from the result files
13_prepare_release.py  export splits, audit and results into release_data/
release_data/
  splits/<dataset>_{train,val,test}.txt   uid lists of the new group-aware splits
  audit/meta_split.csv                    per-image metadata, group and split assignment
  audit/verified_duplicates.csv           all verified near-duplicate pairs
  audit/audit_report.txt                  leakage of the official splits
  results/                                raw evaluation results behind every table
```

## Reproducing the paper

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python 01_download.py && python 02_inventory.py && python 04_harmonize.py
python 05_audit_split.py                    # or use release_data/splits directly
python 07_train_eval.py prep
python 07_train_eval.py train --ds taco --seeds 0 1 2 --device 0      # repeat for each dataset
python 07_train_eval.py train --lodo --ds taco --seeds 0 1 2 --device 0
python 08_analyze.py && python 09_scale_curves.py
python 10_compression.py natural && python 10_compression.py simulate
python 11_zeroshot.py --models yoloworld yoloe gdino gdino_t2 gdino_t3 --device 0
python 12_paper_assets.py
```

Data are expected in `~/data/litter` and outputs are written to `~/litter-xdomain/work`; both can be changed with the
environment variables `LITTER_DATA` and `LITTER_WORK` in `12_paper_assets.py`. Experiments in the paper were run on
NVIDIA B200 GPUs with Ultralytics 8.4.95 and Hugging Face Transformers 4.57.6.

## Licence

Code: MIT (see `LICENSE`). Split lists and audit results: CC BY 4.0. The images and annotations remain under the
licences of their original datasets.

## Citation

A citation entry will be added once the paper is published.
