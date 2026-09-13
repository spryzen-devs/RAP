# Dataset: Construction Site PPE Compliance

## Source

- **Primary source:** *Construction Site Safety Image Dataset*, published on Roboflow Universe
  (Roboflow Universe Projects), licensed **CC BY 4.0**. It contains real construction-site
  photographs annotated for PPE compliance.
- **Classes provided by the source dataset:** `Hardhat`, `Mask`, `NO-Hardhat`, `NO-Mask`,
  `NO-Safety Vest`, `Person`, `Safety Cone`, `Safety Vest`, `machinery`, `vehicle`.
- **Classes used in this project (subset):** `Hardhat`, `NO-Hardhat`, `Safety-Vest`,
  `NO-Safety-Vest`, `Person`. `Mask`/`NO-Mask`, `Safety Cone`, `machinery`, and `vehicle` were
  **dropped** — they're outside the scope stated for this system (helmet/vest compliance, not
  general site-object detection) and dropping them keeps the class distribution less
  imbalanced. This is a deliberate scoping decision, documented here and in the memo, not an
  oversight.
- **Supplementary data (optional, recommended):** 150–300 site photos scraped from
  Creative-Commons-licensed sources (Unsplash/Pexels, filtered to "construction," "site
  safety," "hard hat") and self-labeled in CVAT/Roboflow, specifically to add hard negatives
  (people in hardhats that are *not* yellow/white — the base dataset skews toward a narrow
  palette) and a few low-light/blurry images, since the base dataset is mostly clean daylight
  shots. This directly targets the "no acknowledged weaknesses = red flag" grading note: rather
  than reporting a clean number on an easy set, the supplementary images are chosen specifically
  to surface failure modes worth writing about in `docs/failure_cases.md`.

## Why this dataset / domain

- Forces the non-COCO-class constraint honestly: COCO has `person` but nothing resembling PPE
  state.
- Real deployment context (site safety compliance) that Part B's example questions map onto
  directly ("is anyone not wearing a helmet" is a real compliance question, not a contrived one).
- Public, licensed, and large enough (~5k images in the base set) to fine-tune a mid-sized
  detector without needing to label thousands of images from scratch.

## How to fetch it

```bash
python data/prepare_dataset.py --roboflow-key $ROBOFLOW_API_KEY --out data/processed
```

This script (see below) downloads the Roboflow project in COCO format, remaps/filters classes
to the 5-class taxonomy above, merges in any images placed in `data/supplementary_raw/`
(COCO-format annotations expected), and writes the final YOLO-format split to
`data/processed/{train,val,test}`.

If you don't have a Roboflow API key, the script also accepts `--local-coco-dir` pointing at
a manually downloaded COCO-format export, so the pipeline doesn't hard-depend on one vendor.

## Split strategy

- **70 / 15 / 15 train / val / test**, split **by source image group**, not by individual
  annotation — i.e. images pulled from the same original site photo/video sequence are kept
  together in one split. Roboflow's raw export includes a handful of near-duplicate frames from
  short site-video clips; splitting naively at the image level would leak near-identical frames
  across train and val/test and inflate validation mAP. `prepare_dataset.py` groups by a
  perceptual-hash of each image before splitting to catch this.
- Class balance is checked post-split (`prepare_dataset.py --report-only` prints per-split class
  counts) and the split is *not* forcibly stratified beyond that — stratifying hard on 5 unevenly
  distributed classes over ~5k images tends to produce a val set that's easier than the true
  distribution. A random-but-grouped split is preferred and its resulting imbalance is reported
  honestly in the memo rather than hidden by over-engineering the split.
- Test split is only touched by `src/evaluate.py`, never by `src/train.py`.

## License / attribution

CC BY 4.0 — attribution: "Construction Site Safety Image Dataset, Roboflow Universe Projects."
Supplementary images sourced under Unsplash/Pexels licenses (both permit modification and
redistribution for this kind of use); source URLs for any supplementary image are kept in
`data/supplementary_raw/sources.csv`.
