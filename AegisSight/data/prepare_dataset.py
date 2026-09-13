"""
prepare_dataset.py
-------------------
Fetches the base Construction Site Safety dataset, filters/remaps it to our
5-class taxonomy, merges any self-labeled supplementary images, deduplicates
near-identical frames, and writes a grouped 70/15/15 train/val/test split in
YOLO format under `--out`.

Usage
-----
    # via Roboflow API
    python prepare_dataset.py --roboflow-key $ROBOFLOW_API_KEY --out data/processed

    # from a manually downloaded COCO-format export
    python prepare_dataset.py --local-coco-dir /path/to/export --out data/processed

    # just print split/class statistics for an already-prepared dataset
    python prepare_dataset.py --report-only --out data/processed

This script is intentionally dependency-light and readable end-to-end — every
step it takes (dedup, grouping, remap) is documented in data/README.md so the
split strategy can be reproduced and defended verbally.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Our target taxonomy. Anything from the source dataset not in this map is
# dropped (see data/README.md "Why this dataset / domain" for the rationale).
# ---------------------------------------------------------------------------
CLASS_REMAP = {
    "Hardhat": "Hardhat",
    "NO-Hardhat": "NO-Hardhat",
    "Safety Vest": "Safety-Vest",
    "Safety-Vest": "Safety-Vest",
    "NO-Safety Vest": "NO-Safety-Vest",
    "NO-Safety-Vest": "NO-Safety-Vest",
    "Person": "Person",
}
FINAL_CLASSES = ["Hardhat", "NO-Hardhat", "Safety-Vest", "NO-Safety-Vest", "Person"]
CLASS_TO_ID = {name: i for i, name in enumerate(FINAL_CLASSES)}

SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
RANDOM_SEED = 42


def download_from_roboflow(api_key: str, workspace: str, project: str, version: int, dst: Path) -> Path:
    """Pull the dataset via the Roboflow SDK in COCO format."""
    from roboflow import (
        Roboflow,  # imported lazily so the script runs without it installed
    )

    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    dataset = proj.version(version).download("coco", location=str(dst))
    return Path(dataset.location)


def phash(image_path: Path, hash_size: int = 8) -> str:
    """Cheap perceptual hash (avg-hash) used only to group near-duplicate frames
    before splitting — not a security hash, doesn't need to be exact."""
    from PIL import Image

    img = Image.open(image_path).convert("L").resize((hash_size, hash_size))
    pixels = list(img.getdata())
    avg = sum(pixels) / len(pixels)
    bits = "".join("1" if p > avg else "0" for p in pixels)
    return hashlib.md5(bits.encode()).hexdigest()


def load_coco_annotations(coco_dir: Path) -> dict:
    """Loads a COCO-format {images, annotations, categories} dict, tolerant of
    Roboflow's per-split _annotations.coco.json layout."""
    merged = {"images": [], "annotations": [], "categories": []}
    seen_cat_names = set()
    img_id_offset, ann_id_offset = 0, 0

    for ann_file in sorted(coco_dir.rglob("_annotations.coco.json")):
        with open(ann_file) as f:
            data = json.load(f)
        img_dir = ann_file.parent

        for cat in data["categories"]:
            if cat["name"] not in seen_cat_names:
                merged["categories"].append(cat)
                seen_cat_names.add(cat["name"])

        for img in data["images"]:
            img = dict(img)
            img["_abs_path"] = str(img_dir / img["file_name"])
            img["id"] += img_id_offset
            merged["images"].append(img)

        for ann in data["annotations"]:
            ann = dict(ann)
            ann["id"] += ann_id_offset
            ann["image_id"] += img_id_offset
            merged["annotations"].append(ann)

        img_id_offset = max((i["id"] for i in merged["images"]), default=0) + 1
        ann_id_offset = max((a["id"] for a in merged["annotations"]), default=0) + 1

    return merged


def filter_and_remap(coco: dict) -> dict:
    """Drop annotations for classes outside CLASS_REMAP; remap category ids to
    our 5-class taxonomy; drop images left with zero annotations."""
    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    kept_anns_by_img = defaultdict(list)

    for ann in coco["annotations"]:
        src_name = cat_id_to_name.get(ann["category_id"])
        mapped = CLASS_REMAP.get(src_name)
        if mapped is None:
            continue
        ann = dict(ann)
        ann["category_id"] = CLASS_TO_ID[mapped]
        kept_anns_by_img[ann["image_id"]].append(ann)

    images = [img for img in coco["images"] if kept_anns_by_img.get(img["id"])]
    return {"images": images, "anns_by_image": kept_anns_by_img}


def group_by_phash(images: list) -> dict[str, list]:
    groups = defaultdict(list)
    for img in images:
        try:
            h = phash(Path(img["_abs_path"]))
        except Exception:  # noqa: BLE001
            h = img["file_name"]  # fall back: treat as its own group
        groups[h].append(img)
    return groups


def split_groups(groups: dict[str, list], seed: int = RANDOM_SEED) -> dict[str, list]:
    keys = list(groups.keys())
    random.Random(seed).shuffle(keys)

    n = len(keys)
    n_train = int(n * SPLIT_RATIOS["train"])
    n_val = int(n * SPLIT_RATIOS["val"])

    split_keys = {
        "train": keys[:n_train],
        "val": keys[n_train:n_train + n_val],
        "test": keys[n_train + n_val:],
    }
    return {split: [img for k in ks for img in groups[k]] for split, ks in split_keys.items()}


def write_yolo_split(split_name: str, images: list, anns_by_image: dict, out_root: Path) -> None:
    img_out = out_root / split_name / "images"
    lbl_out = out_root / split_name / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    for img in images:
        src = Path(img["_abs_path"])
        if not src.exists():
            continue
        dst = img_out / src.name
        shutil.copy2(src, dst)

        w, h = img["width"], img["height"]
        lines = []
        for ann in anns_by_image[img["id"]]:
            x, y, bw, bh = ann["bbox"]  # COCO xywh, absolute pixels
            cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
            nw, nh = bw / w, bh / h
            lines.append(f"{ann['category_id']} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        (lbl_out / f"{src.stem}.txt").write_text("\n".join(lines))


def report_stats(out_root: Path) -> None:
    print(f"{'split':<8}{'images':<10}{'boxes':<10}" + "".join(f"{c:<16}" for c in FINAL_CLASSES))
    for split in ("train", "val", "test"):
        lbl_dir = out_root / split / "labels"
        if not lbl_dir.exists():
            continue
        counts = defaultdict(int)
        n_images, n_boxes = 0, 0
        for lbl_file in lbl_dir.glob("*.txt"):
            n_images += 1
            for line in lbl_file.read_text().splitlines():
                if not line.strip():
                    continue
                cls_id = int(line.split()[0])
                counts[FINAL_CLASSES[cls_id]] += 1
                n_boxes += 1
        row = f"{split:<8}{n_images:<10}{n_boxes:<10}"
        row += "".join(f"{counts.get(c, 0):<16}" for c in FINAL_CLASSES)
        print(row)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--roboflow-key", default=None)
    p.add_argument("--workspace", default="roboflow-universe-projects")
    p.add_argument("--project", default="construction-site-safety")
    p.add_argument("--version", type=int, default=27)
    p.add_argument("--local-coco-dir", default=None, help="Manually downloaded COCO export dir")
    p.add_argument("--supplementary-dir", default="data/supplementary_raw",
                    help="Optional extra COCO-format data to merge in before splitting")
    p.add_argument("--out", required=True)
    p.add_argument("--report-only", action="store_true")
    args = p.parse_args()

    out_root = Path(args.out)

    if args.report_only:
        report_stats(out_root)
        return

    if args.local_coco_dir:
        coco_dir = Path(args.local_coco_dir)
    elif args.roboflow_key:
        coco_dir = download_from_roboflow(args.roboflow_key, args.workspace, args.project, args.version,
                                           Path("data/raw"))
    else:
        raise SystemExit("Provide --roboflow-key or --local-coco-dir")

    coco = load_coco_annotations(coco_dir)

    supp_dir = Path(args.supplementary_dir)
    if supp_dir.exists():
        supp_coco = load_coco_annotations(supp_dir)
        id_off = max((i["id"] for i in coco["images"]), default=0) + 1
        ann_off = max((a["id"] for a in coco["annotations"]), default=0) + 1
        for img in supp_coco["images"]:
            img["id"] += id_off
        for ann in supp_coco["annotations"]:
            ann["id"] += ann_off
            ann["image_id"] += id_off
        coco["images"] += supp_coco["images"]
        coco["annotations"] += supp_coco["annotations"]
        print(f"Merged {len(supp_coco['images'])} supplementary images.")

    filtered = filter_and_remap(coco)
    print(f"{len(filtered['images'])} images retain at least one target-class annotation "
          f"(from {len(coco['images'])} total).")

    groups = group_by_phash(filtered["images"])
    print(f"Grouped into {len(groups)} near-duplicate clusters "
          f"(collapsed from {len(filtered['images'])} images).")

    splits = split_groups(groups)

    if out_root.exists():
        shutil.rmtree(out_root)
    for split_name, images in splits.items():
        write_yolo_split(split_name, images, filtered["anns_by_image"], out_root)

    print("\nDone. Split summary:")
    report_stats(out_root)


if __name__ == "__main__":
    main()
