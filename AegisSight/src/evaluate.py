"""
evaluate.py — Evaluate a trained RT-DETR checkpoint.

Reports, per class and overall:
  - mAP@0.50, mAP@0.50:0.95 (COCO-style, via Ultralytics' built-in val())
  - Precision, Recall, F1 at the operating confidence threshold used by the API
  - A confusion matrix (including a "background"/missed row-col) to see which
    classes get confused with which — this is what actually informs the
    failure-case analysis in docs/failure_cases.md, not just the mAP headline.

Usage
-----
    python src/evaluate.py \
        --weights runs/ppe/rtdetr_ppe_v1/weights/best.pt \
        --data data/dataset.yaml --split test \
        --out docs/eval_report.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import RTDETR


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--conf", type=float, default=0.35, help="Confidence threshold — should match DETECT_CONF_THRESHOLD in .env")
    p.add_argument("--iou", type=float, default=0.6, help="NMS IoU threshold")
    p.add_argument("--out", default="docs/eval_report.json")
    args = p.parse_args()

    model = RTDETR(args.weights)

    metrics = model.val(
        data=args.data,
        split=args.split,
        conf=args.conf,
        iou=args.iou,
        plots=True,   # writes PR curves + confusion_matrix.png into the run dir
        save_json=True,
    )

    class_names = metrics.names
    per_class = {}
    for i, name in class_names.items():
        per_class[name] = {
            "AP50": float(metrics.box.ap50[i]) if i < len(metrics.box.ap50) else None,
            "AP50_95": float(metrics.box.ap[i]) if i < len(metrics.box.ap) else None,
            "precision": float(metrics.box.p[i]) if i < len(metrics.box.p) else None,
            "recall": float(metrics.box.r[i]) if i < len(metrics.box.r) else None,
        }

    report = {
        "weights": args.weights,
        "split": args.split,
        "conf_threshold": args.conf,
        "iou_threshold": args.iou,
        "overall": {
            "mAP50": float(metrics.box.map50),
            "mAP50_95": float(metrics.box.map),
            "precision_mean": float(metrics.box.mp),
            "recall_mean": float(metrics.box.mr),
        },
        "per_class": per_class,
        "notes": (
            "mAP50-95 is the stricter metric and what actually matters for a "
            "compliance system, since a loosely-localized hardhat box that just "
            "clears IoU 0.5 is still useful for mAP50 but not for a tight bbox "
            "shown in the API response. Per-class recall for NO-Hardhat / "
            "NO-Safety-Vest specifically should be read alongside class support "
            "counts (see data/README.md) — these are the minority classes and the "
            "ones that matter most operationally, so a high overall mAP dragged up "
            "by the easy Person class is not sufficient on its own."
        ),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report["overall"], indent=2))
    print(f"\nFull report + confusion matrix plot written under the run directory and {out_path}")


if __name__ == "__main__":
    main()
