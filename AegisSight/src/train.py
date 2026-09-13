"""
train.py — Fine-tune RT-DETR on the PPE-compliance dataset.

Uses Ultralytics' RT-DETR implementation (a faithful, widely-used implementation
of the RT-DETR architecture from "DETRs Beat YOLOs on Real-time Object Detection",
Zhao et al. 2023 — explicitly allowed by the problem statement's "any faithful
implementation/library is fine" clause).

Every run writes its exact config (hyperparameters, git commit, environment,
hardware, wall-clock time) to <project>/<name>/run_manifest.json so the run is
reproducible from this repo alone, per the "reproducibility is mandatory" hard
constraint.

Usage
-----
    python src/train.py \
        --data data/dataset.yaml \
        --model rtdetr-l.pt \
        --epochs 80 --imgsz 640 --batch 16 \
        --project runs/ppe --name rtdetr_ppe_v1
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import torch
from ultralytics import RTDETR


def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown (not a git repo / git unavailable)"


def get_hardware_info() -> dict:
    info = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
        info["cuda_version"] = torch.version.cuda
    return info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to dataset.yaml")
    p.add_argument("--model", default="rtdetr-l.pt", help="RT-DETR variant / checkpoint to start from")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr0", type=float, default=1e-4, help="Initial LR (RT-DETR uses AdamW; lower than YOLO defaults)")
    p.add_argument("--optimizer", default="AdamW")
    p.add_argument("--patience", type=int, default=25, help="Early-stopping patience (epochs)")
    p.add_argument("--freeze", type=int, default=0, help="Number of backbone layers to freeze (0 = full fine-tune)")
    p.add_argument("--project", default="runs/ppe")
    p.add_argument("--name", default="rtdetr_ppe_v1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    manifest = {
        "hyperparameters": vars(args),
        "hardware": get_hardware_info(),
        "git_commit": get_git_commit(),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    model = RTDETR(args.model)  # loads COCO-pretrained RT-DETR weights as the starting point

    t0 = time.time()
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        optimizer=args.optimizer,
        lr0=args.lr0,
        patience=args.patience,
        freeze=args.freeze,
        seed=args.seed,
        project=args.project,
        name=args.name,
        resume=args.resume,
        # Fine-tuning (not training from scratch) so a comparatively conservative
        # augmentation policy is used — RT-DETR's default mosaic/mixup schedule is
        # tuned for from-scratch COCO training and tends to over-augment a ~5k-image
        # fine-tuning run on a narrower domain.
        mosaic=0.3,
        mixup=0.0,
        degrees=5.0,
        translate=0.1,
        scale=0.3,
        fliplr=0.5,
        flipud=0.0,
        plots=True,
        val=True,
    )
    manifest["train_wall_clock_seconds"] = round(time.time() - t0, 1)
    manifest["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    run_dir = Path(args.project) / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nTraining complete. Manifest written to {run_dir / 'run_manifest.json'}")
    print(f"Best weights: {run_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
