"""
inference.py — Thin wrapper around the trained RT-DETR checkpoint, shared by
both API endpoints. Kept separate from api/ so it can also be imported by
evaluate.py-style scripts or a notebook without pulling in FastAPI.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image
from ultralytics import RTDETR


@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox_xyxy: list[float]  # [x1, y1, x2, y2] in pixel coordinates


class PPEDetector:
    """Loads once at API startup; `predict()` is called per-request."""

    def __init__(self, weights_path: str, conf_threshold: float = 0.35, device: str | None = None):
        if not Path(weights_path).exists():
            raise FileNotFoundError(
                f"Model weights not found at '{weights_path}'. Train the model first "
                f"(see README §6) or point MODEL_WEIGHTS_PATH at an existing checkpoint."
            )
        self.model = RTDETR(weights_path)
        self.conf_threshold = conf_threshold
        self.device = device
        self.class_names = self.model.names  # {0: "Hardhat", 1: "NO-Hardhat", ...}

    def predict(self, image: Image.Image, conf_threshold: float | None = None) -> tuple[list[Detection], float]:
        conf = conf_threshold if conf_threshold is not None else self.conf_threshold
        t0 = time.time()
        results = self.model.predict(image, conf=conf, device=self.device, verbose=False)
        inference_ms = (time.time() - t0) * 1000

        detections: list[Detection] = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls.item())
                detections.append(
                    Detection(
                        class_name=self.class_names[cls_id],
                        confidence=round(float(box.conf.item()), 4),
                        bbox_xyxy=[round(v, 1) for v in box.xyxy[0].tolist()],
                    )
                )
        return detections, round(inference_ms, 1)

    def as_dicts(self, detections: list[Detection]) -> list[dict]:
        return [asdict(d) for d in detections]
