"""
main.py — FastAPI app exposing:
    POST /detect  — Part A: raw detections for an image
    POST /ask     — Part B: hand-written reasoning layer over the detector
    GET  /health  — trivial liveness probe (bonus: ops hygiene)
"""
from __future__ import annotations

import io
import logging
import os
import time

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image

from api.reasoning import (
    answer_question,
    apply_guardrail,
    classify_intent,
    reason_over_detections,
)
from api.schemas import AskResponse, DetectResponse
from src.inference import PPEDetector

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("ppe_api")

from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _detector
    try:
        _detector = PPEDetector(WEIGHTS_PATH, conf_threshold=CONF_THRESHOLD)
        logger.info(f"Loaded detector weights from {WEIGHTS_PATH} (classes: {_detector.class_names})")
    except FileNotFoundError as e:
        logger.error(str(e))
        _detector = None
    yield

app = FastAPI(
    title="AegisSight Detection & Reasoning API",
    description="RT-DETR-based construction-site PPE compliance detector, plus a hand-written "
                "natural-language reasoning endpoint over its structured output.",
    version="1.0.0",
    lifespan=lifespan,
)

WEIGHTS_PATH = os.environ.get("MODEL_WEIGHTS_PATH", "runs/ppe/rtdetr_ppe_v1/weights/best.pt")
CONF_THRESHOLD = float(os.environ.get("DETECT_CONF_THRESHOLD", "0.35"))

_detector: PPEDetector | None = None

def get_detector() -> PPEDetector:
    if _detector is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model weights not loaded (expected at '{WEIGHTS_PATH}'). Train the model "
                   f"first — see README §6 — then restart the API.",
        )
    return _detector


@app.get("/health")
def health():
    return {
        "status": "ok" if _detector is not None else "degraded",
        "model_loaded": _detector is not None,
        "weights_path": WEIGHTS_PATH,
    }


@app.post("/detect", response_model=DetectResponse)
async def detect(file: UploadFile = File(...), conf_threshold: float | None = Form(default=None)):  # noqa: B008
    detector = get_detector()

    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(status_code=400, detail=f"Unsupported content type: {file.content_type}")

    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Could not decode image file.")

    detections, inference_ms = detector.predict(image, conf_threshold=conf_threshold)
    logger.info(f"/detect {file.filename}: {len(detections)} detections in {inference_ms}ms")

    return DetectResponse(
        image_id=file.filename or "upload",
        width=image.width,
        height=image.height,
        inference_ms=inference_ms,
        detections=detector.as_dicts(detections),
    )


@app.post("/ask", response_model=AskResponse)
async def ask(file: UploadFile = File(...), question: str = Form(...)):  # noqa: B008
    t0 = time.time()

    # --- Stage 1: intent routing (hand-written, see api/reasoning.py) ---
    intent = classify_intent(question)

    detections_out, summary = None, None
    if intent.needs_detector:
        detector = get_detector()
        raw = await file.read()
        try:
            image = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:  # noqa: BLE001
            raise HTTPException(status_code=400, detail="Could not decode image file.")

        detections, _ = detector.predict(image)
        detections_out = detector.as_dicts(detections)
        summary = reason_over_detections(detections_out)

    # --- Stage 3a: guardrail decided in code, before any LLM phrasing ---
    guardrail = apply_guardrail(intent, summary)

    # --- Stage 3b: phrasing (LLM only ever sees the structured summary) ---
    answer_text = answer_question(question, detections_out, intent, guardrail, summary)

    evidence = None
    if guardrail.confidence == "insufficient":
        evidence = {"reason": guardrail.reason_code, "available_classes": detector_classes_or_default()}
    elif summary is not None:
        evidence = {
            "persons_detected": summary.person_count,
            "hardhat_violations": summary.no_hardhat_count,
            "vest_violations": summary.no_vest_count,
            "raw_detections_used": summary.detections_used,
            "mean_confidence": summary.mean_confidence,
        }

    logger.info(
        f"/ask '{question}' -> used_detector={intent.needs_detector} "
        f"confidence={guardrail.confidence} ({round((time.time() - t0) * 1000, 1)}ms)"
    )

    return AskResponse(
        question=question,
        used_detector=intent.needs_detector,
        route_reason=intent.reason,
        answer=answer_text,
        confidence=guardrail.confidence,
        evidence=evidence,
    )


def detector_classes_or_default() -> list[str]:
    if _detector is not None:
        return list(_detector.class_names.values())
    return ["Hardhat", "NO-Hardhat", "Safety-Vest", "NO-Safety-Vest", "Person"]


# Mount static files at the root
# MUST be done after all other routes so that /detect and /ask take precedence
app.mount("/", StaticFiles(directory="static", html=True), name="static")

