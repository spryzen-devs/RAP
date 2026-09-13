"""
test_api.py — Smoke tests for the API.

Two kinds of tests here, deliberately separated:

1. Unit tests for the reasoning layer (api/reasoning.py) — these need NO
   model weights, NO GPU, and NO network, so they run in any environment
   (including CI). They directly exercise the "how does /ask decide to call
   the detector vs not, and what does an insufficient-info case look like"
   requirement from the deliverables list.
2. End-to-end API tests via TestClient — these DO need real trained weights
   at MODEL_WEIGHTS_PATH and are skipped automatically if the file isn't
   there, so `pytest` still passes in a fresh clone before anyone has
   trained anything.
"""
import os
from pathlib import Path

import pytest

from api.reasoning import apply_guardrail, classify_intent, reason_over_detections

# ---------------------------------------------------------------------------
# 1. Reasoning-layer unit tests (no model / no GPU required)
# ---------------------------------------------------------------------------

def test_routes_general_knowledge_question_without_detector():
    intent = classify_intent("What color is a hardhat usually?")
    assert intent.needs_detector is False


def test_routes_compliance_question_to_detector():
    intent = classify_intent("Is anyone not wearing a helmet in this image?")
    assert intent.needs_detector is True
    assert intent.out_of_taxonomy is False
    assert "hardhat" in intent.target_concepts


def test_out_of_taxonomy_question_flagged_even_with_covered_concept_present():
    # "person" is a covered concept, but the actual attribute asked about
    # (boots) isn't — out_of_taxonomy must win over the incidental match.
    intent = classify_intent("Is the person on the far left wearing steel-toe boots?")
    assert intent.out_of_taxonomy is True


def test_general_knowledge_not_pulled_into_detector_path_by_a_bare_keyword():
    # "hardhat" is a covered concept, but "usually" + no image reference
    # marks this as textbook knowledge, not a question about an uploaded image.
    intent = classify_intent("What color is a hardhat usually?")
    assert intent.needs_detector is False


def test_guardrail_insufficient_on_out_of_taxonomy():
    intent = classify_intent("What brand of boots is the worker wearing?")
    result = apply_guardrail(intent, summary=None)
    assert result.confidence == "insufficient"
    assert result.reason_code == "no_matching_class_in_taxonomy"


def test_guardrail_insufficient_on_zero_detections():
    intent = classify_intent("How many people are wearing hardhats?")
    summary = reason_over_detections([])
    result = apply_guardrail(intent, summary)
    assert result.confidence == "insufficient"
    assert result.reason_code == "zero_detections_returned"


def test_guardrail_high_confidence_with_good_detections():
    intent = classify_intent("How many people are wearing hardhats?")
    detections = [
        {"class_name": "Person", "confidence": 0.95, "bbox_xyxy": [0, 0, 10, 10]},
        {"class_name": "Hardhat", "confidence": 0.9, "bbox_xyxy": [0, 0, 10, 10]},
    ]
    summary = reason_over_detections(detections)
    result = apply_guardrail(intent, summary)
    assert result.confidence == "high"


def test_guardrail_insufficient_on_low_confidence_detections():
    intent = classify_intent("Is anyone not wearing a hardhat?")
    detections = [{"class_name": "NO-Hardhat", "confidence": 0.15, "bbox_xyxy": [0, 0, 10, 10]}]
    summary = reason_over_detections(detections)
    result = apply_guardrail(intent, summary)
    assert result.confidence == "insufficient"
    assert result.reason_code == "detections_below_confidence_threshold"


# ---------------------------------------------------------------------------
# 2. End-to-end API tests — only run if real weights exist
# ---------------------------------------------------------------------------

import importlib.util

WEIGHTS_PATH = os.environ.get("MODEL_WEIGHTS_PATH", "runs/ppe/rtdetr_ppe_v1/weights/best.pt")
SAMPLE_IMAGE = Path(__file__).parent / "sample_requests" / "site_photo_01.jpg"

has_ultralytics = importlib.util.find_spec("ultralytics") is not None
has_weights = Path(WEIGHTS_PATH).exists()

pytestmark_e2e = pytest.mark.skipif(
    not (has_weights and has_ultralytics),
    reason="Requires both trained weights and ultralytics to run e2e tests.",
)


@pytestmark_e2e
def test_detect_endpoint_returns_detections():
    from fastapi.testclient import TestClient

    from api.main import app

    with TestClient(app) as client:
        with open(SAMPLE_IMAGE, "rb") as f:
            resp = client.post("/detect", files={"file": ("site_photo_01.jpg", f, "image/jpeg")})
        assert resp.status_code == 200
        body = resp.json()
        assert "detections" in body


@pytestmark_e2e
def test_ask_endpoint_end_to_end():
    from fastapi.testclient import TestClient

    from api.main import app

    with TestClient(app) as client:
        with open(SAMPLE_IMAGE, "rb") as f:
            resp = client.post(
                "/ask",
                files={"file": ("site_photo_01.jpg", f, "image/jpeg")},
                data={"question": "Is anyone not wearing a helmet in this image?"},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["used_detector"] is True
        assert body["confidence"] in ("high", "medium", "insufficient")
