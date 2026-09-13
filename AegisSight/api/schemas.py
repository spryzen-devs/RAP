from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class DetectionOut(BaseModel):
    class_name: str
    confidence: float
    bbox_xyxy: list[float]


class DetectResponse(BaseModel):
    image_id: str
    width: int
    height: int
    inference_ms: float
    detections: list[DetectionOut]


class AskResponse(BaseModel):
    question: str
    used_detector: bool
    route_reason: str
    answer: str
    confidence: Literal["high", "medium", "insufficient"]
    evidence: dict | None = Field(
        default=None,
        description="Structured detection evidence the answer was grounded in, or a reason "
                    "code when confidence == 'insufficient'.",
    )
