"""
# ============================================================
# What does YOLO do in this prototype?
# ============================================================
#
# YOLO answers:
#     "Where is visible damage, and what type does it resemble?"
#
# Each detection contains a rectangular bounding box, nominal damage class, and
# confidence. Confidence means certainty about that detection; it is NOT damage
# severity and must never be converted into a severity grade.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Ultralytics normally creates a settings directory in the user's profile on
# first import. Keep all prototype configuration inside this project instead.
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / "Ultralytics"))

from ultralytics import YOLO


SELECTED_YOLO_CHECKPOINT = PROJECT_ROOT / "runs" / "EXP-002_yolo11n_ontology_corrected" / "weights" / "best.pt"
EXPECTED_CLASS_NAMES = ["minor_damage", "compressed", "hole", "wet"]

# 0.25 matches the confidence used for the completed detector's descriptive
# error analysis and is a clear prototype display threshold. It is not a
# severity threshold or a newly optimized production operating point.
PROTOTYPE_YOLO_CONFIDENCE = 0.25


def load_frozen_yolo():
    """Load the selected completed detector without training it."""
    if not SELECTED_YOLO_CHECKPOINT.is_file():
        raise FileNotFoundError(f"Missing selected YOLO checkpoint: {SELECTED_YOLO_CHECKPOINT}")
    model = YOLO(str(SELECTED_YOLO_CHECKPOINT), task="detect")
    names = [model.names[index] for index in sorted(model.names)]
    if names != EXPECTED_CLASS_NAMES:
        raise RuntimeError(f"Unexpected YOLO class mapping: {names}")
    return model


def predict_damage_regions(
    image: Image.Image,
    model,
    device: torch.device,
    confidence_threshold: float = PROTOTYPE_YOLO_CONFIDENCE,
) -> dict:
    """Run YOLO independently and return boxes, classes, confidence, and timing."""
    rgb_image = image.convert("RGB")
    overall_started = time.perf_counter()

    # PIL images use RGB. Ultralytics handles its required internal conversion,
    # resize, normalization, inference, and non-maximum suppression.
    results = model.predict(
        source=rgb_image,
        imgsz=640,
        conf=confidence_threshold,
        device=0 if device.type == "cuda" else "cpu",
        verbose=False,
    )
    total_yolo_ms = (time.perf_counter() - overall_started) * 1000
    if len(results) != 1:
        raise RuntimeError(f"Expected one YOLO result, received {len(results)}")

    result = results[0]
    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            coordinates = [float(value) for value in box.xyxy[0].cpu().tolist()]
            class_id = int(box.cls[0].cpu().item())
            confidence = float(box.conf[0].cpu().item())
            detections.append({
                # xyxy means left, top, right, bottom in original-image pixels.
                "bounding_box_xyxy": coordinates,
                "class_id": class_id,
                "class_name": result.names[class_id],
                "confidence": confidence,
            })

    speed = result.speed or {}
    return {
        "detections": detections,
        "detection_count": len(detections),
        "confidence_threshold": confidence_threshold,
        "preprocess_ms": float(speed.get("preprocess", 0.0)),
        "inference_ms": float(speed.get("inference", 0.0)),
        "postprocess_ms": float(speed.get("postprocess", 0.0)),
        "end_to_end_yolo_call_ms": total_yolo_ms,
    }
