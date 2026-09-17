"""Generic parcel localization used as the first stage for phone images."""

from __future__ import annotations

import os
import time
from pathlib import Path

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / "Ultralytics"))
from ultralytics import YOLO

GENERIC_PARCEL_CHECKPOINT = PROJECT_ROOT / "runs" / "PARCEL-DET-001_yolo11n_v3" / "weights" / "best.pt"
EXPECTED_CLASS_NAMES = ["Boxes"]
GENERIC_PARCEL_CONF = 0.25
PARCEL_CROP_PADDING = 0.04


def load_parcel_detector(checkpoint_path: Path = GENERIC_PARCEL_CHECKPOINT):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Generic parcel detector weights are missing: {checkpoint_path}")
    model = YOLO(str(checkpoint_path), task="detect")
    names = [model.names[index] for index in sorted(model.names)]
    if names != EXPECTED_CLASS_NAMES:
        raise RuntimeError(f"Unexpected generic parcel classes {names}; expected {EXPECTED_CLASS_NAMES}.")
    return model


def detect_parcels(image: Image.Image, model, device: torch.device,
                   confidence_threshold: float = GENERIC_PARCEL_CONF) -> dict:
    started = time.perf_counter()
    results = model.predict(source=image.convert("RGB"), imgsz=320, conf=confidence_threshold,
                            device=0 if device.type == "cuda" else "cpu", verbose=False)
    elapsed_ms = (time.perf_counter() - started) * 1000
    if len(results) != 1:
        raise RuntimeError(f"Expected one generic parcel result, received {len(results)}")
    detections = []
    if results[0].boxes is not None:
        for box in results[0].boxes:
            detections.append({
                "parcel_bbox": [float(value) for value in box.xyxy[0].cpu().tolist()],
                "parcel_confidence": float(box.conf[0].cpu().item()),
            })
    return {"detections": detections, "detection_count": len(detections),
            "confidence_threshold": confidence_threshold, "inference_ms": elapsed_ms}


def crop_parcel(image: Image.Image, bbox: list[float],
                padding_fraction: float = PARCEL_CROP_PADDING) -> tuple[Image.Image, list[int]]:
    if len(bbox) != 4 or padding_fraction < 0:
        raise ValueError(f"Invalid parcel bounding box: {bbox}")
    width, height = image.size
    left, top, right, bottom = map(float, bbox)
    if right <= left or bottom <= top:
        raise ValueError(f"Invalid or empty parcel bounding box: {bbox}")
    pad_x, pad_y = (right - left) * padding_fraction, (bottom - top) * padding_fraction
    clamped = [max(0, int(left - pad_x)), max(0, int(top - pad_y)),
               min(width, int(right + pad_x + 0.999)), min(height, int(bottom + pad_y + 0.999))]
    if clamped[2] <= clamped[0] or clamped[3] <= clamped[1]:
        raise ValueError(f"Parcel crop is empty after clamping: {bbox}")
    return image.crop(clamped), clamped
