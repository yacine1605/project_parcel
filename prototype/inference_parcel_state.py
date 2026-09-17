"""Open/closed image classification applied to generic parcel crops."""

from __future__ import annotations

import os
import time
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARCEL_STATE_CHECKPOINT = PROJECT_ROOT / "models" / "parcel_state_classifier" / "best.pt"
EXPECTED_STATE_CLASS_TO_IDX = {"closed_box": 0, "open_box": 1}
PARCEL_STATE_OPEN_THRESHOLD_ENV = "PARCEL_STATE_OPEN_THRESHOLD"


def load_parcel_state_classifier(checkpoint_path: Path = PARCEL_STATE_CHECKPOINT) -> dict:
    """Load and validate the crop classifier once during service initialization."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "Open/closed crop-classifier weights are missing. Expected the Colab-exported "
            f"checkpoint at: {checkpoint_path}"
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != "mobilenet_v3_small":
        raise RuntimeError(f"Unsupported parcel-state architecture: {checkpoint.get('architecture')!r}")
    if checkpoint.get("class_to_idx") != EXPECTED_STATE_CLASS_TO_IDX:
        raise RuntimeError(
            f"Unexpected parcel-state classes {checkpoint.get('class_to_idx')}; "
            f"expected {EXPECTED_STATE_CLASS_TO_IDX}."
        )
    threshold = float(checkpoint.get("open_box_threshold", -1))
    if PARCEL_STATE_OPEN_THRESHOLD_ENV in os.environ:
        threshold = float(os.environ[PARCEL_STATE_OPEN_THRESHOLD_ENV])
    if not 0 < threshold < 1:
        raise RuntimeError(f"Invalid open-box threshold in checkpoint/configuration: {threshold}")

    image_size = int(checkpoint.get("image_size", 224))
    normalization = checkpoint.get("normalization", {})
    mean, std = normalization.get("mean"), normalization.get("std")
    if not (isinstance(mean, list) and isinstance(std, list) and len(mean) == len(std) == 3):
        raise RuntimeError("Parcel-state checkpoint has invalid normalization metadata.")

    model = models.mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    preprocessing = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return {
        "model": model,
        "preprocessing": preprocessing,
        "open_box_threshold": threshold,
        "checkpoint_path": str(checkpoint_path),
        "validation_macro_f1": checkpoint.get("validation_macro_f1"),
    }


def classify_parcel_state(
    parcel_crop: Image.Image,
    loaded_models: dict,
    open_threshold: float | None = None,
) -> dict:
    """Classify one already-localized parcel crop as closed or open."""
    started = time.perf_counter()
    bundle = loaded_models["parcel_state_classifier"]
    model, preprocessing = bundle["model"], bundle["preprocessing"]
    device = loaded_models["device"]
    threshold = bundle["open_box_threshold"] if open_threshold is None else float(open_threshold)
    if not 0 < threshold < 1:
        raise ValueError(f"open_threshold must be between 0 and 1; received {threshold}")

    model.to(device)
    tensor = preprocessing(parcel_crop.convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        open_probability = float(torch.softmax(model(tensor), dim=1)[0, 1].cpu().item())
    state = "open_box" if open_probability >= threshold else "closed_box"
    confidence = open_probability if state == "open_box" else 1.0 - open_probability
    return {
        "parcel_state": state,
        "parcel_state_confidence": confidence,
        "open_box_probability": open_probability,
        "open_box_threshold": threshold,
        "warning": None,
        "inference_ms": (time.perf_counter() - started) * 1000,
    }
