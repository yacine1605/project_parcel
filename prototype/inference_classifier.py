"""
# ============================================================
# What does MobileNet do in this prototype?
# ============================================================
#
# MobileNet answers an image-level question:
#     "Does this cropped, closed parcel look damaged or intact?"
#
# It returns one damaged probability and one label. It does not return a
# bounding box. This file loads the exact frozen Phase 10D/11 contract and never
# trains or updates the model.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FREEZE_MANIFEST = PROJECT_ROOT / "models" / "selected_transfer_model" / "selected_transfer_model.json"


def file_sha256(path: Path) -> str:
    """Calculate a checkpoint fingerprint for the frozen-artifact check."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_device() -> torch.device:
    """Choose the GPU when CUDA is available, otherwise use the CPU."""
    # A GPU performs many tensor operations in parallel and is normally faster
    # for neural-network inference. The model and input tensor must be on the
    # same device; otherwise PyTorch cannot perform the forward pass.
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_frozen_preprocessing():
    """Reproduce the exact deterministic validation/test preprocessing."""
    return transforms.Compose([
        # Resize the shorter side to 256 while preserving image aspect ratio.
        transforms.Resize(256),
        # The model expects a 224 x 224 view. CenterCrop is deterministic: webcam
        # and uploaded images never receive random training augmentation.
        transforms.CenterCrop(224),
        # Convert RGB pixels to a float tensor with shape [3, 224, 224].
        transforms.ToTensor(),
        # ImageNet normalization is required because the frozen backbone learned
        # its visual features using this channel scale.
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def load_frozen_classifier(device: torch.device):
    """Load MobileNet, verify its hash, and return its frozen threshold."""
    if not FREEZE_MANIFEST.is_file():
        raise FileNotFoundError(f"Missing classifier freeze manifest: {FREEZE_MANIFEST}")
    contract = json.loads(FREEZE_MANIFEST.read_text(encoding="utf-8"))
    checkpoint_path = PROJECT_ROOT / contract["checkpoint_path"]
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing MobileNet checkpoint: {checkpoint_path}")
    observed_hash = file_sha256(checkpoint_path)
    if observed_hash != contract["checkpoint_sha256"]:
        raise RuntimeError("MobileNet checkpoint hash does not match the frozen contract.")

    # weights=None prevents torchvision from downloading or substituting a
    # different ImageNet checkpoint. Our saved file contains the complete state.
    model = models.mobilenet_v3_large(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 1)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    for parameter in model.parameters():
        # requires_grad=False means this parameter cannot learn or be updated.
        parameter.requires_grad = False
    model = model.to(device)
    model.eval()
    # eval() disables Dropout randomness and tells BatchNorm to use its saved
    # running statistics. It does not alter the learned weights.
    return model, create_frozen_preprocessing(), float(contract["frozen_threshold"]), contract


def synchronize_if_cuda(device: torch.device) -> None:
    """Wait for queued GPU work so measured inference time is meaningful."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def predict_damage_classification(
    image: Image.Image,
    model: nn.Module,
    preprocessing,
    device: torch.device,
    frozen_threshold: float,
) -> dict:
    """Return the frozen MobileNet probability, label, and timing.

    Expected shapes:
        RGB parcel crop before preprocessing: [height, width, 3]
        image tensor:                 [3, 224, 224]
        batch tensor:              [1, 3, 224, 224]
        model logit:                         [1]
    """
    preprocessing_started = time.perf_counter()
    rgb_image = image.convert("RGB")
    image_tensor = preprocessing(rgb_image)
    batch_tensor = image_tensor.unsqueeze(0)
    # unsqueeze(0) adds the batch dimension. Even one image is passed as a batch.
    batch_tensor = batch_tensor.to(device)
    synchronize_if_cuda(device)
    preprocessing_ms = (time.perf_counter() - preprocessing_started) * 1000

    synchronize_if_cuda(device)
    inference_started = time.perf_counter()
    with torch.no_grad():
        # no_grad() turns off gradient storage because inference does not learn.
        # There is no loss, backward pass, optimizer, or weight update here.
        logit = model(batch_tensor).squeeze(1)
    synchronize_if_cuda(device)
    inference_ms = (time.perf_counter() - inference_started) * 1000

    # A logit is an unrestricted raw model output. Sigmoid converts it into a
    # value between 0 and 1 that we interpret as damaged probability.
    probability = float(torch.sigmoid(logit)[0].cpu().item())
    label = "damaged" if probability >= frozen_threshold else "intact"
    # The threshold was selected with validation data and frozen before final
    # test evaluation. It must not be adjusted after seeing prototype examples.
    return {
        "label": label,
        "damaged_probability": probability,
        "threshold": frozen_threshold,
        "raw_logit": float(logit[0].cpu().item()),
        "preprocessing_ms": preprocessing_ms,
        "inference_ms": inference_ms,
        "input_tensor_shape": list(batch_tensor.shape),
    }
