"""Train the parcel-localization stage on the Roboflow parcel-detect v3 export."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "raw" / "parcel-detect.v3-parcel-black_pad-320-.yolov11"
RUN_NAME = "PARCEL-DET-001_yolo11n_v3"

# Prevent Ultralytics from attempting to write settings into the user profile.
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "Ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="0", help="CUDA device number or 'cpu'.")
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def write_runtime_dataset_yaml() -> Path:
    """Correct the exported relative paths without modifying the raw dataset."""
    runtime_yaml = ROOT / "runs" / RUN_NAME / "parcel_detector_data.yaml"
    runtime_yaml.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "path": DATASET.resolve().as_posix(),
        "train": "train/images",
        "val": "valid/images",
        "nc": 1,
        "names": ["Boxes"],
    }
    runtime_yaml.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return runtime_yaml


def main() -> None:
    args = parse_args()
    if not (DATASET / "train" / "images").is_dir() or not (DATASET / "valid" / "images").is_dir():
        raise FileNotFoundError(f"Incomplete parcel detector dataset: {DATASET}")

    model = YOLO(str(ROOT / "yolo11n.pt"))
    model.train(
        data=str(write_runtime_dataset_yaml()),
        imgsz=320,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        pretrained=True,
        seed=42,
        deterministic=True,
        project=str(ROOT / "runs"),
        name=RUN_NAME,
        exist_ok=True,
        plots=True,
        val=True,
        save=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
