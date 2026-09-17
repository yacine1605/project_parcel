"""Extract reproducible Phase 7 HOG features from Parcel Binary V1."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from skimage.feature import hog


IMAGE_SIZE = (128, 128)
HOG_PARAMS = {
    "orientations": 9,
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys",
}


def extract(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        gray = ImageOps.grayscale(image)
        gray = ImageOps.fit(gray, IMAGE_SIZE, method=Image.Resampling.LANCZOS)
        array = np.asarray(gray, dtype=np.float32) / 255.0
    return hog(array, feature_vector=True, **HOG_PARAMS).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("datasets/processed/parcel_binary_v1"))
    parser.add_argument("--output", type=Path, default=Path("datasets/processed/parcel_binary_v1/features"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.dataset / "manifests" / "dataset_manifest.csv"
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))
    timings: dict[str, float] = {}
    failures: list[dict[str, str]] = []
    feature_dim = 0
    for split in ("train", "valid", "test"):
        rows = [r for r in manifest if r["new_split"] == split and r["audit_status"] != "excluded"]
        features, labels, ids = [], [], []
        started = time.perf_counter()
        for row in rows:
            path = args.dataset / split / row["binary_class"] / f'{row["image_id"]}.jpg'
            try:
                vector = extract(path)
                features.append(vector)
                labels.append(1 if row["binary_class"] == "damaged" else 0)
                ids.append(row["image_id"])
            except Exception as exc:
                failures.append({"image_id": row["image_id"], "split": split, "error": f"{type(exc).__name__}: {exc}"})
        X = np.stack(features) if features else np.empty((0, 0), dtype=np.float32)
        y = np.asarray(labels, dtype=np.uint8)
        feature_dim = X.shape[1] if X.size else feature_dim
        np.savez_compressed(args.output / f"hog_{split}.npz", X=X, y=y, image_ids=np.asarray(ids))
        timings[split] = time.perf_counter() - started
        print(split, X.shape, f"{timings[split]:.2f}s")
    metadata = {
        "image_size": list(IMAGE_SIZE),
        "color_conversion": "Pillow grayscale (L), scaled to [0,1]",
        "resize": "ImageOps.fit center crop with Lanczos",
        "hog": {k: list(v) if isinstance(v, tuple) else v for k, v in HOG_PARAMS.items()},
        "feature_dimensionality": feature_dim,
        "processing_seconds": timings,
        "failures": failures,
    }
    (args.output / "hog_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
