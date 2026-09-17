"""Extract Phase 8D train/validation HOG features; never opens the test split."""

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
HOG_PARAMS = {"orientations": 9, "pixels_per_cell": (8, 8), "cells_per_block": (2, 2), "block_norm": "L2-Hys"}


def extract(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        gray = ImageOps.grayscale(image)
        gray = ImageOps.fit(gray, IMAGE_SIZE, method=Image.Resampling.LANCZOS)
        array = np.asarray(gray, dtype=np.float32) / 255.0
    return hog(array, feature_vector=True, **HOG_PARAMS).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("datasets/processed/parcel_binary_v2"))
    parser.add_argument("--output", type=Path, default=Path("models/phase8d_artifacts/features"))
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    with (args.dataset / "manifests/dataset_manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if r["v2_split"] in {"train", "valid"}]
    group_splits: dict[str, set[str]] = {}
    for row in rows:
        group_splits.setdefault(row["group_id"], set()).add(row["v2_split"])
    leaking = {g: sorted(v) for g, v in group_splits.items() if len(v) > 1}
    if leaking:
        raise RuntimeError(f"Train/validation group leakage: {leaking}")
    failures, timings, shapes = [], {}, {}
    for split in ("train", "valid"):
        selected = [r for r in rows if r["v2_split"] == split]
        features, labels, ids, adjudications = [], [], [], []
        started = time.perf_counter()
        for row in selected:
            path = args.dataset / row["v2_relative_path"]
            try:
                vector = extract(path)
                if not np.isfinite(vector).all():
                    raise ValueError("HOG contains NaN or Inf")
                features.append(vector); labels.append(row["v2_binary_class"] == "damaged")
                ids.append(row["image_id"]); adjudications.append(row["phase8b_final_label"])
            except Exception as exc:
                failures.append({"image_id": row["image_id"], "split": split, "error": f"{type(exc).__name__}: {exc}"})
        X = np.stack(features).astype(np.float32)
        if len(X) != len(selected):
            raise RuntimeError(f"{split}: extracted {len(X)} of {len(selected)} manifest rows")
        np.savez_compressed(args.output / f"hog_{split}.npz", X=X, y=np.asarray(labels, dtype=np.uint8), image_ids=np.asarray(ids), adjudications=np.asarray(adjudications))
        timings[split], shapes[split] = time.perf_counter() - started, list(X.shape)
        print(split, X.shape)
    if shapes["train"][1] != shapes["valid"][1]:
        raise RuntimeError("Inconsistent feature dimensionality")
    metadata = {"dataset": "parcel_binary_v2", "splits_opened": ["train", "valid"], "test_opened": False,
        "image_size": list(IMAGE_SIZE), "color_conversion": "Pillow grayscale (L), scaled to [0,1]",
        "resize": "ImageOps.fit center crop with Lanczos", "hog": {k: list(v) if isinstance(v, tuple) else v for k, v in HOG_PARAMS.items()},
        "feature_dimensionality": shapes["train"][1], "shapes": shapes, "processing_seconds": timings,
        "failures": failures, "train_validation_group_leakage": leaking, "all_values_finite": True}
    (args.output / "hog_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__": main()
