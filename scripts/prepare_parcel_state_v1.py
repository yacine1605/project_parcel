"""Derive a two-class parcel-state YOLO dataset without touching test data."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "datasets" / "raw" / "Custom Dataset for Closed and Open Cardboard Box.v21-boxes-and-hands-final.yolov11"
OUTPUT = ROOT / "datasets" / "processed" / "parcel_state_v1"
CLASS_MAP = {0: 0, 2: 1}


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing derived dataset: {OUTPUT}")
    summary = {}
    for split in ("train", "valid"):
        source_images, source_labels = SOURCE / split / "images", SOURCE / split / "labels"
        output_images, output_labels = OUTPUT / split / "images", OUTPUT / split / "labels"
        output_images.mkdir(parents=True)
        output_labels.mkdir(parents=True)
        counts = Counter()
        for image_path in sorted(path for path in source_images.iterdir() if path.is_file()):
            label_path = source_labels / f"{image_path.stem}.txt"
            if not label_path.is_file():
                raise FileNotFoundError(f"Missing label for {image_path}")
            output_lines = []
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                class_id, coordinates = int(parts[0]), parts[1:]
                if len(coordinates) < 4 or len(coordinates) % 2:
                    raise ValueError(f"Malformed polygon in {label_path}: {line}")
                values = [float(value) for value in coordinates]
                if any(value < 0 or value > 1 for value in values):
                    raise ValueError(f"Out-of-range coordinate in {label_path}")
                if class_id in CLASS_MAP:
                    mapped = CLASS_MAP[class_id]
                    output_lines.append(" ".join([str(mapped), *coordinates]))
                    counts[mapped] += 1
            shutil.copy2(image_path, output_images / image_path.name)
            (output_labels / label_path.name).write_text(
                "\n".join(output_lines) + ("\n" if output_lines else ""), encoding="utf-8"
            )
        summary[split] = {
            "images": len(list(output_images.iterdir())),
            "closed_box_instances": counts[0], "open_box_instances": counts[1],
            "background_images": sum(not p.read_text(encoding="utf-8").strip() for p in output_labels.glob("*.txt")),
        }
    (OUTPUT / "data.yaml").write_text(yaml.safe_dump({
        "path": ".", "train": "train/images", "val": "valid/images",
        "nc": 2, "names": ["closed_box", "open_box"],
    }, sort_keys=False), encoding="utf-8")
    manifest = {
        "source": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
        "source_license": "CC BY 4.0",
        "source_url": "https://universe.roboflow.com/cardbox-damage-detection/custom-dataset-for-closed-and-open-cardboard-box/dataset/21",
        "source_class_mapping": {"Closed Cardboard Box": "closed_box", "Open Cardboard Box": "open_box", "Hand": "ignored/background"},
        "splits_used": ["train", "valid"], "test_accessed": False, "summary": summary,
    }
    (OUTPUT / "derivation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
