"""Prepare an experiment-only manifest for compressed-class oversampling."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "processed" / "parcel_damage_v3"
CONFIG_DIR = ROOT / "experiments" / "EXP-003"
MANIFEST = CONFIG_DIR / "train_compressed_x3.txt"
DATA_YAML = CONFIG_DIR / "data.yaml"
SUMMARY = CONFIG_DIR / "sampling_summary.json"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


def image_for_label(label: Path) -> Path:
    image_dir = label.parents[1] / "images"
    for extension in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{label.stem}{extension}"
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"No image for {label}")


def contains_compressed(label: Path) -> bool:
    return any(line.split() and line.split()[0] == "1" for line in label.read_text(encoding="utf-8").splitlines())


def main() -> None:
    labels = sorted((DATASET / "train" / "labels").glob("*.txt"))
    images = [image_for_label(label) for label in labels]
    compressed_images = [image for image, label in zip(images, labels) if contains_compressed(label)]
    manifest_images = images + compressed_images + compressed_images

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text("\n".join(image.as_posix() for image in manifest_images) + "\n", encoding="utf-8")
    DATA_YAML.write_text(
        "\n".join(
            [
                f"train: {MANIFEST.resolve().as_posix()}",
                f"val: {(DATASET / 'valid' / 'images').resolve().as_posix()}",
                f"test: {(DATASET / 'test' / 'images').resolve().as_posix()}",
                "",
                "nc: 4",
                "names:",
                "  0: minor_damage",
                "  1: compressed",
                "  2: hole",
                "  3: wet",
                "",
            ]
        ),
        encoding="utf-8",
    )
    summary = {
        "dataset": DATASET.relative_to(ROOT).as_posix(),
        "original_training_images": len(images),
        "compressed_training_images": len(compressed_images),
        "compressed_exposure_multiplier": 3,
        "manifest_entries": len(manifest_images),
        "additional_sampling_entries": len(manifest_images) - len(images),
        "dataset_files_modified": 0,
    }
    SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
