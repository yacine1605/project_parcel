"""Prepare and persist human review of parcel-state classification crops."""

from __future__ import annotations

import csv
import shutil
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "datasets" / "raw" / "Custom Dataset for Closed and Open Cardboard Box.v21-boxes-and-hands-final.yolov11"
REVIEW_ROOT = ROOT / "datasets" / "processed" / "parcel_state_classification_review"
MANIFEST_PATH = REVIEW_ROOT / "manifest.csv"
EXPORT_ROOT = ROOT / "datasets" / "processed" / "parcel_state_classification_v1"

DECISIONS = ("closed_box", "open_box", "exclude")
SOURCE_LABELS = {0: "closed_box", 1: "hand", 2: "open_box"}
FIELDS = (
    "candidate_id", "split", "source_image", "source_label", "suggested_label",
    "crop_path", "decision", "reviewed_at",
)


def _read_polygons(label_path: Path) -> list[tuple[int, list[float]]]:
    polygons = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        coordinates = [float(value) for value in parts[1:]]
        if len(coordinates) < 6 or len(coordinates) % 2:
            raise ValueError(f"Malformed polygon in {label_path}: {line}")
        if any(value < 0 or value > 1 for value in coordinates):
            raise ValueError(f"Out-of-range polygon in {label_path}: {line}")
        polygons.append((int(parts[0]), coordinates))
    return polygons


def _polygon_crop(image: Image.Image, coordinates: list[float], padding: float) -> Image.Image:
    width, height = image.size
    xs, ys = coordinates[0::2], coordinates[1::2]
    x1, x2 = min(xs) * width, max(xs) * width
    y1, y2 = min(ys) * height, max(ys) * height
    pad_x, pad_y = (x2 - x1) * padding, (y2 - y1) * padding
    box = (
        max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y)),
        min(width, int(x2 + pad_x + 0.999)), min(height, int(y2 + pad_y + 0.999)),
    )
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f"Invalid crop box {box}")
    return image.crop(box)


def build_review_candidates(
    source: Path = SOURCE,
    review_root: Path = REVIEW_ROOT,
    padding: float = 0.04,
) -> Path:
    """Create review candidates from train/valid only; never modify the source."""
    manifest_path = review_root / "manifest.csv"
    if manifest_path.exists():
        raise FileExistsError(f"Review manifest already exists: {manifest_path}")

    rows: list[dict[str, str]] = []
    for split in ("train", "valid"):
        image_dir, label_dir = source / split / "images", source / split / "labels"
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f"Missing source split: {split}")
        for image_path in sorted(path for path in image_dir.iterdir() if path.is_file()):
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                raise FileNotFoundError(f"Missing annotation: {label_path}")
            polygons = _read_polygons(label_path)
            parcels = [(class_id, points) for class_id, points in polygons if class_id in (0, 2)]
            candidates = parcels or [(1, [])]
            with Image.open(image_path) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
                for object_index, (class_id, points) in enumerate(candidates):
                    candidate_id = f"{split}__{image_path.stem}__{object_index:02d}"
                    crop = _polygon_crop(image, points, padding) if points else image.copy()
                    relative_crop = Path("candidates") / split / f"{candidate_id}.jpg"
                    crop_path = review_root / relative_crop
                    crop_path.parent.mkdir(parents=True, exist_ok=True)
                    crop.save(crop_path, quality=95)
                    source_label = SOURCE_LABELS[class_id]
                    rows.append({
                        "candidate_id": candidate_id,
                        "split": split,
                        "source_image": str(image_path.relative_to(source)).replace("\\", "/"),
                        "source_label": source_label,
                        "suggested_label": source_label if source_label != "hand" else "exclude",
                        "crop_path": str(relative_crop).replace("\\", "/"),
                        "decision": "",
                        "reviewed_at": "",
                    })

    review_root.mkdir(parents=True, exist_ok=True)
    _write_manifest(manifest_path, rows)
    return manifest_path


def load_manifest(manifest_path: Path = MANIFEST_PATH) -> list[dict[str, str]]:
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_manifest(manifest_path: Path, rows: list[dict[str, str]]) -> None:
    temporary = manifest_path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(manifest_path)


def assign_candidate(
    candidate_id: str,
    decision: str,
    manifest_path: Path = MANIFEST_PATH,
    export_root: Path = EXPORT_ROOT,
) -> dict[str, str]:
    """Persist a human decision and synchronize its classification-folder copy."""
    if decision not in DECISIONS:
        raise ValueError(f"Decision must be one of {DECISIONS}; received {decision!r}")
    rows = load_manifest(manifest_path)
    matches = [row for row in rows if row["candidate_id"] == candidate_id]
    if len(matches) != 1:
        raise KeyError(f"Expected one candidate {candidate_id!r}; found {len(matches)}")
    row = matches[0]
    review_root = manifest_path.parent
    source_crop = review_root / row["crop_path"]
    if not source_crop.is_file():
        raise FileNotFoundError(f"Candidate crop is missing: {source_crop}")

    for label in ("closed_box", "open_box"):
        old_copy = export_root / row["split"] / label / source_crop.name
        if old_copy.exists():
            old_copy.unlink()
    if decision != "exclude":
        destination = export_root / row["split"] / decision / source_crop.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_crop, destination)

    row["decision"] = decision
    row["reviewed_at"] = datetime.now(timezone.utc).isoformat()
    _write_manifest(manifest_path, rows)
    return row
