"""Export every compressed annotation for manual ontology review.

This script is read-only with respect to the frozen dataset. It creates contact
sheets and a CSV checklist under reports/compressed_ontology_review/.
"""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "processed" / "parcel_damage_v2"
OUTPUT = ROOT / "reports" / "compressed_ontology_review"
CLASS_NAMES = ["minor_damage", "compressed", "hole", "wet"]
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
TILE_SIZE = (320, 260)
GRID = (4, 3)


def source_name_flag(name: str) -> str:
    """Return a review hint only; filenames are never treated as ground truth."""
    normalized = name.lower()
    flags = []
    for label, pattern in (
        ("tear", r"tear"),
        ("hole_or_puncture", r"hole|puncture"),
        ("broken", r"broken"),
        ("creased", r"creas"),
        ("compressed", r"compress|crush"),
    ):
        if re.search(pattern, normalized):
            flags.append(label)
    return ";".join(flags)


def image_for_stem(directory: Path, stem: str) -> Path | None:
    for extension in IMAGE_EXTENSIONS:
        candidate = directory / f"{stem}{extension}"
        if candidate.exists():
            return candidate
    return None


def read_labels(path: Path) -> list[tuple[int, float, float, float, float]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        class_id, x, y, width, height = line.split()
        rows.append((int(class_id), float(x), float(y), float(width), float(height)))
    return rows


def annotated_tile(image_path: Path, labels, title: str) -> Image.Image:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    image.thumbnail((TILE_SIZE[0], TILE_SIZE[1] - 36))
    tile = Image.new("RGB", TILE_SIZE, "#181818")
    left = (TILE_SIZE[0] - image.width) // 2
    top = 30 + (TILE_SIZE[1] - 30 - image.height) // 2
    tile.paste(image, (left, top))
    draw = ImageDraw.Draw(tile)
    draw.text((6, 7), title[:48], fill="white", font=ImageFont.load_default())
    colors = ("#4da3ff", "#35d04f", "#ff4d4d", "#ffe24d")
    for class_id, x, y, width, height in labels:
        x1 = left + (x - width / 2) * image.width
        y1 = top + (y - height / 2) * image.height
        x2 = left + (x + width / 2) * image.width
        y2 = top + (y + height / 2) * image.height
        color = colors[class_id]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        draw.text((x1 + 2, max(top, y1 - 12)), CLASS_NAMES[class_id], fill=color)
    return tile


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = []
    tiles = []
    for split in ("train", "valid"):
        labels_dir = DATASET / split / "labels"
        images_dir = DATASET / split / "images"
        for label_path in sorted(labels_dir.glob("*.txt")):
            labels = read_labels(label_path)
            compressed = [row for row in labels if row[0] == 1]
            if not compressed:
                continue
            image_path = image_for_stem(images_dir, label_path.stem)
            if image_path is None:
                continue
            other_classes = sorted({CLASS_NAMES[row[0]] for row in labels if row[0] != 1})
            records.append({
                "split": split,
                "image": image_path.relative_to(ROOT).as_posix(),
                "label": label_path.relative_to(ROOT).as_posix(),
                "compressed_boxes": len(compressed),
                "coexisting_classes": ";".join(other_classes),
                "source_name_flag": source_name_flag(image_path.name),
                "review_decision": "",
                "review_notes": "",
            })
            tiles.append(annotated_tile(image_path, labels, f"{split} | {image_path.name}"))

    csv_path = OUTPUT / "compressed_review_checklist.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    per_page = GRID[0] * GRID[1]
    for page_index in range(math.ceil(len(tiles) / per_page)):
        sheet = Image.new("RGB", (GRID[0] * TILE_SIZE[0], GRID[1] * TILE_SIZE[1]), "#101010")
        for index, tile in enumerate(tiles[page_index * per_page : (page_index + 1) * per_page]):
            sheet.paste(tile, ((index % GRID[0]) * TILE_SIZE[0], (index // GRID[0]) * TILE_SIZE[1]))
        sheet.save(OUTPUT / f"compressed_contact_sheet_{page_index + 1:02d}.jpg", quality=92)

    print(f"Exported {len(records)} images to {OUTPUT}")


if __name__ == "__main__":
    main()
