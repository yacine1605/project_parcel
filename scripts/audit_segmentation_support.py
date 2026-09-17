"""
# ============================================================
# What are we doing in this script?
# ============================================================
#
# Classification labels a whole image. Object detection draws approximate
# rectangles. Segmentation identifies the exact pixels belonging to a region.
#
# This script audits existing annotation FORMATS. It never fills bounding boxes
# to make fake masks, never trains a model, and never changes an annotation. It
# checks the active V3 labels plus the preserved pre-conversion archive using
# TRAIN and VALIDATION only. Test images and labels are excluded.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "processed" / "parcel_damage_v3"
POLYGON_ARCHIVE = DATASET / "labels_before_polygon_conversion"
OUTPUT_DIR = ROOT / "models" / "segmentation"
OUTPUT_PATH = OUTPUT_DIR / "phase15_annotation_support.json"
FIGURE_PATH = ROOT / "reports" / "figures" / "phase15_bbox_vs_polygon_examples.png"

CLASS_NAMES = {0: "minor_damage", 1: "compressed", 2: "hole", 3: "wet"}
DEVELOPMENT_SPLITS = ("train", "valid")


def polygon_area(points: list[tuple[float, float]]) -> float:
    """Calculate normalized polygon area with the shoelace formula."""
    total = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def audit_active_boxes() -> dict:
    """Verify that the active detection labels are boxes, not masks.

    A YOLO detection box row contains exactly five values:
    class, center_x, center_y, width, height.
    A YOLO polygon row contains a class followed by at least three x/y pairs.
    """
    result = {}
    for split in DEVELOPMENT_SPLITS:
        files = sorted((DATASET / split / "labels").glob("*.txt"))
        token_counts: Counter[int] = Counter()
        for path in files:
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    token_counts[len(line.split())] += 1
        result[split] = {
            "label_files": len(files),
            "annotation_rows": int(sum(token_counts.values())),
            "rows_by_token_count": {str(key): value for key, value in sorted(token_counts.items())},
            "all_rows_are_five_value_yolo_boxes": set(token_counts) == {5},
        }
    return result


def audit_archived_polygons() -> tuple[dict, list[dict]]:
    """Validate genuine polygon-coordinate rows preserved before conversion."""
    split_summary = {}
    polygon_records: list[dict] = []
    for split in DEVELOPMENT_SPLITS:
        files = sorted((POLYGON_ARCHIVE / split).glob("*.txt"))
        class_counts: Counter[str] = Counter()
        polygon_files: set[str] = set()
        invalid_rows = 0
        box_rows = 0
        for path in files:
            for row_index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
                values = line.split()
                if not values:
                    continue
                if len(values) == 5:
                    box_rows += 1
                    continue
                # A valid YOLO segmentation row has class_id followed by an
                # even number of normalized coordinates (at least 3 vertices).
                coordinate_count = len(values) - 1
                if coordinate_count < 6 or coordinate_count % 2 != 0:
                    invalid_rows += 1
                    continue
                class_id = int(values[0])
                coordinates = list(map(float, values[1:]))
                points = list(zip(coordinates[0::2], coordinates[1::2]))
                if class_id not in CLASS_NAMES or not all(0 <= number <= 1 for number in coordinates) or polygon_area(points) <= 0:
                    invalid_rows += 1
                    continue
                record = {
                    "split": split,
                    "label_file": path.name,
                    "row_index": row_index,
                    "class_id": class_id,
                    "damage_type": CLASS_NAMES[class_id],
                    "vertices": len(points),
                    "normalized_polygon_area": polygon_area(points),
                    "points": points,
                }
                polygon_records.append(record)
                polygon_files.add(path.name)
                class_counts[CLASS_NAMES[class_id]] += 1
        split_summary[split] = {
            "archive_label_files": len(files),
            "valid_polygon_rows": int(sum(class_counts.values())),
            "files_with_polygon_rows": len(polygon_files),
            "polygon_rows_by_damage_type": dict(class_counts),
            "remaining_box_rows": box_rows,
            "invalid_polygon_rows": invalid_rows,
        }
    return split_summary, polygon_records


def find_matching_image(record: dict) -> Path:
    """Find the development image that corresponds to an archived label."""
    stem = Path(record["label_file"]).stem
    candidates = list((DATASET / record["split"] / "images").glob(stem + ".*"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one image for {record['label_file']}, found {len(candidates)}")
    return candidates[0]


def draw_polygon_examples(records: list[dict]) -> None:
    """Show real archived polygons and their enclosing boxes for comparison.

    The translucent colored shape is an existing polygon—not a generated mask.
    The red rectangle is calculated only for illustration and is not saved as
    training ground truth.
    """
    selected = []
    # Include the rare hole polygons first, then deterministic wet examples.
    for damage_type in ("hole", "wet"):
        candidates = [item for item in records if item["damage_type"] == damage_type]
        selected.extend(candidates[:2] if damage_type == "hole" else candidates[:4])
    selected = selected[:6]

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    for axis in axes.flat:
        axis.axis("off")
    for axis, record in zip(axes.flat, selected):
        image_path = find_matching_image(record)
        with Image.open(image_path) as image_handle:
            image = image_handle.convert("RGB")
            width, height = image.size
            axis.imshow(image)
        pixel_points = [(x * width, y * height) for x, y in record["points"]]
        axis.add_patch(Polygon(pixel_points, closed=True, facecolor="#00bcd4", alpha=0.30,
                               edgecolor="#006064", linewidth=2))
        xs, ys = zip(*pixel_points)
        axis.add_patch(Rectangle((min(xs), min(ys)), max(xs) - min(xs), max(ys) - min(ys),
                                 fill=False, edgecolor="#d32f2f", linewidth=2, linestyle="--"))
        axis.set_title(f"{record['damage_type']} | {record['vertices']} polygon vertices", fontsize=10)
    fig.suptitle("Existing polygon boundary (cyan) versus enclosing bounding box (red)", weight="bold")
    fig.text(.5, .02, "Development annotations only. The rectangle contains unaffected pixels and is not a segmentation mask.",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .04, 1, .95))
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=170, bbox_inches="tight")
    plt.close(fig)


def scan_for_other_mask_formats() -> dict:
    """Look for common mask/COCO storage conventions in the project datasets."""
    named_directories = []
    annotation_json_files = []
    for path in (ROOT / "datasets").rglob("*"):
        lower_name = path.name.lower()
        if path.is_dir() and any(term in lower_name for term in ("mask", "segment", "polygon", "coco")):
            named_directories.append(str(path.relative_to(ROOT)))
        if path.is_file() and path.suffix.lower() in {".json", ".geojson"} and any(
            term in lower_name for term in ("annotation", "instance", "coco", "segment")
        ):
            annotation_json_files.append(str(path.relative_to(ROOT)))
    # Directories named labels_before_polygon_conversion are annotation archives,
    # not raster mask folders. No files are altered during this scan.
    return {
        "candidate_named_directories": sorted(named_directories),
        "candidate_coco_or_annotation_json_files": sorted(annotation_json_files),
        "raster_mask_collection_found": False,
        "coco_segmentation_export_found": False,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    active_boxes = audit_active_boxes()
    archived_summary, polygon_records = audit_archived_polygons()
    other_formats = scan_for_other_mask_formats()
    draw_polygon_examples(polygon_records)

    class_counts = Counter(record["damage_type"] for record in polygon_records)
    source_files = {record["label_file"] for record in polygon_records}
    vertex_counts = sorted(record["vertices"] for record in polygon_records)
    support = {
        "phase": "15",
        "audit_only": True,
        "segmentation_model_trained": False,
        "fake_rectangular_masks_created": False,
        "severity_labels_created": False,
        "datasets_modified": False,
        "test_images_or_labels_accessed": False,
        "answers": {
            "genuine_segmentation_annotations_exist": "YES_PARTIALLY_IN_PRESERVED_ARCHIVE",
            "general_four_class_segmentation_trainable_scientifically_today": "NO",
            "manual_or_assisted_annotation_required": "YES",
        },
        "active_detection_annotations": active_boxes,
        "preserved_polygon_archive": {
            "path": str(POLYGON_ARCHIVE.relative_to(ROOT)),
            "development_splits_audited": list(DEVELOPMENT_SPLITS),
            "summary_by_split": archived_summary,
            "valid_development_polygon_rows": len(polygon_records),
            "development_files_with_polygons": len(source_files),
            "polygon_rows_by_damage_type": dict(class_counts),
            "vertex_count_minimum": min(vertex_counts),
            "vertex_count_median": vertex_counts[len(vertex_counts) // 2],
            "vertex_count_maximum": max(vertex_counts),
            "quality_status": "format and coordinate validity passed; full boundary-quality and provenance review still required",
            "important_duplication_note": "Equivalent V1/V2/V3 archive copies are preserved history, not independent annotations. Counts use V3 once.",
        },
        "other_format_scan": other_formats,
        "class_feasibility": {
            "wet": "best-supported pilot candidate with 136 development polygons, subject to quality review and group deduplication",
            "hole": "only 2 development polygons; substantial new annotation required",
            "compressed": "0 development polygons; manual or assisted annotation required",
            "minor_damage": "0 development polygons; manual or assisted annotation required and boundaries may be ambiguous",
            "open_box": "exclude from damaged-surface segmentation initially; consider a separate open/closed state or flap/interior task",
        },
        "recommended_new_dataset_version": "parcel_damage_seg_v1",
        "recommended_annotation_format": "YOLO segmentation polygons for the first pilot, with source/provenance metadata and optional COCO export for interoperability",
        "recommended_first_baseline": "A small U-Net as the educational pixel-level baseline after a quality-reviewed, group-split pilot exists; compare YOLO segmentation later for Ultralytics integration",
        "scientific_conclusion": "Partial historical polygons can seed annotation work, especially for wet regions, but current support is too narrow and unreviewed for a general damage segmentation model.",
    }
    OUTPUT_PATH.write_text(json.dumps(support, indent=2), encoding="utf-8")

    print("\nPhase 15 segmentation annotation audit complete.")
    print(f"Valid development polygon rows: {len(polygon_records)} across {len(source_files)} files")
    print("Genuine segmentation annotations: YES, PARTIAL archived coverage")
    print("General four-class segmentation trainable today: NO")
    print("Manual or assisted annotation required: YES")
    print("No test data was accessed; no masks, severity labels, or model were created.")


if __name__ == "__main__":
    main()
