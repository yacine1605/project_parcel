"""
# ============================================================
# What are we doing in this script?
# ============================================================
#
# Severity means how serious the physical parcel damage is. It is not model
# confidence, classification probability, bounding-box area, or box count.
# A severity model needs trustworthy labels that describe seriousness.
#
# This script audits existing annotation schemas and computes descriptive box
# geometry from the YOLO TRAIN and VALIDATION annotations only. It does not
# inspect test images/labels, load a model, train a model, or rewrite a label.
# Box area is reported as a possible proxy feature, never as severity truth.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
YOLO_ROOT = ROOT / "datasets" / "processed" / "parcel_damage_v3"
BINARY_MANIFEST = ROOT / "datasets" / "processed" / "parcel_binary_v2" / "manifests" / "dataset_manifest.csv"
OUTPUT_DIR = ROOT / "models" / "severity"
FIGURE_DIR = ROOT / "reports" / "figures"
SUPPORT_PATH = OUTPUT_DIR / "phase14_annotation_support.json"
SUMMARY_CSV = OUTPUT_DIR / "phase14_box_area_summary.csv"
FIGURE_PATH = FIGURE_DIR / "phase14_box_area_distributions.png"

CLASS_NAMES = {
    0: "minor_damage",
    1: "compressed",
    2: "hole",
    3: "wet",
}


def read_yolo_development_annotations() -> tuple[list[dict], dict]:
    """Read normalized boxes from train/valid while explicitly excluding test.

    A YOLO row is: class_id, center_x, center_y, width, height. Coordinates are
    normalized to the image, so width * height approximates the fraction of the
    image covered by the annotation rectangle.
    """
    records: list[dict] = []
    split_counts: dict[str, dict] = {}
    for split in ("train", "valid"):
        label_dir = YOLO_ROOT / split / "labels"
        label_files = sorted(label_dir.glob("*.txt"))
        image_box_counts: Counter[str] = Counter()
        for label_path in label_files:
            rows = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            for box_index, line in enumerate(rows):
                values = line.split()
                if len(values) != 5:
                    raise RuntimeError(f"Malformed YOLO row in {label_path}: {line}")
                class_id = int(values[0])
                center_x, center_y, width, height = map(float, values[1:])
                if class_id not in CLASS_NAMES:
                    raise RuntimeError(f"Unknown class {class_id} in {label_path}")
                if not (0 < width <= 1 and 0 < height <= 1):
                    raise RuntimeError(f"Invalid normalized box size in {label_path}")
                relative_area = width * height
                records.append({
                    "split": split,
                    "label_file": label_path.name,
                    "box_index": box_index,
                    "class_id": class_id,
                    "damage_type": CLASS_NAMES[class_id],
                    "normalized_width": width,
                    "normalized_height": height,
                    "relative_box_area": relative_area,
                })
                image_box_counts[label_path.name] += 1
        split_counts[split] = {
            "label_files": len(label_files),
            "boxes": int(sum(image_box_counts.values())),
            "images_with_multiple_boxes": int(sum(count > 1 for count in image_box_counts.values())),
        }
    return records, split_counts


def summarize_box_areas(records: list[dict]) -> dict:
    """Summarize geometry per nominal damage type without grading severity."""
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        grouped[record["damage_type"]].append(record["relative_box_area"])

    summary = {}
    for damage_type in CLASS_NAMES.values():
        values = np.asarray(grouped[damage_type], dtype=float)
        summary[damage_type] = {
            "boxes": int(values.size),
            "minimum": float(values.min()),
            "p10": float(np.quantile(values, 0.10)),
            "p25": float(np.quantile(values, 0.25)),
            "median": float(np.median(values)),
            "mean": float(values.mean()),
            "p75": float(np.quantile(values, 0.75)),
            "p90": float(np.quantile(values, 0.90)),
            "maximum": float(values.max()),
        }
    return summary


def inspect_annotation_schemas() -> dict:
    """Record which potentially useful fields exist, without reading test data.

    Only the binary manifest header and its train/validation rows are used here.
    The schema itself shows whether an explicit severity column exists.
    """
    with BINARY_MANIFEST.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        development_rows = [row for row in reader if row.get("v2_split") in {"train", "valid"}]

    severity_like_column_names = [
        name for name in columns
        if any(term in name.lower() for term in ("severity", "grade", "reject", "accept", "outcome", "integrity", "area"))
    ]
    return {
        "binary_manifest_columns": columns,
        "binary_development_rows_audited": len(development_rows),
        "severity_like_column_names": severity_like_column_names,
        "available_binary_labels": sorted({row["v2_binary_class"] for row in development_rows}),
        "available_review_labels": sorted({row["phase8b_final_label"] for row in development_rows}),
        "available_review_flags": sorted({row["phase8b_review_flags"] for row in development_rows if row["phase8b_review_flags"]}),
        "yolo_row_schema": ["class_id", "center_x", "center_y", "normalized_width", "normalized_height"],
        "yolo_class_mapping": {str(key): value for key, value in CLASS_NAMES.items()},
    }


def write_summary_csv(summary: dict) -> None:
    """Save exact distribution statistics in a small reusable table."""
    columns = ["damage_type", "boxes", "minimum", "p10", "p25", "median", "mean", "p75", "p90", "maximum"]
    with SUMMARY_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for damage_type, values in summary.items():
            writer.writerow({"damage_type": damage_type, **values})


def plot_box_areas(records: list[dict]) -> None:
    """Plot proxy distributions; the reference line is not a severity cutoff."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    ordered = list(CLASS_NAMES.values())
    data = [[record["relative_box_area"] for record in records if record["damage_type"] == name] for name in ordered]

    axes[0].boxplot(data, tick_labels=ordered, showfliers=True)
    axes[0].set_ylabel("Relative bounding-box area (width × height)")
    axes[0].set_title("Development annotations by damage type")
    axes[0].tick_params(axis="x", rotation=20)

    bins = np.linspace(0, 1, 31)
    for values, name in zip(data, ordered):
        axes[1].hist(values, bins=bins, histtype="step", linewidth=1.8, label=f"{name} (n={len(values)})")
    axes[1].set_xlabel("Relative bounding-box area")
    axes[1].set_ylabel("Annotated regions")
    axes[1].set_title("Overlapping area distributions")
    axes[1].legend(fontsize=8)

    fig.suptitle("Phase 14: box size is a proxy, not severity ground truth", weight="bold")
    fig.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    records, split_counts = read_yolo_development_annotations()
    summary = summarize_box_areas(records)
    schemas = inspect_annotation_schemas()
    write_summary_csv(summary)
    plot_box_areas(records)

    support = {
        "phase": "14",
        "audit_only": True,
        "severity_model_trained": False,
        "datasets_modified": False,
        "test_images_or_labels_accessed": False,
        "scope": {
            "yolo_dataset": "parcel_damage_v3",
            "yolo_splits_audited": ["train", "valid"],
            "explicitly_excluded_split": "test",
            "binary_dataset": "parcel_binary_v2",
            "binary_rows_audited": ["train", "valid"],
        },
        "answers": {
            "explicit_severity_labels_exist": "NO",
            "reliable_severity_derivable_without_guessing": "NO",
            "manual_annotation_required": "YES",
        },
        "annotation_schema": schemas,
        "development_annotation_counts": split_counts,
        "box_area_by_damage_type": summary,
        "signals": {
            "explicit_severity_label": {"exists": False, "reliable": False, "kind": "missing"},
            "damage_type": {"exists": True, "reliable": "suitable_as_nominal ontology after prior review", "kind": "proxy/context"},
            "bounding_box_area": {"exists": True, "reliable": "geometrically measurable but annotation-dependent", "kind": "proxy_only"},
            "number_of_damage_regions": {"exists": True, "reliable": "annotation-dependent", "kind": "proxy_only"},
            "open_box_state": {"exists": True, "reliable": "reviewed binary subgroup metadata", "kind": "state/context_not_severity"},
            "damage_localization": {"exists": True, "reliable": "available for four detector classes", "kind": "proxy/context"},
            "operational_outcome": {"exists": False, "reliable": False, "kind": "missing"},
            "contents_exposed_extent": {"exists": False, "reliable": False, "kind": "missing"},
            "structural_integrity_grade": {"exists": False, "reliable": False, "kind": "missing"},
        },
        "scientific_conclusion": "Current annotations cannot supervise severity without subjective label invention. Box area, region count, type, and open state may become input features only after humans create a documented severity ground truth.",
        "required_next_action": "Create and validate a manual ordinal severity annotation policy, run a double-annotated pilot, measure agreement, adjudicate disagreements, and freeze a new group-controlled dataset version before modeling.",
    }
    SUPPORT_PATH.write_text(json.dumps(support, indent=2), encoding="utf-8")

    print("\nPhase 14 annotation-support audit complete.")
    print("Explicit severity labels: NO")
    print("Reliable severity derivable without guessing: NO")
    print("Manual annotation required: YES")
    print(f"Development boxes audited: {len(records)} (train + validation only)")
    print("No test images or labels were accessed, and no severity model was trained.")


if __name__ == "__main__":
    main()
