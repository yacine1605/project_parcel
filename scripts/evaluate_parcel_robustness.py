"""
# ============================================================
# What are we doing in this script?
# ============================================================
#
# This script evaluates the EXISTING frozen MobileNet + YOLO prototype on a new,
# human-labeled controlled-capture dataset. It never trains, recalibrates, or
# changes either model. If the capture manifest is empty, it stops instead of
# inventing robustness numbers.
#
# Classification metrics answer how often the image-level damaged/intact result
# is correct. YOLO rows describe known localized detections. True detector mAP
# requires human bounding-box ground truth; image-level expected class names are
# not a substitute for localization annotations.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import sklearn
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from prototype.inspection_pipeline import inspect_parcel_image, load_prototype_models, open_supported_image  # noqa: E402


DEFAULT_MANIFEST = PROJECT_ROOT / "datasets" / "robustness" / "parcel_robustness_v1" / "manifest.csv"
OUTPUT_ROOT = PROJECT_ROOT / "models" / "robustness" / "phase19_20_artifacts"
SCENARIO_FIELDS = ["lighting", "view_angle", "distance_cm", "parcel_appearance", "damage_appearance", "damage_size"]


def binary_metrics(rows: list[dict]) -> dict:
    """Calculate transparent damaged-positive confusion-matrix metrics."""
    tp = sum(row["true_binary_label"] == "damaged" and row["classifier_label"] == "damaged" for row in rows)
    tn = sum(row["true_binary_label"] == "intact" and row["classifier_label"] == "intact" for row in rows)
    fp = sum(row["true_binary_label"] == "intact" and row["classifier_label"] == "damaged" for row in rows)
    fn = sum(row["true_binary_label"] == "damaged" and row["classifier_label"] == "intact" for row in rows)
    safe = lambda numerator, denominator: numerator / denominator if denominator else None
    precision = safe(tp, tp + fp)
    recall = safe(tp, tp + fn)
    specificity = safe(tn, tn + fp)
    return {
        "n": len(rows), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": safe(tp + tn, len(rows)), "precision": precision,
        "recall": recall, "specificity": specificity,
        "f1": safe(2 * precision * recall, precision + recall) if precision is not None and recall is not None else None,
        "average_total_latency_ms": float(np.mean([row["total_latency_ms"] for row in rows])),
    }


def read_manifest(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise RuntimeError(
            "The robustness manifest contains no captures. Collect controlled, human-labeled images before evaluation."
        )
    for row in rows:
        if row["true_binary_label"] not in {"damaged", "intact"}:
            raise ValueError(f"Invalid binary label for {row['sample_id']}: {row['true_binary_label']}")
        image_path = PROJECT_ROOT / row["image_path"]
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing capture for {row['sample_id']}: {image_path}")
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate frozen models on controlled robustness captures.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    try:
        manifest_rows = read_manifest(args.manifest)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        # A short stop message is easier for a learner to understand than a
        # traceback. The non-zero exit still tells automation evaluation did not run.
        raise SystemExit(f"Robustness evaluation stopped: {error}") from None
    models = load_prototype_models()
    predictions = []
    annotated_root = OUTPUT_ROOT / "annotated"
    annotated_root.mkdir(parents=True, exist_ok=True)

    for index, row in enumerate(manifest_rows, start=1):
        image = open_supported_image(PROJECT_ROOT / row["image_path"])
        evidence, annotated = inspect_parcel_image(image, models)
        annotated_path = annotated_root / f"{row['sample_id']}.jpg"
        annotated.save(annotated_path)
        predicted_classes = sorted(set(evidence["yolo_damage_types"]))
        predictions.append({
            **row,
            "classifier_label": evidence["classifier_label"],
            "classifier_probability": evidence["classifier_probability"],
            "classifier_correct": evidence["classifier_label"] == row["true_binary_label"],
            "yolo_detection_count": evidence["yolo_detection_count"],
            "yolo_predicted_classes": ",".join(predicted_classes),
            "prototype_decision": evidence["prototype_decision"],
            "classifier_latency_ms": evidence["timing_ms"]["classifier_forward"],
            "yolo_latency_ms": evidence["timing_ms"]["yolo_inference"],
            "total_latency_ms": evidence["timing_ms"]["total_pipeline"],
            "annotated_image_path": annotated_path.relative_to(PROJECT_ROOT).as_posix(),
        })
        print(f"[{index}/{len(manifest_rows)}] {row['sample_id']} -> {evidence['prototype_decision']}")

    scenario_metrics = []
    for field in SCENARIO_FIELDS:
        groups: dict[str, list[dict]] = defaultdict(list)
        for row in predictions:
            groups[str(row[field])].append(row)
        for value, group_rows in sorted(groups.items()):
            scenario_metrics.append({"scenario_dimension": field, "scenario_value": value, **binary_metrics(group_rows)})

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_ROOT / "robustness_predictions.csv", predictions)
    write_csv(OUTPUT_ROOT / "scenario_metrics.csv", scenario_metrics)
    summary = {
        "status": "evaluated_on_controlled_captures",
        "dataset": "parcel_robustness_v1",
        "sample_count": len(predictions),
        "overall_classifier_metrics": binary_metrics(predictions),
        "detector_map_status": "not_computable_without_human_bounding_box_ground_truth",
        "models_modified": False,
        "classifier_threshold_modified": False,
        "versions": {"python": sys.version, "torch": torch.__version__, "scikit_learn": sklearn.__version__, "platform": platform.platform()},
        "generated_unix_time": time.time(),
    }
    (OUTPUT_ROOT / "robustness_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Robustness evaluation complete.")
    print(f"Samples: {len(predictions)}")
    print(f"Damaged recall: {summary['overall_classifier_metrics']['recall']}")
    print(f"Specificity: {summary['overall_classifier_metrics']['specificity']}")
    print("No model weight or threshold was changed.")


if __name__ == "__main__":
    main()
