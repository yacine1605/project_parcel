"""Run the four-stage parcel-inspection prototype on one local image."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .inspection_pipeline import inspect_parcel_image, load_prototype_models, open_supported_image, save_completed_inspection, save_evidence_json
except ImportError:
    from inspection_pipeline import inspect_parcel_image, load_prototype_models, open_supported_image, save_completed_inspection, save_evidence_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect one parcel image with independent MobileNet and YOLO models.")
    parser.add_argument("image", type=Path, help="JPG/PNG/BMP/WebP parcel image")
    parser.add_argument("--annotated-output", type=Path, default=Path("prototype_outputs/annotated_parcel.jpg"))
    parser.add_argument("--json-output", type=Path, default=Path("prototype_outputs/inspection_result.json"))
    parser.add_argument("--save-to-database", action="store_true", help="Persist the completed result after inference")
    parser.add_argument("--database", type=Path, default=Path("data/parcel_inspections.db"))
    parser.add_argument("--package-id", type=str, default=None, help="Real barcode/package ID when available; otherwise a prototype ID is generated")
    args = parser.parse_args()

    image = open_supported_image(args.image)
    models = load_prototype_models()
    evidence, annotated = inspect_parcel_image(image, models)
    args.annotated_output.parent.mkdir(parents=True, exist_ok=True)
    annotated.save(args.annotated_output, quality=94)
    save_evidence_json(evidence, args.json_output)
    inspection_id = None
    if args.save_to_database:
        inspection_id = save_completed_inspection(
            evidence=evidence,
            image_path=args.image,
            annotated_image_path=args.annotated_output,
            package_id=args.package_id,
            database_path=args.database,
        )

    print("\nParcel inspection complete.")
    if evidence["classifier_executed"]:
        print(f"Damage classifier: {evidence['classifier_label']} (damaged probability {evidence['classifier_probability']:.3f})")
    else:
        print("Damage classifier: not run")
    if evidence["damage_detector_executed"]:
        print(f"Damage detections: {evidence['yolo_detection_count']} — {evidence['yolo_damage_types'] or 'none'}")
    else:
        print("Damage detector: not run")
    print(f"Prototype decision: {evidence['prototype_decision']}")
    print(f"Total end-to-end pipeline time: {evidence['timing_ms']['total_pipeline']:.1f} ms")
    print(f"Annotated image: {args.annotated_output}")
    print(f"Evidence JSON: {args.json_output}")
    if inspection_id is not None:
        print(f"Saved SQLite inspection row ID: {inspection_id}")


if __name__ == "__main__":
    main()
