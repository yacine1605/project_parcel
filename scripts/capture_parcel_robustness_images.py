"""
# ============================================================
# What are we doing in this script?
# ============================================================
#
# This tool captures CONTROLLED webcam images for a new robustness dataset.
# It does not run, train, or modify a model. A human records the known parcel
# condition and the controlled camera scenario at capture time.
#
# A fair robustness study photographs the SAME physical parcel across conditions.
# For example, parcel P001 can be photographed under normal, bright, and dark
# lighting while its actual damaged/intact label remains unchanged.
#
# Run ``python scripts/capture_parcel_robustness_images.py --help`` first.
"""

from __future__ import annotations

import argparse
import csv
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "datasets" / "robustness" / "parcel_robustness_v1"
MANIFEST_PATH = DATASET_ROOT / "manifest.csv"
FIELDNAMES = [
    "sample_id", "image_path", "parcel_id", "true_binary_label",
    "expected_damage_classes", "lighting", "view_angle", "distance_cm",
    "parcel_appearance", "damage_appearance", "damage_size",
    "capture_timestamp_utc", "notes",
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture one controlled parcel robustness image.")
    parser.add_argument("--parcel-id", required=True, help="Stable ID for the physical parcel, e.g. P001.")
    parser.add_argument("--label", required=True, choices=["damaged", "intact"])
    parser.add_argument("--lighting", required=True, choices=["bright", "normal", "dark"])
    parser.add_argument("--view-angle", required=True, choices=["front", "side", "top", "45_degrees"])
    parser.add_argument("--distance-cm", required=True, type=float)
    parser.add_argument("--parcel-appearance", required=True, help="Example: brown_cardboard or printed_box.")
    parser.add_argument("--damage-appearance", default="none", help="Example: open_box, hole, compressed, or none.")
    parser.add_argument("--damage-size", default="not_applicable", choices=["small", "medium", "large", "not_applicable"])
    parser.add_argument("--expected-classes", default="", help="Comma-separated known YOLO classes; blank is allowed.")
    parser.add_argument("--notes", default="")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV webcam index; usually 0.")
    return parser.parse_args()


def initialize_manifest() -> None:
    """Create the CSV header once without overwriting earlier captures."""
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    if not MANIFEST_PATH.exists():
        with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as file:
            csv.DictWriter(file, fieldnames=FIELDNAMES).writeheader()


def main() -> None:
    args = parse_arguments()
    if args.distance_cm <= 0:
        raise ValueError("--distance-cm must be greater than zero.")
    initialize_manifest()

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError(f"Webcam {args.camera} could not be opened.")

    print("Live capture is open. Press SPACE to save one image or Q to cancel.")
    saved_frame = None
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("The webcam stopped returning frames.")
            cv2.putText(frame, "SPACE: capture   Q: cancel", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Controlled parcel robustness capture", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == 32:
                saved_frame = frame
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()

    if saved_frame is None:
        print("Capture cancelled. The manifest was not changed.")
        return

    timestamp = datetime.now(timezone.utc)
    sample_id = f"RB-{timestamp.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    relative_path = Path("datasets") / "robustness" / "parcel_robustness_v1" / "images" / f"{sample_id}.jpg"
    absolute_path = PROJECT_ROOT / relative_path
    absolute_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(absolute_path), saved_frame):
        raise RuntimeError(f"OpenCV could not save {absolute_path}")

    row = {
        "sample_id": sample_id,
        "image_path": relative_path.as_posix(),
        "parcel_id": args.parcel_id,
        "true_binary_label": args.label,
        "expected_damage_classes": args.expected_classes,
        "lighting": args.lighting,
        "view_angle": args.view_angle,
        "distance_cm": args.distance_cm,
        "parcel_appearance": args.parcel_appearance,
        "damage_appearance": args.damage_appearance,
        "damage_size": args.damage_size,
        "capture_timestamp_utc": timestamp.isoformat(),
        "notes": args.notes,
    }
    with MANIFEST_PATH.open("a", newline="", encoding="utf-8") as file:
        csv.DictWriter(file, fieldnames=FIELDNAMES).writerow(row)
    print(f"Saved {sample_id} to {absolute_path}")
    print("The human label describes the physical parcel; it was not generated by a model.")


if __name__ == "__main__":
    main()
