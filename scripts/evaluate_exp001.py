"""Evaluate EXP-001's best checkpoint on the frozen V2 test split."""

from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    model = YOLO(ROOT / "runs" / "EXP-001_yolo11n_baseline" / "weights" / "best.pt")
    model.val(
        data=str(ROOT / "datasets" / "processed" / "parcel_damage_v2" / "data.yaml"),
        split="test",
        imgsz=640,
        batch=16,
        device=0,
        workers=4,
        project=str(ROOT / "runs"),
        name="EXP-001_yolo11n_baseline_test",
        exist_ok=False,
        plots=True,
        save_json=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
