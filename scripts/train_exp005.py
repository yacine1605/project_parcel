"""Train EXP-005: YOLO11n with mild realistic geometric augmentation."""

from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    model = YOLO("yolo11n.pt")
    model.train(
        data=str(ROOT / "datasets" / "processed" / "parcel_damage_v3" / "data.yaml"),
        imgsz=640,
        epochs=75,
        batch=16,
        device=0,
        workers=4,
        pretrained=True,
        seed=42,
        deterministic=True,
        degrees=5.0,
        perspective=0.0005,
        project=str(ROOT / "runs"),
        name="EXP-005_yolo11n_mild_geometry_aug",
        exist_ok=False,
        plots=True,
        val=True,
        save=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
