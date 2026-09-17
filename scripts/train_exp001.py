"""Train the Phase 4 EXP-001 YOLO11n detection baseline."""

from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    model = YOLO("yolo11n.pt")
    model.train(
        data=str(ROOT / "datasets" / "processed" / "parcel_damage_v2" / "data.yaml"),
        imgsz=640,
        epochs=75,
        batch=16,
        device=0,
        workers=4,
        pretrained=True,
        seed=42,
        deterministic=True,
        project=str(ROOT / "runs"),
        name="EXP-001_yolo11n_baseline",
        exist_ok=False,
        plots=True,
        val=True,
        save=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
