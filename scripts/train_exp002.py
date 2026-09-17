"""Train EXP-002: YOLO11n on ontology-corrected parcel_damage_v3."""

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
        project=str(ROOT / "runs"),
        name="EXP-002_yolo11n_ontology_corrected",
        exist_ok=False,
        plots=True,
        val=True,
        save=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
