"""Train EXP-003: YOLO11n with compressed images sampled at 3x exposure."""

from pathlib import Path

from ultralytics import YOLO

from prepare_exp003 import main as prepare_manifest


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    prepare_manifest()
    model = YOLO("yolo11n.pt")
    model.train(
        data=str(ROOT / "experiments" / "EXP-003" / "data.yaml"),
        imgsz=640,
        epochs=75,
        batch=16,
        device=0,
        workers=4,
        pretrained=True,
        seed=42,
        deterministic=True,
        project=str(ROOT / "runs"),
        name="EXP-003_yolo11n_compressed_x3",
        exist_ok=False,
        plots=True,
        val=True,
        save=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
