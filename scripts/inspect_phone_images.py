"""Run the parcel-state pipeline on one phone image or a folder of images."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prototype.inspection_pipeline import (
    SUPPORTED_EXTENSIONS, inspect_parcel_image, load_prototype_models,
    open_supported_image, save_evidence_json,
)


def discover_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if source.is_dir():
        return sorted(path for path in source.rglob("*") if path.suffix.lower() in SUPPORTED_EXTENSIONS)
    raise FileNotFoundError(f"Input does not exist: {source}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect one phone image or every supported image in a folder.")
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, default=Path("prototype_outputs/phone_images"))
    parser.add_argument("--debug", action="store_true", help="Save intermediate crops and overlays.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    images = discover_images(args.source)
    if not images:
        raise ValueError(f"No supported images found under: {args.source}")
    models = load_prototype_models()
    args.output.mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(images):
        image = open_supported_image(path)
        stem = f"{index:04d}_{path.stem}" if len(images) > 1 else path.stem
        result, annotated = inspect_parcel_image(
            image, models, debug=args.debug, debug_dir=args.output / "debug" / stem,
        )
        annotated.save(args.output / f"{stem}_annotated.jpg", quality=94)
        save_evidence_json(result, args.output / f"{stem}_result.json")
        print(f"{path}: {result['status']} — {len(result['parcels'])} parcel(s)")
        for parcel in result["parcels"]:
            print(f"  Parcel {parcel['parcel_index']}: {parcel['final_status']}")


if __name__ == "__main__":
    main()
