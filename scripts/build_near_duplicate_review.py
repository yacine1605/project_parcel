"""Build side-by-side evidence for near-duplicate candidates from the audit JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def fit(image: Image.Image, width: int, height: int) -> Image.Image:
    result = image.convert("RGB")
    result.thumbnail((width, height), Image.Resampling.LANCZOS)
    return result


def comparison(pair: dict, index: int, destination: Path) -> None:
    panel_width, image_height, header = 700, 620, 100
    canvas = Image.new("RGB", (panel_width * 2, image_height + header), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    distance = int(pair["hamming_distance"])
    score = 1 - distance / 64
    draw.text((12, 10), f"Candidate {index:02d} | dHash distance: {distance}/64 | similarity: {score:.4f}", fill="black", font=font)
    for side, key in enumerate(("a", "b")):
        path = Path(pair[key])
        label = f"{pair[key + '_split']} | {path.name}"
        draw.text((side * panel_width + 12, 36), label, fill="black", font=font)
        with Image.open(path) as raw:
            image = fit(raw, panel_width - 20, image_height - 10)
        x = side * panel_width + (panel_width - image.width) // 2
        y = header + (image_height - image.height) // 2
        canvas.paste(image, (x, y))
    canvas.save(destination, quality=94)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", type=Path, default=Path("reports/dataset_v1_stats.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/figures/near_duplicates"))
    args = parser.parse_args()
    stats = json.loads(args.stats.read_text(encoding="utf-8"))
    pairs = stats["duplicates"]["near_cross_split_pairs_threshold_5"]
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, pair in enumerate(pairs, 1):
        filename = f"candidate_{index:02d}.jpg"
        comparison(pair, index, args.output / filename)
        manifest.append({"candidate": index, **pair, "similarity": 1 - pair["hamming_distance"] / 64, "figure": filename})
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Created {len(manifest)} comparison panels in {args.output}")


if __name__ == "__main__":
    main()
