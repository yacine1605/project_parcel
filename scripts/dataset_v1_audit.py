"""Read-only Phase 3 audit for the parcel_damage_v1 YOLO dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))
import matplotlib.pyplot as plt
import yaml
from PIL import Image, ImageDraw, ImageFont


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
SPLITS = ("train", "valid", "test")
SMALL_AREA = 0.01
LARGE_AREA = 0.25
TINY_AREA = 0.0001
HUGE_AREA = 0.90


def image_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def dhash(image: Image.Image, size: int = 8) -> int:
    gray = image.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    pixels = list(gray.getdata())
    value = 0
    for y in range(size):
        for x in range(size):
            value = (value << 1) | (pixels[y * (size + 1) + x] > pixels[y * (size + 1) + x + 1])
    return value


def parse_names(config: dict) -> tuple[int, list[str]]:
    nc = int(config["nc"])
    raw = config["names"]
    if isinstance(raw, list):
        names = [str(x) for x in raw]
    else:
        names = [str(raw.get(i, raw.get(str(i), f"class_{i}"))) for i in range(nc)]
    return nc, names


def issue(kind: str, split: str, path: Path, line: int | None = None, detail: str = "") -> dict:
    result = {"type": kind, "split": split, "file": str(path)}
    if line is not None:
        result["line"] = line
    if detail:
        result["detail"] = detail
    return result


def audit(dataset: Path) -> tuple[dict, dict[str, list[dict]]]:
    config = yaml.safe_load((dataset / "data.yaml").read_text(encoding="utf-8"))
    nc, names = parse_names(config)
    stats: dict = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset.resolve()), "dataset_version": dataset.name,
        "data_yaml": config, "nc": nc, "names": names, "splits": {},
        "class_totals": {}, "bbox": {}, "duplicates": {}, "issues": [],
    }
    records: dict[str, list[dict]] = {s: [] for s in SPLITS}
    class_instances = Counter()
    class_images = Counter()
    all_areas: dict[int, list[float]] = defaultdict(list)
    exact: dict[str, list[tuple[str, str]]] = defaultdict(list)
    perceptual: list[tuple[str, str, int]] = []

    for split in SPLITS:
        images_dir, labels_dir = dataset / split / "images", dataset / split / "labels"
        images = image_files(images_dir) if images_dir.exists() else []
        labels = sorted(labels_dir.glob("*.txt")) if labels_dir.exists() else []
        image_by_stem, label_by_stem = {p.stem: p for p in images}, {p.stem: p for p in labels}
        missing = sorted(set(image_by_stem) - set(label_by_stem))
        orphan = sorted(set(label_by_stem) - set(image_by_stem))
        empty = [p for p in labels if not p.read_text(encoding="utf-8-sig").strip()]
        split_instances, split_class_images = Counter(), Counter()
        multi_class = zero_box = corrupt = 0
        dimensions, split_areas = Counter(), []

        for stem in missing:
            stats["issues"].append(issue("image_without_label", split, image_by_stem[stem]))
        for stem in orphan:
            stats["issues"].append(issue("label_without_image", split, label_by_stem[stem]))

        for image_path in images:
            rec = {"split": split, "image": str(image_path), "stem": image_path.stem, "boxes": []}
            try:
                with Image.open(image_path) as im:
                    im.verify()
                with Image.open(image_path) as im:
                    width, height = im.size
                    dimensions[f"{width}x{height}"] += 1
                    rec.update(width=width, height=height, aspect_ratio=width / height)
                    digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
                    exact[digest].append((split, str(image_path)))
                    perceptual.append((split, str(image_path), dhash(im)))
            except Exception as exc:
                corrupt += 1
                stats["issues"].append(issue("corrupt_image", split, image_path, detail=str(exc)))
                records[split].append(rec)
                continue

            label_path = label_by_stem.get(image_path.stem)
            if label_path and label_path.read_text(encoding="utf-8-sig").strip():
                seen_rows = set()
                for line_no, raw in enumerate(label_path.read_text(encoding="utf-8-sig").splitlines(), 1):
                    row = raw.strip()
                    if not row:
                        continue
                    if row in seen_rows:
                        stats["issues"].append(issue("duplicate_annotation_row", split, label_path, line_no, row))
                    seen_rows.add(row)
                    parts = row.split()
                    if len(parts) != 5:
                        stats["issues"].append(issue("malformed_row", split, label_path, line_no, f"expected 5 values, got {len(parts)}"))
                        continue
                    try:
                        cls_float, x, y, w, h = map(float, parts)
                    except ValueError:
                        stats["issues"].append(issue("non_numeric_value", split, label_path, line_no, row))
                        continue
                    cls = int(cls_float)
                    if cls_float != cls or not 0 <= cls < nc:
                        stats["issues"].append(issue("invalid_class_id", split, label_path, line_no, parts[0]))
                        continue
                    if any(v < 0 for v in (x, y, w, h)):
                        stats["issues"].append(issue("negative_coordinate", split, label_path, line_no, row))
                    if any(v > 1 for v in (x, y, w, h)):
                        stats["issues"].append(issue("value_above_one", split, label_path, line_no, row))
                    if w <= 0:
                        stats["issues"].append(issue("non_positive_width", split, label_path, line_no, str(w)))
                    if h <= 0:
                        stats["issues"].append(issue("non_positive_height", split, label_path, line_no, str(h)))
                    if x - w / 2 < 0 or x + w / 2 > 1 or y - h / 2 < 0 or y + h / 2 > 1:
                        stats["issues"].append(issue("box_outside_image", split, label_path, line_no, row))
                    area = w * h
                    if area < TINY_AREA:
                        stats["issues"].append(issue("suspicious_tiny_box", split, label_path, line_no, f"area={area:.8f}"))
                    if area > HUGE_AREA:
                        stats["issues"].append(issue("suspicious_huge_box", split, label_path, line_no, f"area={area:.6f}"))
                    rec["boxes"].append({"class_id": cls, "x": x, "y": y, "w": w, "h": h, "area": area})
                    split_instances[cls] += 1
                    class_instances[cls] += 1
                    split_areas.append(area)
                    all_areas[cls].append(area)
            classes_here = {b["class_id"] for b in rec["boxes"]}
            for cls in classes_here:
                split_class_images[cls] += 1
                class_images[cls] += 1
            if len(classes_here) > 1:
                multi_class += 1
            if not rec["boxes"]:
                zero_box += 1
            records[split].append(rec)

        stats["splits"][split] = {
            "images": len(images), "label_files": len(labels),
            "images_without_labels": len(missing), "labels_without_images": len(orphan),
            "empty_label_files": len(empty), "non_empty_label_files": len(labels) - len(empty),
            "corrupt_images": corrupt, "instances": sum(split_instances.values()),
            "instances_per_class": {names[i]: split_instances[i] for i in range(nc)},
            "images_per_class": {names[i]: split_class_images[i] for i in range(nc)},
            "multi_class_images": multi_class, "zero_box_images": zero_box,
            "average_boxes_per_image": sum(split_instances.values()) / len(images) if images else 0,
            "common_resolutions": [{"resolution": k, "count": v} for k, v in dimensions.most_common(15)],
            "width_min": min((r.get("width", math.inf) for r in records[split]), default=None),
            "width_max": max((r.get("width", 0) for r in records[split]), default=None),
            "height_min": min((r.get("height", math.inf) for r in records[split]), default=None),
            "height_max": max((r.get("height", 0) for r in records[split]), default=None),
        }

    total_instances = sum(class_instances.values())
    stats["class_totals"] = {
        names[i]: {"instances": class_instances[i], "images": class_images[i],
                   "percent_instances": 100 * class_instances[i] / total_instances if total_instances else 0}
        for i in range(nc)
    }
    stats["bbox"] = {
        names[i]: {
            "count": len(all_areas[i]),
            "mean_relative_area": sum(all_areas[i]) / len(all_areas[i]) if all_areas[i] else None,
            "median_relative_area": sorted(all_areas[i])[len(all_areas[i]) // 2] if all_areas[i] else None,
            "min_relative_area": min(all_areas[i], default=None), "max_relative_area": max(all_areas[i], default=None),
            "small": sum(a < SMALL_AREA for a in all_areas[i]),
            "medium": sum(SMALL_AREA <= a < LARGE_AREA for a in all_areas[i]),
            "large": sum(a >= LARGE_AREA for a in all_areas[i]),
        } for i in range(nc)
    }

    exact_groups = [v for v in exact.values() if len(v) > 1]
    exact_cross = [g for g in exact_groups if len({x[0] for x in g}) > 1]
    exact_pairs = {tuple(sorted((a[1], b[1]))) for g in exact_groups for a in g for b in g if a < b}
    near_cross = []
    for i, a in enumerate(perceptual):
        for b in perceptual[i + 1:]:
            if a[0] != b[0]:
                distance = hamming(a[2], b[2])
                pair = tuple(sorted((a[1], b[1])))
                if distance <= 5 and pair not in exact_pairs:
                    near_cross.append({"a": a[1], "a_split": a[0], "b": b[1], "b_split": b[0], "hamming_distance": distance})
    stats["duplicates"] = {
        "exact_duplicate_groups": [[{"split": s, "file": p} for s, p in g] for g in exact_groups],
        "exact_cross_split_groups": [[{"split": s, "file": p} for s, p in g] for g in exact_cross],
        "near_cross_split_pairs_threshold_5": near_cross,
        "perceptual_hash": "64-bit dHash", "near_duplicate_threshold": 5,
    }
    serious = {"corrupt_image", "malformed_row", "non_numeric_value", "invalid_class_id", "negative_coordinate",
               "value_above_one", "non_positive_width", "non_positive_height", "box_outside_image"}
    stats["freeze_decision"] = {
        "ready": not any(x["type"] in serious for x in stats["issues"]) and not exact_cross and not near_cross,
        "blocking_reasons": [],
    }
    if any(x["type"] in serious for x in stats["issues"]):
        stats["freeze_decision"]["blocking_reasons"].append("Invalid/corrupt images or annotations were found.")
    if exact_cross:
        stats["freeze_decision"]["blocking_reasons"].append("Exact cross-split duplicate leakage was found.")
    if near_cross:
        stats["freeze_decision"]["blocking_reasons"].append("Potential near-duplicate cross-split leakage requires visual review.")
    return stats, records


def create_figures(stats: dict, records: dict[str, list[dict]], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    names = stats["names"]
    counts = [stats["class_totals"][n]["instances"] for n in names]
    fig, ax = plt.subplots(figsize=(9, 5)); ax.bar(names, counts); ax.set_ylabel("Bounding-box instances")
    ax.set_title("Class distribution (all splits)"); ax.tick_params(axis="x", rotation=25); fig.tight_layout()
    fig.savefig(output / "class_distribution.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, name in enumerate(names):
        areas = [b["area"] for rows in records.values() for r in rows for b in r["boxes"] if b["class_id"] == i]
        if areas: ax.hist(areas, bins=30, alpha=.45, label=name)
    ax.set_xlabel("Relative box area (width × height)"); ax.set_ylabel("Instances"); ax.set_title("Bounding-box area distribution")
    ax.legend(); fig.tight_layout(); fig.savefig(output / "bbox_area_distribution.png", dpi=160); plt.close(fig)

    all_rows = [r for rows in records.values() for r in rows if r.get("width")]
    fig, ax = plt.subplots(figsize=(7, 6)); ax.scatter([r["width"] for r in all_rows], [r["height"] for r in all_rows], s=8, alpha=.4)
    ax.set_xlabel("Width (px)"); ax.set_ylabel("Height (px)"); ax.set_title("Image resolution distribution"); fig.tight_layout()
    fig.savefig(output / "image_resolutions.png", dpi=160); plt.close(fig)

    for class_id, name in enumerate(names):
        candidates = [r for rows in records.values() for r in rows if any(b["class_id"] == class_id for b in r["boxes"])]
        draw_grid(candidates[:20], names, output / f"samples_{class_id}_{name}.jpg", f"Class: {name}")
    special = {
        "very_small_damage": sorted(all_rows, key=lambda r: min((b["area"] for b in r["boxes"]), default=2))[:20],
        "large_damage": sorted(all_rows, key=lambda r: max((b["area"] for b in r["boxes"]), default=-1), reverse=True)[:20],
        "multiple_damages": sorted((r for r in all_rows if len(r["boxes"]) > 1), key=lambda r: len(r["boxes"]), reverse=True)[:20],
    }
    for title, rows in special.items():
        draw_grid(rows, names, output / f"samples_{title}.jpg", title.replace("_", " ").title())


def draw_grid(rows: list[dict], names: list[str], destination: Path, title: str) -> None:
    if not rows: return
    thumb, columns = 280, 4
    rows_n = math.ceil(len(rows) / columns)
    canvas = Image.new("RGB", (columns * thumb, rows_n * thumb + 34), "white")
    draw = ImageDraw.Draw(canvas); draw.text((8, 8), title, fill="black", font=ImageFont.load_default())
    colors = ("#ff3b30", "#34c759", "#007aff", "#ff9500", "#af52de")
    for index, rec in enumerate(rows):
        with Image.open(rec["image"]) as raw:
            im = raw.convert("RGB"); im.thumbnail((thumb, thumb - 34))
        tile = Image.new("RGB", (thumb, thumb), "#eeeeee"); xoff = (thumb - im.width) // 2; yoff = 24 + (thumb - 24 - im.height) // 2
        tile.paste(im, (xoff, yoff)); td = ImageDraw.Draw(tile)
        for box in rec["boxes"]:
            x1 = xoff + (box["x"] - box["w"] / 2) * im.width; y1 = yoff + (box["y"] - box["h"] / 2) * im.height
            x2 = xoff + (box["x"] + box["w"] / 2) * im.width; y2 = yoff + (box["y"] + box["h"] / 2) * im.height
            color = colors[box["class_id"] % len(colors)]; td.rectangle((x1, y1, x2, y2), outline=color, width=3)
            td.text((max(0, x1), max(0, y1 - 12)), names[box["class_id"]], fill=color, font=ImageFont.load_default())
        td.text((4, 4), rec["split"], fill="black", font=ImageFont.load_default())
        canvas.paste(tile, ((index % columns) * thumb, 34 + (index // columns) * thumb))
    canvas.save(destination, quality=92)


def write_report(stats: dict, destination: Path) -> None:
    lines = ["# Parcel Damage Dataset v1 — Phase 3 Audit", "", f"Generated: `{stats['generated_at_utc']}`",
             f"Dataset: `{stats['dataset_path']}`", "", "## Freeze decision", "",
             f"**{'READY TO FREEZE' if stats['freeze_decision']['ready'] else 'NOT READY — STOP BEFORE TRAINING'}**", ""]
    lines += [f"- {x}" for x in stats["freeze_decision"]["blocking_reasons"]] or ["No automated blocking issue was found; mandatory human visual review is still required."]
    lines += ["", "## Classes", "", f"`nc: {stats['nc']}`", "", "| ID | Name | Instances | Images | Instance share |",
              "|---:|---|---:|---:|---:|"]
    for i, name in enumerate(stats["names"]):
        c = stats["class_totals"][name]; lines.append(f"| {i} | {name} | {c['instances']} | {c['images']} | {c['percent_instances']:.2f}% |")
    lines += ["", "## Split validation", "", "| Split | Images | Labels | Missing | Orphan | Empty | Corrupt | Boxes | Zero-box | Multi-class | Avg boxes/image |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, s in stats["splits"].items():
        lines.append(f"| {name} | {s['images']} | {s['label_files']} | {s['images_without_labels']} | {s['labels_without_images']} | {s['empty_label_files']} | {s['corrupt_images']} | {s['instances']} | {s['zero_box_images']} | {s['multi_class_images']} | {s['average_boxes_per_image']:.3f} |")
    lines += ["", "## Bounding boxes", "", "Small `<1%`, medium `1–25%`, large `≥25%` of image area.", "",
              "| Class | Mean area | Median area | Min | Max | Small | Medium | Large |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, b in stats["bbox"].items():
        fmt = lambda x: "n/a" if x is None else f"{x:.6f}"
        lines.append(f"| {name} | {fmt(b['mean_relative_area'])} | {fmt(b['median_relative_area'])} | {fmt(b['min_relative_area'])} | {fmt(b['max_relative_area'])} | {b['small']} | {b['medium']} | {b['large']} |")
    lines += ["", "## Image resolutions", ""]
    for split, s in stats["splits"].items():
        common = ", ".join(f"{x['resolution']} ({x['count']})" for x in s["common_resolutions"][:8])
        lines.append(f"- **{split}:** width {s['width_min']}–{s['width_max']} px; height {s['height_min']}–{s['height_max']} px. Common: {common}")
    d = stats["duplicates"]
    lines += ["", "## Leakage check", "", f"- Exact duplicate groups: {len(d['exact_duplicate_groups'])}",
              f"- Exact cross-split groups: {len(d['exact_cross_split_groups'])}",
              f"- Potential near-duplicate cross-split pairs (64-bit dHash distance ≤5, excluding exact pairs): {len(d['near_cross_split_pairs_threshold_5'])}",
              "- Near-duplicate matches are candidates and require visual review; they are not automatically deleted.", "", "## Annotation issues", ""]
    by_type = Counter(x["type"] for x in stats["issues"])
    if by_type:
        lines += ["| Type | Count |", "|---|---:|"] + [f"| {k} | {v} |" for k, v in sorted(by_type.items())]
    else: lines.append("No syntax or integrity issues detected.")
    lines += ["", "Full file-level findings are preserved in `dataset_v1_stats.json`.", "", "## Figures", "",
              "- `figures/class_distribution.png`", "- `figures/bbox_area_distribution.png`", "- `figures/image_resolutions.png`",
              "- Per-class and special-case labeled sample grids under `figures/`.", "", "## Known limitations", "",
              "- The automated audit cannot verify semantic label correctness, lighting difficulty, background clutter, or package orientation; sample grids require human inspection.",
              "- dHash is a screening method. Flagged near duplicates need visual confirmation, and visually similar images can evade it.",
              "- `minor_damage` may be broader than the intended target ontology and should be confirmed visually before training."]
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--dataset", type=Path, required=True); parser.add_argument("--reports", type=Path, default=Path("reports"))
    args = parser.parse_args(); stats, records = audit(args.dataset)
    args.reports.mkdir(parents=True, exist_ok=True); create_figures(stats, records, args.reports / "figures")
    (args.reports / "dataset_v1_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(stats, args.reports / "dataset_v1_report.md")
    print(json.dumps({"freeze_decision": stats["freeze_decision"], "splits": stats["splits"], "classes": stats["class_totals"], "issue_counts": Counter(x["type"] for x in stats["issues"]), "exact_cross_split": len(stats["duplicates"]["exact_cross_split_groups"]), "near_cross_split": len(stats["duplicates"]["near_cross_split_pairs_threshold_5"])}, indent=2))


if __name__ == "__main__": main()
