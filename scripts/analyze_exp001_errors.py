"""Reproducible validation error analysis for EXP-001.

This script never writes inside the frozen dataset. It evaluates the best EXP-001
checkpoint on parcel_damage_v2/valid, exports machine-readable tables, annotated
failure examples, plots, and reports/EXP-001_error_analysis.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "datasets" / "processed" / "parcel_damage_v2"
DEFAULT_WEIGHTS = ROOT / "runs" / "EXP-001_yolo11n_baseline" / "weights" / "best.pt"
DEFAULT_OUTPUT = ROOT / "runs" / "EXP-001_error_analysis"
DEFAULT_REPORT = ROOT / "reports" / "EXP-001_error_analysis.md"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
COLORS = {"gt": (0, 220, 0), "tp": (255, 160, 0), "fp": (0, 0, 255), "low": (0, 220, 255)}


@dataclass
class Box:
    cls: int
    xyxy: np.ndarray
    conf: float = 1.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--device", default="0", help="Ultralytics device, e.g. 0 or cpu")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25, help="Operating threshold for TP/FP/FN")
    p.add_argument("--low-conf", type=float, default=0.05, help="Floor for low-confidence review")
    p.add_argument("--iou", type=float, default=0.50, help="TP matching IoU")
    p.add_argument("--localization-iou", type=float, default=0.10)
    p.add_argument("--max-examples", type=int, default=12, help="Images exported per review category")
    return p.parse_args()


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = np.maximum(a[:2], b[:2])
    x2, y2 = np.minimum(a[2:], b[2:])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (aa + bb - inter + 1e-9)


def load_gt(label_path: Path, width: int, height: int) -> list[Box]:
    boxes: list[Box] = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = [float(x) for x in line.split()]
        if len(values) < 5:
            continue
        cls = int(values[0])
        # Frozen V2 is expected to be detection format. Supporting polygon rows here
        # makes the analysis robust without changing the source annotations.
        if len(values) == 5:
            cx, cy, bw, bh = values[1:]
            x1, y1, x2, y2 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
        else:
            pts = np.asarray(values[1:], dtype=float).reshape(-1, 2)
            x1, y1 = pts.min(axis=0)
            x2, y2 = pts.max(axis=0)
        boxes.append(Box(cls, np.array([x1 * width, y1 * height, x2 * width, y2 * height])))
    return boxes


def greedy_match(gt: list[Box], pred: list[Box], threshold: float) -> tuple[list[tuple[int, int, float]], set[int], set[int]]:
    candidates = []
    for pi, p in enumerate(pred):
        for gi, g in enumerate(gt):
            if p.cls == g.cls:
                score = iou(p.xyxy, g.xyxy)
                if score >= threshold:
                    candidates.append((p.conf, score, pi, gi))
    used_p: set[int] = set()
    used_g: set[int] = set()
    matches = []
    for _, score, pi, gi in sorted(candidates, reverse=True):
        if pi not in used_p and gi not in used_g:
            used_p.add(pi); used_g.add(gi)
            matches.append((pi, gi, score))
    return matches, used_p, used_g


def metric_rows(val_metrics, names: dict[int, str], counts: Counter, totals: dict[int, Counter]) -> list[dict]:
    rows = []
    present = list(map(int, val_metrics.box.ap_class_index.tolist()))
    by_cls = {cls: val_metrics.box.class_result(i) for i, cls in enumerate(present)}
    for cls, name in names.items():
        p, r, ap50, ap = by_cls.get(cls, (float("nan"),) * 4)
        tp, fp, fn = totals[cls]["tp"], totals[cls]["fp"], totals[cls]["fn"]
        rows.append({"class_id": cls, "class": name, "ground_truth": counts[cls], "tp": tp, "fp": fp,
                     "fn": fn, "precision": tp / (tp + fp) if tp + fp else 0.0,
                     "recall": tp / (tp + fn) if tp + fn else 0.0,
                     "ap50": float(ap50), "ap50_95": float(ap),
                     "ultralytics_best_f1_precision": float(p), "ultralytics_best_f1_recall": float(r)})
    return rows


def draw_example(image_path: Path, gt: list[Box], pred: list[Box], names: dict[int, str], title: str, out: Path,
                 focus_gt: int | None = None, focus_pred: int | None = None) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        return
    for gi, box in enumerate(gt):
        color = COLORS["gt"]
        thick = 4 if gi == focus_gt else 2
        q = box.xyxy.astype(int)
        cv2.rectangle(image, tuple(q[:2]), tuple(q[2:]), color, thick)
        cv2.putText(image, f"GT {names[box.cls]}", (q[0], max(18, q[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, .52, color, 2)
    for pi, box in enumerate(pred):
        color = COLORS["low"] if box.conf < 0.25 else (COLORS["tp"] if any(iou(box.xyxy, g.xyxy) >= .5 and box.cls == g.cls for g in gt) else COLORS["fp"])
        thick = 4 if pi == focus_pred else 2
        q = box.xyxy.astype(int)
        cv2.rectangle(image, tuple(q[:2]), tuple(q[2:]), color, thick)
        cv2.putText(image, f"P {names[box.cls]} {box.conf:.2f}", (q[0], min(image.shape[0] - 5, q[3] + 18)), cv2.FONT_HERSHEY_SIMPLEX, .52, color, 2)
    cv2.rectangle(image, (0, 0), (image.shape[1], 34), (25, 25, 25), -1)
    cv2.putText(image, title, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, .62, (255, 255, 255), 2)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), image)


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else float("nan")


def main() -> None:
    args = parse_args()
    args.data_dir = args.data_dir.resolve(); args.weights = args.weights.resolve()
    args.output_dir = args.output_dir.resolve(); args.report = args.report.resolve()
    if args.data_dir != DEFAULT_DATA.resolve():
        print(f"Using explicitly supplied dataset: {args.data_dir}")
    for required in (args.data_dir / "data.yaml", args.weights, args.data_dir / "valid" / "images", args.data_dir / "valid" / "labels"):
        if not required.exists():
            raise FileNotFoundError(required)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "Ultralytics"))
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
    from ultralytics import YOLO

    config = yaml.safe_load((args.data_dir / "data.yaml").read_text(encoding="utf-8"))
    raw_names = config["names"]
    names = {int(k): str(v) for k, v in raw_names.items()} if isinstance(raw_names, dict) else dict(enumerate(raw_names))
    compressed_id = next(k for k, v in names.items() if v.lower() == "compressed")
    images = sorted(p for p in (args.data_dir / "valid" / "images").iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.weights))
    val_metrics = model.val(data=str(args.data_dir / "data.yaml"), split="val", imgsz=args.imgsz,
                            batch=args.batch, device=args.device, workers=4, conf=0.001, iou=0.7,
                            plots=False, save=False, verbose=False, project=str(args.output_dir.parent),
                            name="_metrics", exist_ok=True)
    results = model.predict(source=[str(p) for p in images], imgsz=args.imgsz, conf=args.low_conf,
                            iou=0.7, max_det=300, device=args.device, stream=True, verbose=False)

    gt_counts: Counter = Counter(); totals: dict[int, Counter] = defaultdict(Counter)
    size_values: dict[int, list[float]] = defaultdict(list)
    size_pixels_640: dict[int, list[float]] = defaultdict(list)
    events: list[dict] = []; category_events: dict[str, list[dict]] = defaultdict(list)
    confusion: Counter = Counter(); low_conf_count = 0

    for image_path, result in zip(images, results, strict=True):
        h, w = result.orig_shape
        gt = load_gt(args.data_dir / "valid" / "labels" / f"{image_path.stem}.txt", w, h)
        all_pred = [Box(int(c), np.asarray(xy, dtype=float), float(cf)) for xy, cf, c in zip(
            result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy())]
        pred = [p for p in all_pred if p.conf >= args.conf]
        for g in gt:
            gt_counts[g.cls] += 1
            area_ratio = max(0., g.xyxy[2] - g.xyxy[0]) * max(0., g.xyxy[3] - g.xyxy[1]) / (w * h)
            size_values[g.cls].append(area_ratio)
            size_pixels_640[g.cls].append(area_ratio * args.imgsz * args.imgsz)
        matches, used_p, used_g = greedy_match(gt, pred, args.iou)
        for pi, gi, score in matches:
            totals[gt[gi].cls]["tp"] += 1
            if gt[gi].cls == compressed_id:
                category_events["true_positives"].append({"image": image_path, "gt": gt, "pred": all_pred, "gi": gi,
                                                          "pi": all_pred.index(pred[pi]), "score": score, "rank": pred[pi].conf})
        for gi, g in enumerate(gt):
            if gi not in used_g:
                totals[g.cls]["fn"] += 1
        for pi, p in enumerate(pred):
            if pi not in used_p:
                totals[p.cls]["fp"] += 1

        # Compressed GT failure taxonomy is mutually exclusive: confusion, localization, background.
        for gi, g in enumerate(gt):
            if g.cls != compressed_id or gi in used_g:
                continue
            overlaps = [(iou(g.xyxy, p.xyxy), pi, p) for pi, p in enumerate(all_pred)]
            other = [x for x in overlaps if x[2].cls != compressed_id and x[0] >= args.iou and x[2].conf >= args.conf]
            local = [x for x in overlaps if x[2].cls == compressed_id and args.localization_iou <= x[0] < args.iou and x[2].conf >= args.conf]
            low = [x for x in overlaps if x[2].cls == compressed_id and x[0] >= args.iou and args.low_conf <= x[2].conf < args.conf]
            if other:
                score, pi, p = max(other); category = "confused_with_other_classes"; confusion[names[p.cls]] += 1
            elif local:
                score, pi, p = max(local); category = "localization_failures"
            elif low:
                score, pi, p = max(low, key=lambda x: x[2].conf); category = "low_confidence_detections"; low_conf_count += 1
            else:
                score, pi = 0.0, None; category = "compressed_to_background"
            event = {"image": image_path, "gt": gt, "pred": all_pred, "gi": gi, "pi": pi, "score": score,
                     "rank": (all_pred[pi].conf if pi is not None else 0.0), "category": category}
            category_events[category].append(event); events.append(event)

        # Compressed false positives at the operating threshold.
        for pi, p in enumerate(pred):
            if p.cls == compressed_id and pi not in used_p:
                api = all_pred.index(p)
                category_events["false_positives"].append({"image": image_path, "gt": gt, "pred": all_pred,
                                                            "gi": None, "pi": api, "score": max([iou(p.xyxy, g.xyxy) for g in gt] or [0]),
                                                            "rank": p.conf})

    rows = metric_rows(val_metrics, names, gt_counts, totals)
    with (args.output_dir / "per_class_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)

    size_rows = []
    for cls, name in names.items():
        vals = size_values[cls]
        size_rows.append({"class_id": cls, "class": name, "n": len(vals), "median_area_pct": percentile(vals, 50) * 100,
                          "mean_area_pct": float(np.mean(vals)) * 100, "p25_area_pct": percentile(vals, 25) * 100,
                          "p75_area_pct": percentile(vals, 75) * 100, "median_area_px_at_640": percentile(size_pixels_640[cls], 50),
                          "small_lt_1pct_pct": 100 * sum(v < .01 for v in vals) / len(vals) if vals else 0})
    with (args.output_dir / "object_size_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=size_rows[0].keys()); writer.writeheader(); writer.writerows(size_rows)

    plt.figure(figsize=(9, 5))
    plt.boxplot([np.asarray(size_values[c]) * 100 for c in names], tick_labels=[names[c] for c in names], showfliers=False)
    plt.ylabel("Bounding-box area (% of image)"); plt.title("Validation object-size distribution by class")
    plt.grid(axis="y", alpha=.25); plt.tight_layout(); plt.savefig(args.output_dir / "object_size_distribution.png", dpi=160); plt.close()

    exported = {}
    for category, items in category_events.items():
        # Most informative first: low confidence near threshold, worst TP IoU, otherwise highest confidence/overlap.
        reverse = category not in {"true_positives"}
        ordered = sorted(items, key=lambda e: e["rank"] if category != "true_positives" else -e["score"], reverse=reverse)
        selected = []; seen = set()
        for event in ordered:
            if event["image"] in seen: continue
            seen.add(event["image"]); selected.append(event)
            if len(selected) >= args.max_examples: break
        for n, event in enumerate(selected, 1):
            filename = f"{n:02d}_{event['image'].stem}.jpg"
            draw_example(event["image"], event["gt"], event["pred"], names, category.replace("_", " "),
                         args.output_dir / "annotated_examples" / category / filename, event.get("gi"), event.get("pi"))
        exported[category] = len(selected)

    compressed = next(r for r in rows if r["class_id"] == compressed_id)
    comp_size = next(r for r in size_rows if r["class_id"] == compressed_id)
    other_medians = [r["median_area_pct"] for r in size_rows if r["class_id"] != compressed_id]
    smaller_than_all = comp_size["median_area_pct"] < min(other_medians)
    taxonomy = {k: len(category_events[k]) for k in ("compressed_to_background", "confused_with_other_classes", "localization_failures", "low_confidence_detections")}
    dominant = max(taxonomy, key=taxonomy.get)
    fn_total = max(1, compressed["fn"])
    class_counts = [r["ground_truth"] for r in rows]
    compressed_underrepresented = compressed["ground_truth"] < float(np.median(class_counts)) * .5
    if smaller_than_all and comp_size["small_lt_1pct_pct"] >= 35:
        recommendation = "Test higher input resolution first because compressed objects are materially smaller in validation."
    elif compressed_underrepresented and (taxonomy["compressed_to_background"] + taxonomy["low_confidence_detections"]) / fn_total >= .40:
        recommendation = "Test class/data balancing with targeted compressed examples first; compressed is underrepresented and confidence/background misses are substantial."
    elif taxonomy["confused_with_other_classes"] / fn_total >= .25:
        recommendation = "Investigate compressed annotation consistency and class definitions before increasing capacity; class confusion is substantial."
    else:
        recommendation = "Test a larger YOLO model first; failures are mixed and do not show a dominant size, localization, or ontology signal."

    report_lines = [
        "# EXP-001 Error Analysis", "", "Status: **COMPLETE — validation-only decision analysis**", "",
        "## Scope and method", "",
        f"This analysis uses only the `{args.data_dir.relative_to(ROOT).as_posix()}/valid` split ({len(images)} images) and checkpoint `{args.weights.relative_to(ROOT).as_posix()}`. The held-out test set was not used for decisions. No dataset files were changed.", "",
        f"EXP-001 was YOLO11n, `imgsz={args.imgsz}`, 75 epochs, batch 16, seed 42. TP/FP/FN use confidence ≥ {args.conf:.2f}, class-aware greedy matching, and IoU ≥ {args.iou:.2f}. Precision and recall below are recomputed from those counts. AP uses Ultralytics' confidence sweep with its standard COCO-style IoU thresholds. Predictions down to {args.low_conf:.2f} are retained for failure diagnosis.", "",
        "## Per-class validation results", "",
        "| Class | GT | TP | FP | FN | Precision | Recall | AP50 | AP50–95 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        report_lines.append(f"| {r['class']} | {r['ground_truth']} | {r['tp']} | {r['fp']} | {r['fn']} | {r['precision']:.3f} | {r['recall']:.3f} | {r['ap50']:.3f} | {r['ap50_95']:.3f} |")
    report_lines += ["", "## Compressed error analysis", "",
                     f"At the fixed operating point, compressed has **{compressed['tp']} TP, {compressed['fp']} FP, and {compressed['fn']} FN** (precision {compressed['precision']:.3f}, recall {compressed['recall']:.3f}). Its AP50 is {compressed['ap50']:.3f}, but AP50–95 falls to {compressed['ap50_95']:.3f}, a gap of {compressed['ap50']-compressed['ap50_95']:.3f}.", "",
                     "Unmatched compressed ground truths were assigned once to this mutually exclusive taxonomy (priority: other-class confusion, localization, low confidence, then background):", "",
                     "| Failure mode | Count | Share of compressed FN |", "|---|---:|---:|",
                     f"| Compressed → background | {taxonomy['compressed_to_background']} | {taxonomy['compressed_to_background']/fn_total:.1%} |",
                     f"| Confused with another class | {taxonomy['confused_with_other_classes']} | {taxonomy['confused_with_other_classes']/fn_total:.1%} |",
                     f"| Localization failure (IoU {args.localization_iou:.2f}–{args.iou:.2f}) | {taxonomy['localization_failures']} | {taxonomy['localization_failures']/fn_total:.1%} |",
                     f"| Correct-class detection below confidence threshold | {taxonomy['low_confidence_detections']} | {taxonomy['low_confidence_detections']/fn_total:.1%} |", "",
                     f"Compressed false positives: **{len(category_events['false_positives'])}**. Other-class labels assigned to compressed GT at IoU ≥ {args.iou:.2f}: " + (", ".join(f"`{k}`={v}" for k, v in confusion.most_common()) or "none") + ".", "",
                     f"The dominant compressed FN mode is **{dominant.replace('_', ' ')}**. Annotated sets include TP, FP, background misses, confusion, localization, and low-confidence cases; green boxes are ground truth, orange correct detections, red errors, and yellow low-confidence detections.", "",
                     "## Object-size evidence", "", "Sizes are based on validation ground-truth bounding boxes relative to each original image.", "",
                     "| Class | Objects | Median area | IQR area | Median pixels at 640 | Area <1% |", "|---|---:|---:|---:|---:|---:|"]
    for r in size_rows:
        report_lines.append(f"| {r['class']} | {r['n']} | {r['median_area_pct']:.2f}% | {r['p25_area_pct']:.2f}–{r['p75_area_pct']:.2f}% | {r['median_area_px_at_640']:.0f} | {r['small_lt_1pct_pct']:.1f}% |")
    size_conclusion = ("Compressed has the smallest median box area of every class." if smaller_than_all else
                       "Compressed does not have the smallest median box area across classes.")
    report_lines += ["", f"**Conclusion:** {size_conclusion} Its median is {comp_size['median_area_pct']:.2f}% of image area, and {comp_size['small_lt_1pct_pct']:.1f}% of compressed boxes occupy less than 1% of the image. Therefore, the size evidence " + ("supports" if smaller_than_all else "does not by itself support") + " resolution as the primary explanation.", "",
                     "![Validation object-size distributions](../runs/EXP-001_error_analysis/object_size_distribution.png)", "",
                     "## EXP-002 recommendation", "", f"**{recommendation}**", "",
                     "This is a one-variable-at-a-time recommendation grounded in validation evidence. Keep the held-out test split untouched until an EXP-002 configuration is selected using validation and trained once.", "",
                     "## Reproducibility and outputs", "", "Run from the repository root:", "",
                     "```powershell", "$env:YOLO_CONFIG_DIR=(Join-Path (Get-Location) 'Ultralytics')", "$env:MPLCONFIGDIR=(Join-Path (Get-Location) '.matplotlib-cache')", ".\\venv\\Scripts\\python.exe .\\scripts\\analyze_exp001_errors.py", "```", "",
                     "Machine-readable outputs: `runs/EXP-001_error_analysis/per_class_metrics.csv`, `object_size_summary.csv`, and `analysis_summary.json`. Annotated examples are under `runs/EXP-001_error_analysis/annotated_examples/`."]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report_text = "\n".join(report_lines) + "\n"
    try:
        args.report.write_text(report_text, encoding="utf-8")
        written_report = args.report
    except PermissionError:
        # Some managed Windows sandboxes protect pre-existing report directories.
        # Preserve the complete output beside the analysis artifacts in that case.
        written_report = args.output_dir / args.report.name
        written_report.write_text(report_text, encoding="utf-8")
        print(f"Warning: could not write {args.report}; wrote fallback {written_report}")
    summary = {"dataset": str(args.data_dir), "split": "valid", "checkpoint": str(args.weights),
               "settings": vars(args) | {"data_dir": str(args.data_dir), "weights": str(args.weights), "output_dir": str(args.output_dir), "report": str(args.report)},
               "per_class": rows, "compressed_failure_taxonomy": taxonomy, "compressed_confusions": dict(confusion),
               "object_sizes": size_rows, "compressed_is_smallest_median": smaller_than_all,
               "recommendation": recommendation, "annotated_examples_exported": exported}
    (args.output_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {written_report}")


if __name__ == "__main__":
    main()
