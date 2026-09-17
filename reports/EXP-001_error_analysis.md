# EXP-001 Error Analysis

Status: **COMPLETE — validation-only decision analysis**

## Scope and method

This analysis uses only the `datasets/processed/parcel_damage_v2/valid` split (444 images) and checkpoint `runs/EXP-001_yolo11n_baseline/weights/best.pt`. The held-out test set was not used for decisions. No dataset files were changed.

EXP-001 was YOLO11n, `imgsz=640`, 75 epochs, batch 16, seed 42. TP/FP/FN use confidence ≥ 0.25, class-aware greedy matching, and IoU ≥ 0.50. Precision and recall below are recomputed from those counts. AP uses Ultralytics' confidence sweep with its standard COCO-style IoU thresholds. Predictions down to 0.05 are retained for failure diagnosis.

## Per-class validation results

| Class | GT | TP | FP | FN | Precision | Recall | AP50 | AP50–95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| minor_damage | 34 | 15 | 15 | 19 | 0.500 | 0.441 | 0.459 | 0.284 |
| compressed | 42 | 12 | 21 | 30 | 0.364 | 0.286 | 0.520 | 0.188 |
| hole | 437 | 318 | 187 | 119 | 0.630 | 0.728 | 0.758 | 0.377 |
| wet | 168 | 120 | 67 | 48 | 0.642 | 0.714 | 0.743 | 0.391 |

## Compressed error analysis

At the fixed operating point, compressed has **12 TP, 21 FP, and 30 FN** (precision 0.364, recall 0.286). Its AP50 is 0.520, but AP50–95 falls to 0.188, a gap of 0.332.

Unmatched compressed ground truths were assigned once to this mutually exclusive taxonomy (priority: other-class confusion, localization, low confidence, then background):

| Failure mode | Count | Share of compressed FN |
|---|---:|---:|
| Compressed → background | 10 | 33.3% |
| Confused with another class | 9 | 30.0% |
| Localization failure (IoU 0.10–0.50) | 8 | 26.7% |
| Correct-class detection below confidence threshold | 3 | 10.0% |

Compressed false positives: **21**. Other-class labels assigned to compressed GT at IoU ≥ 0.50: `hole`=5, `minor_damage`=3, `wet`=1.

The dominant compressed FN mode is **compressed to background**, although it accounts for only one third of misses; confusion and localization are nearly as important. Annotated sets include TP, FP, background misses, confusion, localization, and low-confidence cases. Green boxes are ground truth, orange correct detections, red errors, and yellow low-confidence detections.

## Object-size evidence

Sizes are based on validation ground-truth bounding boxes relative to each original image.

| Class | Objects | Median area | IQR area | Median pixels at 640 | Area <1% |
|---|---:|---:|---:|---:|---:|
| minor_damage | 34 | 61.33% | 37.75–69.41% | 251197 | 0.0% |
| compressed | 42 | 35.66% | 24.31–49.34% | 146075 | 0.0% |
| hole | 437 | 3.54% | 1.83–8.20% | 14487 | 10.8% |
| wet | 168 | 11.48% | 5.76–29.92% | 47008 | 0.0% |

**Conclusion:** Compressed does not have the smallest median box area across classes. Its median is 35.66% of image area, and 0.0% of compressed boxes occupy less than 1% of the image. Therefore, the size evidence does not support higher resolution as the first response.

![Validation object-size distributions](../runs/EXP-001_error_analysis/object_size_distribution.png)

## EXP-002 recommendation

**Test class/data balancing with targeted compressed examples first.** Compressed is underrepresented (42 validation objects versus 437 holes), while confidence/background misses account for 43.3% of its false negatives. Because compressed boxes are already large, resolution is not the evidence-led first variable. Before training EXP-002, manually review the exported confusion and localization examples for class-definition or annotation inconsistency; if that review finds systematic ambiguity, annotation/class-definition investigation should supersede balancing.

Use a one-variable-at-a-time comparison against EXP-001 on validation. Keep the held-out test split untouched until the EXP-002 configuration is selected using validation and trained once.

## Reproducibility and outputs

Run from the repository root:

```powershell
$env:YOLO_CONFIG_DIR=(Join-Path (Get-Location) 'Ultralytics')
$env:MPLCONFIGDIR=(Join-Path (Get-Location) '.matplotlib-cache')
.\venv\Scripts\python.exe .\scripts\analyze_exp001_errors.py
```

Machine-readable outputs: `runs/EXP-001_error_analysis/per_class_metrics.csv`, `object_size_summary.csv`, and `analysis_summary.json`. Annotated examples are under `runs/EXP-001_error_analysis/annotated_examples/`.
