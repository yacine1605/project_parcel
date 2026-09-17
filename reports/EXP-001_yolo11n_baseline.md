# EXP-001 — YOLO11n Baseline

Status: **COMPLETE**

## Configuration

| Setting | Value |
|---|---|
| Model | YOLO11n pretrained |
| Dataset | `parcel_damage_v2` (frozen) |
| Image size | 640 |
| Epochs | 75 |
| Batch | 16 |
| Augmentation | Ultralytics default baseline |
| Optimizer | AdamW (selected by Ultralytics auto optimizer) |
| Seed | 42, deterministic mode enabled |
| GPU | NVIDIA GeForce RTX 5060, 8151 MiB |
| Software | Ultralytics 8.4.129; PyTorch 2.13.0+cu130; Python 3.13.14 |
| Training time | 0.308 hours (18.5 minutes) |

## Best-checkpoint validation results

Validation set: 444 images, 681 instances.

| Class | Precision | Recall | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|
| all | 0.573 | 0.626 | 0.620 | 0.310 |
| minor_damage | 0.534 | 0.441 | 0.459 | 0.284 |
| compressed | 0.519 | 0.541 | 0.520 | 0.189 |
| hole | 0.617 | 0.748 | 0.757 | 0.377 |
| wet | 0.620 | 0.774 | 0.743 | 0.390 |

Validation speed per image: 0.1 ms preprocess, 1.3 ms inference, and 0.9 ms postprocess.

## Held-out test results

Test set: 419 images, 665 instances.

| Class | Precision | Recall | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|
| all | 0.563 | 0.584 | 0.571 | 0.284 |
| minor_damage | 0.573 | 0.485 | 0.474 | 0.258 |
| compressed | 0.435 | 0.325 | 0.304 | 0.102 |
| hole | 0.637 | 0.761 | 0.765 | 0.364 |
| wet | 0.607 | 0.765 | 0.742 | 0.412 |

Test speed per image: 0.7 ms preprocess, 2.1 ms inference, and 0.8 ms postprocess.

## Interpretation

- The baseline detects `hole` and `wet` most reliably.
- `compressed` is the primary weakness: test recall is 0.325 and mAP50–95 is 0.102.
- `minor_damage` is the second weakest class, consistent with its broad ontology and low representation.
- The validation-to-test decrease is moderate and establishes a useful reference point for Phase 5 comparisons.

## Preserved artifacts

- Training run and curves: `runs/EXP-001_yolo11n_baseline/`
- Best weights: `runs/EXP-001_yolo11n_baseline/weights/best.pt`
- Last weights: `runs/EXP-001_yolo11n_baseline/weights/last.pt`
- Validation confusion matrices and prediction samples: training run directory
- Test confusion matrices, curves, samples, and COCO JSON predictions: `runs/EXP-001_yolo11n_baseline_test/`
- Reproducible training entry point: `scripts/train_exp001.py`
- Reproducible test entry point: `scripts/evaluate_exp001.py`
