# EXP-005 — YOLO11n Mild Geometric Augmentation

Status: **COMPLETE — NOT SELECTED AS DEFAULT**

## Controlled change

EXP-005 uses frozen `parcel_damage_v3` and the EXP-002 YOLO11n settings. The
only policy change is realistic geometric augmentation:

- rotation: ±5 degrees (`degrees=5.0`)
- perspective: `0.0005`

All other Ultralytics defaults, ordinary sampling, resolution, epochs, batch,
seed, and deterministic mode remain unchanged.

| Setting | Value |
|---|---|
| Model | YOLO11n pretrained |
| Epochs | 75 |
| Training time | 0.441 hours (26.5 minutes) |
| Dataset | Frozen V3 |

## Validation results

| Class | Precision | Recall | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|
| all | 0.692 | 0.547 | 0.612 | 0.298 |
| minor_damage | 0.611 | 0.357 | 0.456 | 0.247 |
| compressed | 0.711 | 0.375 | 0.463 | 0.219 |
| hole | 0.699 | 0.712 | 0.765 | 0.364 |
| wet | 0.747 | 0.744 | 0.763 | 0.363 |

Validation speed per image: 0.9 ms preprocess, 3.2 ms inference, and 1.3 ms
postprocess.

## Comparison with ordinary augmentation

| Metric | EXP-002 baseline | EXP-005 geometry | Change |
|---|---:|---:|---:|
| Precision | 0.584 | 0.692 | +0.108 |
| Recall | 0.561 | 0.547 | -0.014 |
| mAP50 | 0.588 | 0.612 | +0.024 |
| mAP50–95 | 0.296 | 0.298 | +0.002 |
| Compressed recall | 0.344 | 0.375 | +0.031 |
| Compressed mAP50 | 0.412 | 0.463 | +0.051 |

The augmentation improves precision, AP, and compressed-class behavior, but
reduces overall recall because minor-damage and hole recall decline. This is a
useful precision-oriented checkpoint, but it does not satisfy the project's
recall-first default selection rule.

## Decision

EXP-005 is not selected as the default model. EXP-002 remains the recall-first
YOLO11n reference on V3. EXP-005 is retained as a candidate for later confidence
threshold analysis and real-world rotated-view testing.

The held-out test set was not evaluated. The next augmentation experiment should
test a mild photometric policy (reduced brightness/contrast variation) without
geometric changes, selected only on validation. Blur and noise require a custom
augmentation pipeline and should be tested separately rather than bundled into
the same experiment.

## Artifacts

- Training run: `runs/EXP-005_yolo11n_mild_geometry_aug/`
- Best weights: `runs/EXP-005_yolo11n_mild_geometry_aug/weights/best.pt`
- Training script: `scripts/train_exp005.py`
