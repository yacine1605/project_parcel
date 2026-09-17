# EXP-006 — YOLO11n Mild Photometric Augmentation

Status: **COMPLETE — NOT SELECTED**

## Controlled change

EXP-006 uses frozen `parcel_damage_v3` and the EXP-002 YOLO11n settings. It
changes only the default HSV augmentation strength:

| Parameter | EXP-002 default | EXP-006 mild |
|---|---:|---:|
| `hsv_h` | 0.015 | 0.010 |
| `hsv_s` | 0.700 | 0.350 |
| `hsv_v` | 0.400 | 0.250 |

No geometric augmentation was added. Training used 75 epochs, batch 16, 640 px,
seed 42, deterministic mode, and ordinary sampling. Training time was 0.477
hours (28.6 minutes).

## Validation results

| Class | Precision | Recall | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|
| all | 0.515 | 0.562 | 0.568 | 0.285 |
| minor_damage | 0.444 | 0.364 | 0.397 | 0.232 |
| compressed | 0.422 | 0.344 | 0.339 | 0.143 |
| hole | 0.599 | 0.769 | 0.766 | 0.377 |
| wet | 0.596 | 0.774 | 0.770 | 0.388 |

Validation speed per image: 0.9 ms preprocess, 3.0 ms inference, and 1.3 ms
postprocess.

## Comparison with EXP-002

| Metric | EXP-002 default | EXP-006 mild HSV | Change |
|---|---:|---:|---:|
| Precision | 0.584 | 0.515 | -0.069 |
| Recall | 0.561 | 0.562 | +0.001 |
| mAP50 | 0.588 | 0.568 | -0.020 |
| mAP50–95 | 0.296 | 0.285 | -0.011 |
| Compressed recall | 0.344 | 0.344 | 0.000 |
| Compressed mAP50 | 0.412 | 0.339 | -0.073 |

## Phase 6 decision

Milder photometric jitter provides no meaningful recall benefit and reduces
precision and AP. EXP-006 is rejected. EXP-005 improved AP but reduced recall,
so it is retained only as a precision-oriented candidate.

**EXP-002 remains the selected recall-first YOLO11n detector on V3.** The
Ultralytics default augmentation policy remains the training default. No Phase 6
candidate was evaluated on the held-out test set.

The next roadmap phase is the classical image-level machine-learning baseline:
damaged versus intact using HOG features and an SVM. This requires confirming
how intact/negative images are represented, because the current detection
dataset contains no empty-label images.

## Artifacts

- Training run: `runs/EXP-006_yolo11n_mild_photometric_aug/`
- Best weights: `runs/EXP-006_yolo11n_mild_photometric_aug/weights/best.pt`
- Training script: `scripts/train_exp006.py`
