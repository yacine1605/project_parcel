# EXP-003 — YOLO11n Compressed 3× Oversampling

Status: **COMPLETE — REJECTED AFTER HELD-OUT EVALUATION**

## Controlled change

EXP-003 uses frozen `parcel_damage_v3` and the EXP-002 settings. The only
training change is sampling each of the 147 training images containing
`compressed` three times per epoch. The experiment-only manifest contains 2,375
entries (2,081 unique images plus 294 repeat entries). No dataset file changed.

| Setting | Value |
|---|---|
| Model | YOLO11n pretrained |
| Image size | 640 |
| Epochs | 75 |
| Batch | 16 |

| Seed | 42, deterministic mode |
| Training time | 0.504 hours (30.7 minutes) |
| GPU | NVIDIA GeForce RTX 5060, 8151 MiB |

## Validation results

| Class | Precision | Recall | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|
| all | 0.638 | 0.568 | 0.583 | 0.289 |
| minor_damage | 0.667 | 0.386 | 0.446 | 0.235 |
| compressed | 0.555 | 0.469 | 0.412 | 0.178 |
| hole | 0.654 | 0.725 | 0.759 | 0.380 |
| wet | 0.677 | 0.690 | 0.715 | 0.362 |

Compared with EXP-002 on the identical V3 validation set, compressed recall
improved from 0.344 to 0.469 (+0.125), overall recall improved by 0.007, and
precision improved by 0.054. Overall mAP50 decreased by 0.005 and mAP50–95 by
0.007. This validation result justified one held-out evaluation.

## Held-out test results

| Class | Precision | Recall | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|
| all | 0.600 | 0.468 | 0.525 | 0.261 |
| minor_damage | 0.620 | 0.362 | 0.457 | 0.257 |
| compressed | 0.313 | 0.175 | 0.168 | 0.068 |
| hole | 0.750 | 0.683 | 0.762 | 0.361 |
| wet | 0.718 | 0.650 | 0.713 | 0.356 |

## Decision

The targeted validation improvement did not generalize. Compressed test recall
fell to 0.175 and overall test recall to 0.468. EXP-003 is rejected for model
selection and 3× repeated-image oversampling should not be pursued further.

The test set has now been inspected for this experiment and must not be used to
choose the next training change. Subsequent experiments must be selected on
validation only. A new untouched external or real-world evaluation set should
be reserved for the final selected system.

The evidence points away from more aggressive repetition of the same 147
compressed images. The next roadmap phase should compare YOLO11s against
YOLO11n on V3 using ordinary sampling, with selection performed strictly on
validation.

## Artifacts

- Training run: `runs/EXP-003_yolo11n_compressed_x3/`
- Best weights: `runs/EXP-003_yolo11n_compressed_x3/weights/best.pt`
- Sampling configuration: `experiments/EXP-003/`
- Preparation script: `scripts/prepare_exp003.py`
- Training script: `scripts/train_exp003.py`
