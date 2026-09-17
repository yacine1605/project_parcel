# EXP-004 — YOLO11s Baseline on V3

Status: **COMPLETE — NOT SELECTED**

## Configuration

EXP-004 changes only model size relative to EXP-002. Both use frozen
`parcel_damage_v3`, ordinary sampling, pretrained weights, 640 px images, 75
epochs, batch 16, seed 42, deterministic mode, and default Ultralytics
augmentation.

| Setting | Value |
|---|---|
| Model | YOLO11s pretrained |
| Parameters | 9,414,348 fused |
| Compute | 21.4 GFLOPs |
| Training time | 0.656 hours (39.4 minutes) |
| Best checkpoint size | 19.18 MB |
| GPU | NVIDIA GeForce RTX 5060, 8151 MiB |

## Validation results

| Class | Precision | Recall | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|
| all | 0.516 | 0.570 | 0.560 | 0.283 |
| minor_damage | 0.480 | 0.364 | 0.405 | 0.228 |
| compressed | 0.406 | 0.438 | 0.358 | 0.150 |
| hole | 0.575 | 0.760 | 0.739 | 0.362 |
| wet | 0.605 | 0.720 | 0.740 | 0.392 |

Measured validation speed per image: 1.0 ms preprocess, 15.4 ms inference, and
2.0 ms postprocess.

## Phase 5 comparison

| Model | Precision | Recall | mAP50 | mAP50–95 | Inference | Checkpoint |
|---|---:|---:|---:|---:|---:|---:|
| YOLO11n (EXP-002) | 0.584 | 0.561 | 0.588 | 0.296 | 1.3 ms | 5.48 MB |
| YOLO11s (EXP-004) | 0.516 | 0.570 | 0.560 | 0.283 | 15.4 ms | 19.18 MB |

YOLO11s improves overall recall by only 0.009 and compressed recall by 0.094.
It loses 0.068 precision, 0.028 mAP50, and 0.013 mAP50–95 while its checkpoint
is 3.5 times larger and measured inference is much slower on the same GPU.

## Decision

**YOLO11n is selected as the Phase 5 accuracy-versus-latency winner.** YOLO11s
does not provide enough recall improvement to justify its lower overall accuracy
and substantially greater runtime and storage cost. EXP-004 is not evaluated on
the held-out test set; selection was made strictly on validation.

The next phase should test realistic augmentation with YOLO11n on frozen V3,
using EXP-002 as the ordinary-sampling reference. The already-inspected test set
must remain closed during augmentation selection; final robustness should use a
new external or real-world evaluation set.

## Artifacts

- Training run: `runs/EXP-004_yolo11s_baseline_v3/`
- Best weights: `runs/EXP-004_yolo11s_baseline_v3/weights/best.pt`
- Training script: `scripts/train_exp004.py`
- Pretrained weights: `yolo11s.pt`
