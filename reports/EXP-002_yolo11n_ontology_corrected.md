# EXP-002 — YOLO11n Ontology-Corrected Baseline

Status: **COMPLETE — NOT SELECTED**

## Configuration

| Setting | Value |
|---|---|
| Model | YOLO11n pretrained |
| Dataset | `parcel_damage_v3` (frozen) |
| Image size | 640 |
| Epochs | 75 |
| Batch | 16 |
| Augmentation | Ultralytics default baseline |
| Optimizer | AdamW, selected by Ultralytics auto optimizer |
| Seed | 42, deterministic mode |
| GPU | NVIDIA GeForce RTX 5060, 8151 MiB |
| Software | Ultralytics 8.4.129; PyTorch 2.13.0+cu130; Python 3.13.14 |
| Training time | 0.312 hours (18.7 minutes) |

The only intended experimental change from EXP-001 was the reviewed annotation
policy in V3. Model and training settings were held constant.

## Best-checkpoint validation results

Validation set: 444 images, 681 instances.

| Class | Precision | Recall | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|
| all | 0.584 | 0.561 | 0.588 | 0.296 |
| minor_damage | 0.448 | 0.409 | 0.414 | 0.218 |
| compressed | 0.571 | 0.344 | 0.412 | 0.191 |
| hole | 0.618 | 0.746 | 0.763 | 0.378 |
| wet | 0.699 | 0.744 | 0.763 | 0.399 |

Validation speed per image: 0.1 ms preprocess, 1.3 ms inference, and 0.9 ms
postprocess.

## Comparison with EXP-001

| Metric | EXP-001 / V2 | EXP-002 / V3 | Change |
|---|---:|---:|---:|
| Precision | 0.573 | 0.584 | +0.011 |
| Recall | 0.626 | 0.561 | -0.065 |
| mAP50 | 0.620 | 0.588 | -0.032 |
| mAP50–95 | 0.310 | 0.296 | -0.014 |

This is not a perfectly label-invariant comparison because V3 deliberately
changes the validation ground truth. It measures the complete revised task, not
just a model response against identical labels. The revised `compressed` class
has 32 validation instances instead of 42, while `minor_damage` has 44 instead
of 34.

## Decision

EXP-002 is **not selected** because recall—the operational priority—decreased by
0.065 and neither mAP metric improved. The held-out test split was not evaluated,
because the experiment was rejected using validation as planned.

V3 remains the scientifically preferable dataset because its ontology was
visually reviewed. The next controlled experiment should address the now-smaller
`compressed` training class through targeted balancing while retaining V3 and
all other EXP-002 settings. This should be named EXP-003, not treated as a retry
of EXP-002.

## Preserved artifacts

- Run: `runs/EXP-002_yolo11n_ontology_corrected/`
- Best weights: `runs/EXP-002_yolo11n_ontology_corrected/weights/best.pt`
- Last weights: `runs/EXP-002_yolo11n_ontology_corrected/weights/last.pt`
- Reproducible entry point: `scripts/train_exp002.py`
- V3 freeze record: `reports_v3/FREEZE.md`
- V3 change log: `reports/dataset_v3_changes.json`
