# Phase 7 — Classical Damaged-vs-Intact Baseline Summary

## Outcome

Phase 7 produced a reproducible HOG + linear-SVM image classifier and a new leakage-controlled binary split. Final test damaged recall is **71.84%** with **98 false negatives out of 348 damaged images**. This is a useful classical reference point, but it is not operationally adequate for a recall-first damage-screening gate.

## Dataset preparation and leakage control

- Source: 4,148 images (2,478 damaged / 1,670 nominal intact).
- Retained: 3,959 images after 175 redundant exact-copy exclusions and 14 mixed-label perceptual-group exclusions.
- New seed-42 group split: train 2,771; validation 594; test 594.
- Split class counts: train 1,623/1,148, validation 348/246, test 348/246 (damaged/intact).
- Roboflow rotation siblings, exact hashes, and conservative pHash matches remain within one split.
- Automated checks found zero group leakage and zero retained mixed-class groups.
- Raw data, `parcel_damage_v3`, existing YOLO experiments, and frozen detector datasets were not modified.

The main unresolved limitation is label quality: all 1,640 retained intact images require manual review, and seven visually confirmed open-box examples are flagged. Undocumented multiple views of the same physical parcel cannot be ruled out. Results therefore describe source-label prediction, not fully adjudicated parcel condition.

## HOG and selected SVM

- 128×128 grayscale, 9 orientations, 8×8 pixels/cell, 2×2 cells/block, L2-Hys.
- 8,100 features per image.
- Selected on validation only: `LinearSVC(C=0.001, class_weight=None)` with train-only sparse scaling.

| Metric | Validation | Final test |
|---|---:|---:|
| Accuracy | 74.41% | 71.04% |
| Damaged precision | 79.17% | 77.16% |
| Damaged recall | 76.44% | 71.84% |
| Damaged F1 | 77.78% | 74.40% |
| ROC-AUC | 79.73% | 77.24% |

## Strengths and weaknesses versus YOLO

HOG+SVM is fast to train, inexpensive, deterministic, simple to deploy on CPU, and useful as a sanity-check baseline. It requires only image-level labels. Its weaknesses are sensitivity to background/pose, weak representation of subtle localized damage, no localization, and a high false-negative rate.

YOLO can localize damage and potentially learn richer appearance cues, but requires trustworthy bounding boxes and a fair image-level decision rule for binary comparison. YOLO mAP and HOG+SVM classification accuracy measure different tasks and must not be compared directly.

## Operational assessment and next experiment

The current baseline is useful for certification benchmarking and pipeline validation, not for autonomous parcel rejection or acceptance. It could serve as a low-cost secondary score only with human review and a validation-calibrated recall threshold.

Recommended next experiment: complete blinded manual adjudication of the intact manifest (including an explicit policy for open boxes), rebuild `parcel_binary_v2` with physical-parcel/session group IDs where possible, then calibrate the frozen model family’s decision threshold on validation for a target damaged recall (for example ≥90%) and assess whether specificity remains usable on a newly frozen test split. A fair image-level YOLO comparison may follow on that same adjudicated split without retraining or selecting on its test set.
