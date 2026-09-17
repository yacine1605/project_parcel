# Phase 7D — Final HOG + SVM Evaluation

The frozen validation-selected model was evaluated **once** on the untouched 594-image Phase 7 binary test split.

## Final metrics

| Metric | Result |
|---|---:|
| Accuracy | 71.04% |
| Damaged precision | 77.16% |
| Damaged recall / sensitivity | 71.84% |
| Damaged F1 | 74.40% |
| Specificity | 69.92% |
| ROC-AUC | 77.24% |
| False negatives | **98** |
| False positives | **74** |

Confusion matrix, rows=true and columns=predicted, ordered `[intact, damaged]`:

| | Predicted intact | Predicted damaged |
|---|---:|---:|
| True intact | 172 | 74 |
| True damaged | 98 | 250 |

## Per-class metrics

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Intact | 63.70% | 69.92% | 66.67% | 246 |
| Damaged | 77.16% | 71.84% | 74.40% | 348 |
| Macro average | 70.43% | 70.88% | 70.54% | 594 |
| Weighted average | 71.59% | 71.04% | 71.20% | 594 |

## False-negative risk

The 98 missed damaged images span 54 leakage-control groups, so the problem is not merely repeated rotations of one parcel. The most confident misses include source names `leaking67`, `train_image_7289`, `train_image_7278`, `crushed18`, and `leaking97`. This suggests HOG shape/edge statistics can fail when damage texture is subtle, localized, low contrast, or dominated by parcel/background appearance. Exact image IDs, decision scores, and paths are preserved in `models/phase7_artifacts/phase7_test_predictions.csv`.

A 28.16% damaged miss rate is too high for a recall-first operational gate at the default SVM threshold. Threshold calibration on validation—not test—could trade specificity for recall in a future experiment, but was not retroactively performed after viewing test results.

HOG+SVM is image-level classification only. It provides no damage localization. Its accuracy must not be compared directly with YOLO mAP. A future fair image-level YOLO comparison must define `damaged` as at least one detection over a validation-selected confidence threshold and `intact` otherwise, then evaluate on the same Phase 7 groups after confirming no training leakage.
