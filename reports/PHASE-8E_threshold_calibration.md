# Phase 8E — Validation-Only Operating Threshold Calibration

## Outcome

The frozen Phase 8D model was calibrated on validation scores only. It was not retrained. The frozen test split was not loaded, featurized, scored, or evaluated.

The predefined damaged-recall target was **≥90%**. No more specific criterion was encoded, so the fallback selected, among feasible thresholds, maximum specificity, then damaged precision, then proximity to zero. The frozen threshold is **`-0.462288626498`** and achieved the requirement with **90.05% damaged recall**.

## Threshold-0 reproduction gate

| Metric | Threshold 0 |
|---|---:|
| Accuracy | 72.64% |
| Damaged precision | 80.26% |
| Damaged recall | 77.81% |
| Damaged F1 | 79.02% |
| Specificity | 62.50% |
| ROC-AUC | 77.83% |
| FP / FN | 75 / 87 |
| Confusion matrix | `[[125, 75], [87, 305]]` |

This exactly reproduces Phase 8D, so calibration proceeded. ROC-AUC remains 77.83% because thresholding does not change score ranking.

## Selected operating point

| Metric | Threshold 0 | Calibrated `-0.462289` | Change |
|---|---:|---:|---:|
| Accuracy | 72.64% | 71.11% | −1.52 pp |
| Damaged precision | 80.26% | 72.78% | −7.48 pp |
| Damaged recall | 77.81% | **90.05%** | +12.24 pp |
| Damaged F1 | 79.02% | 80.50% | +1.49 pp |
| Specificity | 62.50% | **34.00%** | −28.50 pp |
| False positives | 75 | **132** | +57 |
| False negatives | 87 | **39** | −48 |
| Confusion matrix | `[[125, 75], [87, 305]]` | `[[68, 132], [39, 353]]` | — |

Meeting the recall requirement prevents 48 damaged misses while producing 57 additional false alarms on validation. The model reaches the recall target, but its 34% specificity makes it unsuitable as a standalone acceptance gate without substantial manual-review capacity.

## Damaged-subgroup analysis

| Subgroup | Support | Recall at 0 | Recall calibrated | Change | FN at 0 | FN calibrated |
|---|---:|---:|---:|---:|---:|---:|
| Ordinary damaged | 343 | 81.92% | 93.00% | +11.08 pp | 62 | 24 |
| Open box mapped to damaged | 49 | 48.98% | 69.39% | +20.41 pp | 25 | 15 |

Subgroups were analyzed only after global threshold selection and did not tune the threshold. Calibration improves open-box recall substantially, but 30.61% of open-box validation samples remain missed.

## Frozen artifacts

- Full sweep: `models/phase8e_artifacts/threshold_calibration.csv`
- Frozen threshold: `models/phase8e_artifacts/selected_threshold.json`
- Enriched validation predictions: `models/phase8e_artifacts/validation_predictions.csv`
- Selected-point neighborhood: `models/phase8e_artifacts/selected_threshold_neighborhood.csv`
- Curves: recall, specificity, precision, and FP/FN under `models/phase8e_artifacts/`
- Reproducibility metadata: `models/phase8e_artifacts/reproducibility_metadata.json`
- Reproduction script: `scripts/calibrate_hog_svm_threshold_v2.py`

The threshold is frozen before test evaluation and must not be changed after test results are seen. Phase 8E stops here.
