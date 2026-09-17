# Phase 8D — HOG + Linear SVM Validation on Frozen V2

## Outcome

The Phase 7 classical pipeline was reproduced on frozen `parcel_binary_v2` using train and validation only. The test directory, test features, test predictions, and test metrics were not opened or produced.

The recall-first selection policy chose **`LinearSVC(C=0.0001, class_weight=None)`**. This configuration is frozen for Phase 8E threshold calibration; its default decision threshold has not been calibrated.

## Feature and leakage verification

- HOG: 128×128 grayscale; 9 orientations; 8×8 pixels/cell; 2×2 cells/block; L2-Hys.
- Resize behavior: Pillow `ImageOps.fit`, Lanczos, matching Phase 7.
- Train features: 2,765 × 8,100.
- Validation features: 592 × 8,100.
- Actual feature dimensionality: **8,100**.
- Unreadable images: 0; NaN/Inf values: 0.
- Train/validation leakage-control groups spanning splits: 0.
- Scaling: `StandardScaler(with_mean=False)`, fitted on train only and applied to validation.

## Controlled model selection

Selection maximized damaged recall, then damaged F1, precision, ROC-AUC, and finally preferred lower C. No subgroup metric participated in selection.

| C | Class weight | Accuracy | Damaged precision | Damaged recall | Damaged F1 | Specificity | ROC-AUC |
|---:|:---|---:|---:|---:|---:|---:|---:|
| **0.0001** | **None** | **72.64%** | **80.26%** | **77.81%** | **79.02%** | **62.50%** | **77.83%** |
| 0.001 | None | 70.95% | 80.73% | 73.72% | 77.07% | 65.50% | 76.08% |
| 0.01 | None | 70.44% | 80.06% | 73.72% | 76.76% | 64.00% | 75.60% |
| 0.1 | None | 70.61% | 80.11% | 73.98% | 76.92% | 64.00% | 75.52% |
| 0.0001 | balanced | 72.47% | 82.44% | 74.23% | 78.12% | 69.00% | 77.71% |
| 0.001 | balanced | 70.61% | 80.79% | 72.96% | 76.68% | 66.00% | 76.05% |
| 0.01 | balanced | 70.61% | 80.45% | 73.47% | 76.80% | 65.00% | 75.60% |
| 0.1 | balanced | 70.61% | 80.11% | 73.98% | 76.92% | 64.00% | 75.52% |

## Selected validation result

| Metric | Result |
|---|---:|
| Accuracy | 72.64% |
| Damaged precision | 80.26% |
| Damaged recall | **77.81%** |
| Damaged F1 | 79.02% |
| Intact recall / specificity | **62.50%** |
| ROC-AUC | **77.83%** |
| False negatives | **87** |
| False positives | **75** |

Confusion matrix (rows=true, columns=predicted; order intact, damaged): `[[125, 75], [87, 305]]`.

## Damaged-subgroup diagnosis

| True damaged subgroup | Support | Recall | False negatives |
|---|---:|---:|---:|
| Ordinary adjudicated damaged | 343 | 81.92% | 62 |
| Open box mapped to damaged | 49 | **48.98%** | 25 |

Open-box images are 12.5% of validation damaged samples but contribute 28.7% of damaged false negatives. This is consistent with HOG learning visible structural appearance more readily than the operational policy that an otherwise sound open carton is unacceptable. This diagnostic did not alter model selection.

## Phase 7 comparison

These are different validation sets, so changes describe the combined effect of adjudication and the new leakage-controlled split—not a paired-image improvement estimate.

| Metric | Phase 7 validation | Phase 8D validation | Change |
|---|---:|---:|---:|
| Accuracy | 74.41% | 72.64% | −1.78 pp |
| Damaged precision | 79.17% | 80.26% | +1.10 pp |
| Damaged recall | 76.44% | 77.81% | +1.37 pp |
| Damaged F1 | 77.78% | 79.02% | +1.24 pp |
| ROC-AUC | 79.73% | 77.83% | −1.90 pp |

Recall and F1 improved modestly, while accuracy and ranking performance declined. The principal adverse change is intact false positives: Phase 7 specificity was 71.54% (`176/246`), versus 62.50% (`125/200`) here. Damaged miss rate improved from 23.56% to 22.19%, but open-box policy cases remain a distinct weakness. Improved label quality did not yield a general performance increase for HOG; it exposed a harder, operationally defined boundary that grayscale shape/texture features represent poorly.

## Frozen artifacts and next step

- Model and train-fitted scaler: `models/hog_svm_phase8d_v2.joblib`
- Validation results and candidates: `models/phase8d_artifacts/validation_results.json`
- Model-selection CSV: `models/phase8d_artifacts/model_selection.csv`
- Validation predictions: `models/phase8d_artifacts/validation_predictions.csv`
- Confusion matrix: `models/phase8d_artifacts/validation_confusion_matrix.png`
- HOG metadata and train/validation features: `models/phase8d_artifacts/features/`
- Reproducibility metadata: `models/phase8d_artifacts/reproducibility_metadata.json`
- Scripts: `scripts/extract_hog_features_v2.py`, `scripts/train_hog_svm_v2.py`

Phase 8E may calibrate a decision threshold using these validation scores for a predefined damaged-recall target. It must freeze that threshold before opening the test split.
