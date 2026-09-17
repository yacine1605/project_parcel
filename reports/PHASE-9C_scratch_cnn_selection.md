# Phase 9C — Validation Threshold Calibration and Scratch-CNN Selection

## Outcome

Both frozen scratch CNN checkpoints were calibrated independently using their saved validation probabilities. No model was trained or loaded, no image was loaded, and the test split was not accessed.

The predefined requirement was damaged recall ≥90%. Among eligible thresholds, each candidate maximized specificity, then precision, then F1, then proximity to 0.5. Both calibrated candidates reached exactly **90.05% recall**. Phase 9A won because its calibrated specificity was **53.00%**, compared with **44.50%** for Phase 9B.

The selected scratch system is therefore:

- Model: **Phase 9A custom CNN without augmentation**
- Checkpoint: `models/phase9a_custom_cnn/best_model.pt`
- Checkpoint SHA-256: `2a1aacc9650f076a90d94a564557d6300571b91d545637df11ae84293bf8f4c7`
- Frozen probability threshold: **`0.400157928467`**
- Dataset: `parcel_binary_v2`
- Architecture: `ParcelDamageCNN`, 93,825 parameters
- Preprocessing: resize 224×224 RGB, tensor conversion, ImageNet normalization
- Training augmentation: none

## Threshold calibration results

| Candidate | Default threshold | Calibrated threshold | Recall target achieved | Calibrated specificity |
|---|---:|---:|---|---:|
| Phase 9A, no augmentation | 0.5 | **0.400157928467** | YES, 90.05% | **53.00%** |
| Phase 9B, augmentation | 0.5 | **0.529901325703** | YES, 90.05% | **44.50%** |

Phase 9A needed a lower threshold to detect more damaged parcels. Phase 9B's default threshold already exceeded the recall target, so calibration raised its threshold to recover specificity until the next higher score transition would violate the target.

## Required four-way comparison

| Metric | 9A default | 9A calibrated | 9B default | 9B calibrated |
|---|---:|---:|---:|---:|
| Threshold | 0.500000 | **0.400158** | 0.500000 | **0.529901** |
| Damaged recall | 81.38% | **90.05%** | 91.58% | **90.05%** |
| Specificity | 69.00% | **53.00%** | 39.50% | **44.50%** |
| Damaged precision | 83.73% | **78.97%** | 74.79% | **76.08%** |
| Damaged F1 | 82.54% | **84.15%** | 82.34% | **82.48%** |
| Accuracy | 77.20% | **77.53%** | 73.99% | **74.66%** |
| Balanced accuracy | 75.19% | **71.53%** | 65.54% | **67.28%** |
| True positives | 319 | 353 | 359 | 353 |
| True negatives | 138 | 106 | 79 | 89 |
| False negatives | 73 | **39** | 33 | **39** |
| False positives | 62 | **94** | 121 | **111** |
| Ordinary-damaged recall | 86.30% | **94.17%** | 94.75% | **94.17%** |
| Open-box recall | 46.94% | **61.22%** | 69.39% | **61.22%** |

Ranking metrics are threshold-independent:

| Candidate | ROC-AUC | PR-AUC |
|---|---:|---:|
| Phase 9A | **81.60%** | **88.83%** |
| Phase 9B | 79.25% | 86.99% |

## Calibrated confusion matrices

Rows are true labels and columns are predicted labels, ordered intact then damaged.

- Phase 9A: `[[106, 94], [39, 353]]`
- Phase 9B: `[[89, 111], [39, 353]]`

At the common recall operating point, both candidates miss the same 39 damaged validation samples. Phase 9A correctly leaves 17 more intact parcels as intact, producing 94 false positives rather than 111. Its precision, F1, accuracy, balanced accuracy, ROC-AUC, and PR-AUC are also higher.

## Subgroup analysis

Subgroup performance did not participate in threshold selection. At their independently calibrated thresholds, both candidates happen to produce the same subgroup results:

| Subgroup | Support | Detected | False negatives | Recall |
|---|---:|---:|---:|---:|
| Ordinary damaged | 343 | 323 | 20 | 94.17% |
| Open box mapped to damaged | 49 | 30 | 19 | 61.22% |

Calibration improves Phase 9A's ordinary-damaged recall by 7.87 points and open-box recall by 14.29 points relative to its default threshold. For Phase 9B, raising the threshold reduces ordinary recall by 0.58 points and open-box recall by 8.16 points while recovering five points of specificity.

## Selection decision

**Winner: Phase 9A custom CNN without augmentation.**

Both candidates satisfy the operational recall constraint, so specificity is the first model-selection discriminator. Phase 9A wins by 8.50 specificity points and 17 fewer false positives at identical recall and false-negative count. No later tie-break is needed, although Phase 9A also has higher precision, F1, ROC-AUC, and PR-AUC and uses the simpler training pipeline.

The validation trade-off for the selected model is:

- 353/392 damaged parcels detected;
- 39/392 damaged parcels missed;
- 106/200 intact parcels correctly left intact;
- 94/200 intact parcels sent to review unnecessarily.

This threshold is now frozen before any scratch-CNN test evaluation. It must not be changed after future test results are seen.

## Reproducibility and leakage control

- Inputs were saved validation prediction CSVs only.
- Candidate inventories matched exactly: 592 unique validation samples each.
- Labels matched between candidates.
- Frozen Phase 9B subgroup metadata was reused for both candidates.
- Checkpoints were hashed but not loaded or modified.
- No image files or dataset manifest were opened.
- No training, refitting, or threshold tuning on test occurred.
- Test accessed: **No**.

## Artifacts

- Calibration/selection script: `scripts/calibrate_and_select_custom_cnn.py`
- Phase 9A sweep: `models/phase9c_artifacts/phase9a_threshold_calibration.csv`
- Phase 9B sweep: `models/phase9c_artifacts/phase9b_threshold_calibration.csv`
- Four-point comparison: `models/phase9c_artifacts/scratch_cnn_comparison.csv`
- Frozen selection: `models/phase9c_artifacts/selected_scratch_cnn.json`
- Reproducibility metadata: `models/phase9c_artifacts/reproducibility_metadata.json`
- Threshold and recall/specificity plots: `models/phase9c_artifacts/*.png`

Phase 9C stops here. No scratch-CNN test evaluation or transfer-learning experiment was started.
