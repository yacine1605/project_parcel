# Phase 8F — Final Frozen-Test Evaluation

## Final outcome

The exact Phase 8E system was evaluated once on the previously unopened `parcel_binary_v2` test split. All pre-evaluation integrity checks passed. The frozen test set contains 592 images: 392 damaged and 200 intact.

**YES — final damaged recall reached the predefined ≥90% operational target.** Test recall is **91.07%** (357/392; 95% Wilson CI 87.84–93.51%). No model, scaler, HOG setting, threshold, dataset, or label was modified after observing the result.

## Frozen system

- Dataset: `parcel_binary_v2`
- HOG: 8,100 dimensions; 128×128 grayscale; 9 orientations; 8×8 pixels/cell; 2×2 cells/block; L2-Hys
- Scaling: train-fitted `StandardScaler(with_mean=False)`
- Classifier: `LinearSVC(C=0.0001, class_weight=None)`
- Frozen threshold: **`-0.462288626498`**
- Prediction rule: damaged when `decision_score >= threshold`
- Threshold source: Phase 8E validation only; target ≥90% damaged recall

The classifier was not retrained, the scaler was not refitted, and the threshold was not recalibrated.

## Pre-evaluation integrity gate

Every check completed before test image decoding and HOG extraction.

| Check | Result |
|---|---|
| Phase 8C audit marked frozen; 3,949 rows expected | PASS |
| Manifest has 3,949 rows; test has exactly 592 | PASS |
| On-disk image inventory exactly matches manifest | PASS |
| All 3,949 current file hashes match manifest hashes | PASS |
| Added / missing / modified images | 0 / 0 / 0 |
| Group IDs crossing splits | 0 |
| Exact hashes crossing splits | 0 |
| Phase 8D model hash matches Phase 8E record | PASS |
| Threshold and pre-test freeze flag match | PASS |

Recorded SHA-256 values:

- Dataset manifest: `79f4307694e5a9415523e72dfe15af7720af566f32af907db767a89a8ea87284`
- Phase 8C audit: `66ac82edfdfba984ded56eb12d67d02748bcef03103cf2be214b97bf9d82f19b`
- Phase 8D model: `253036c590c70395140aff3fa99fd8ca07290f998d04dd14ba5fb6f9e4fd0d8a`
- Phase 8E threshold artifact at evaluation: `88cf1f7fedd1c0d6501fc198060b8f7c493f6eac98c71fb9ad5f7d9e1766ec79`

## Final test metrics

| Metric | Final test result |
|---|---:|
| Accuracy | 70.61% |
| Damaged precision | 71.98% |
| Damaged recall / sensitivity | **91.07%** |
| Damaged recall 95% CI | 87.84–93.51% |
| Damaged F1 | 80.41% |
| Specificity | **30.50%** |
| Specificity 95% CI | 24.54–37.20% |
| False-positive rate | 69.50% |
| False-negative rate | 8.93% |
| Balanced accuracy | 60.79% |
| ROC-AUC | 73.96% |
| PR-AUC / average precision | 82.10% |
| True positives | 357 |
| True negatives | 61 |
| False positives | **139** |
| False negatives | **35** |

Confusion matrix (rows=true, columns=predicted; intact then damaged):

`[[61, 139], [35, 357]]`

## Validation versus final test

| Metric | Frozen calibrated validation | Final test | Change |
|---|---:|---:|---:|
| Accuracy | 71.11% | 70.61% | −0.50 pp |
| Damaged precision | 72.78% | 71.98% | −0.80 pp |
| Damaged recall | 90.05% | 91.07% | +1.02 pp |
| Damaged F1 | 80.50% | 80.41% | −0.10 pp |
| Specificity | 34.00% | 30.50% | −3.50 pp |
| False negatives | 39 / 392 | 35 / 392 | −4 |
| False positives | 132 / 200 | 139 / 200 | +7 |

Test performance is broadly consistent with validation at the frozen operating point. Recall generalized slightly above target, while specificity was modestly worse. This ordinary variation is not grounds for changing the threshold.

## Mandatory frozen subgroups

| Damaged subgroup | Validation recall | Test support | Test TP / FN | Test recall | 95% CI |
|---|---:|---:|---:|---:|---:|
| Ordinary damaged | 93.00% | 343 | 319 / 24 | **93.00%** | 89.80–95.25% |
| Open box mapped to damaged | 69.39% | 49 | 38 / 11 | **77.55%** | 64.12–86.98% |

These definitions were frozen before test evaluation. Open boxes remain harder than ordinary visible damage, although their test recall is 8.16 percentage points above validation.

## Operational interpretation

- Damaged parcels missed: **35/392 (8.93%)**.
- Intact parcels falsely routed for inspection: **139/200 (69.50%)**.
- Operational recall objective: **achieved**.

The calibrated classifier is defensible as a high-recall screening baseline, but its false-alarm burden is severe: more than two-thirds of truly intact test parcels would be sent for review. It is not suitable for autonomous acceptance/rejection and would require substantial manual-review capacity.

## Descriptive error analysis

Error sheets were generated only after primary metrics and predictions were permanently recorded.

False negatives include clean-looking open cartons, high-key studio renders, partial or cluttered parcel views, and damaged boxes whose strongest cues are localized holes, corner deformation, or subtle compression. Several ordinary-damaged misses have obvious local defects to a person, illustrating how global grayscale HOG can underweight small regions. Open-box misses commonly resemble intact rectangular boxes except for a narrow lid/flap gap or packing context.

False positives are dominated by intact boxes with strong edges and texture: tape, labels, printed branding, seams, handles, stacked parcels, perspective distortion, patterned backgrounds, and rendered/studio lighting. These cues resemble the gradients found in damaged training images. The low calibrated threshold deliberately amplifies this confusion to meet recall.

This analysis is descriptive only. It did not trigger threshold changes, sample removal, relabeling, retraining, or any other test-driven modification.

## Permanent audit artifacts

- Metrics: `models/phase8f_artifacts/final_test_metrics.json`
- Predictions: `models/phase8f_artifacts/final_test_predictions.csv`
- Integrity record: `models/phase8f_artifacts/integrity_check.json`
- Subgroups: `models/phase8f_artifacts/subgroup_results.json`
- Confusion matrix: `models/phase8f_artifacts/final_confusion_matrix.png`
- ROC curve: `models/phase8f_artifacts/final_roc_curve.png`
- PR curve: `models/phase8f_artifacts/final_pr_curve.png`
- FN/FP contact sheets: `models/phase8f_artifacts/false_negative_contact_sheet.jpg`, `false_positive_contact_sheet.jpg`
- Reproducibility: `models/phase8f_artifacts/reproducibility_metadata.json`
- Evaluation script: `scripts/evaluate_hog_svm_v2_final.py`

Phase 8F is closed. Any future architecture is a separate experiment and must not use this test result iteratively for development.
