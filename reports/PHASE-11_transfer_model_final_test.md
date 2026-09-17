# Phase 11 — Final Test Evaluation of `transfer-mobile-v1`

## Final outcome

The exact Phase 10D system was evaluated once on the 592-image `parcel_binary_v2` test split. Every pre-inference integrity check passed. No weight, threshold, architecture, preprocessing rule, image, or label was changed.

**NO — final damaged recall did not achieve the predefined ≥90% target.** Test recall is **89.29%** (350/392; 95% Wilson CI 85.83–91.98%). The result is preserved as obtained. The threshold remains `0.434584259987`.

## Frozen system evaluated

| Component | Value |
|---|---|
| Model version | `transfer-mobile-v1` |
| Architecture | MobileNetV3-Large |
| Pretrained source | ImageNet V2 |
| Training | Strictly frozen backbone + trained final linear head |
| Parameters | 4,203,313 |
| Input | RGB; resize shorter side 256; center crop 224×224 |
| Normalization | ImageNet mean/std |
| Frozen threshold | `0.434584259987` |
| Checkpoint SHA-256 | `f255600ad26289e7a3cc3b277926ffd91c8384a013a4203808b12d4ea77629d1` |

Inference followed the frozen sequence:

```text
Image [3,224,224]
  → batch [B,3,224,224]
  → MobileNet output logits [B]
  → sigmoid probabilities [B]
  → probability >= 0.434584259987
  → damaged / intact
```

`model.eval()` disabled Dropout and used fixed BatchNorm statistics. `torch.no_grad()` disabled gradient tracking because test data must never update model weights.

## Phase 11A integrity gate

All checks ran before any test image was decoded:

| Check | Result |
|---|---|
| Checkpoint hash matches Phase 10D | PASS |
| Dataset version is `parcel_binary_v2` | PASS |
| Dataset manifest hash matches frozen record | PASS |
| Test split contains exactly 592 images | PASS |
| On-disk image inventory exactly matches manifest | PASS |
| All 3,949 image hashes match manifest | PASS |
| Train/validation/test group overlap | 0 |
| Exact-hash cross-split leakage | 0 |
| Phase 8C dataset audit remains frozen | PASS |
| Threshold matches Phase 10D | PASS |
| Model status was frozen before test | PASS |

Added, missing, or modified dataset images: 0 / 0 / 0.

## Final test metrics

| Metric | Final test result |
|---|---:|
| Test images | 592 |
| Accuracy | 85.64% |
| Damaged precision | 89.06% |
| Damaged recall | **89.29%** |
| Recall 95% CI | 85.83–91.98% |
| Damaged F1 | 89.17% |
| Specificity | **78.50%** |
| Specificity 95% CI | 72.30–83.63% |
| Balanced accuracy | 83.89% |
| ROC-AUC | 93.35% |
| PR-AUC | 97.15% |
| True positives | 350 |
| True negatives | 157 |
| False positives | **43** |
| False negatives | **42** |

Confusion matrix, rows=true and columns=predicted, intact then damaged:

```text
[[157, 43],
 [ 42,350]]
```

Operationally, 42/392 damaged parcels (10.71%) were missed, while 43/200 intact parcels (21.50%) were unnecessarily sent to inspection.

## Frozen subgroup evaluation

These subgroup definitions existed before test predictions and did not affect threshold selection.

| Subgroup | Support | Detected | False negatives | Recall | 95% CI |
|---|---:|---:|---:|---:|---:|
| Ordinary damaged | 343 | 336 | 7 | **97.96%** | 95.85–99.01% |
| Open box | 49 | 14 | 35 | **28.57%** | 17.85–42.41% |

Open boxes represent only 12.5% of damaged test samples but contribute 83.3% of all false negatives. The overall target miss is therefore driven primarily by the operational open-box label, not ordinary visible damage.

## Validation versus final test

| Metric | Validation | Test | Difference |
|---|---:|---:|---:|
| Recall | 90.31% | 89.29% | −1.02 pp |
| Specificity | 86.00% | 78.50% | −7.50 pp |
| Precision | 92.67% | 89.06% | −3.61 pp |
| F1 | 91.47% | 89.17% | −2.30 pp |
| Accuracy | 88.85% | 85.64% | −3.21 pp |
| ROC-AUC | 94.89% | 93.35% | −1.54 pp |
| PR-AUC | 97.60% | 97.15% | −0.45 pp |

Test performance is directionally consistent with validation: ranking remains strong and the operating metrics are moderately lower. Recall missed the target by 0.71 percentage points. This difference is not a reason to alter the threshold; doing so would violate the pre-test freeze.

The largest subgroup shift is open-box recall, from 51.02% validation to 28.57% test. Ordinary-damaged recall increased from 95.92% to 97.96%.

## Final classification comparison

Evaluation scopes are kept explicit. The scratch CNN has not received a final test evaluation, so its validation metrics must not be presented as test performance.

| Model | Evaluation scope | Recall | Specificity | Precision | F1 | ROC-AUC | PR-AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| HOG + LinearSVC | Final test | **91.07%** | 30.50% | 71.98% | 80.41% | 73.96% | Not recorded |
| Scratch CNN Phase 9A | Validation only | 90.05% | 53.00% | 78.97% | 84.15% | 81.60% | 88.83% |
| MobileNetV3-Large | **Final test** | 89.29% | **78.50%** | **89.06%** | **89.17%** | **93.35%** | **97.15%** |

HOG achieves 1.79 points more final-test recall, but MobileNet improves specificity by 48 points, precision by 17.08 points, F1 by 8.76 points, and ROC-AUC by 19.39 points. MobileNet provides a much more usable operational balance, although it narrowly misses the recall target.

No valid final-test comparison with the scratch CNN is possible yet.

## Deployment measurements

| Measurement | Result |
|---|---:|
| GPU | NVIDIA GeForce RTX 5060 |
| Average GPU forward latency | 0.899 ms/image |
| Forward throughput | 1,112 images/second |
| Checkpoint size | 16.23 MiB |
| Parameters | 4,203,313 |

Latency measures model forward execution after the tensor is on the GPU. It excludes disk decoding and host-to-device transfer and includes the first batch of this final run, so it is not identical to the earlier warmed validation estimate.

## Descriptive error analysis

Primary metrics and predictions were permanently saved before images were inspected.

### False negatives

Most false negatives are clean-looking open cartons or packing scenes: narrow lid gaps, upright cartons with subtle open flaps, boxes actively being taped, stacks containing open cartons, and scenes where the open state is contextual rather than visible structural damage. Several appear almost identical to intact rectangular boxes.

The seven ordinary-damaged misses include localized or subtle deformation, cluttered stacks, branded parcels, partial views, and damage occupying a small portion of the image. Strong object identity and branding can dominate the global ImageNet representation while local damage remains weak.

### False positives

False positives commonly show intact boxes with tape, labels, printed branding, seams, stacked geometry, strong perspective edges, shadows, patterned backgrounds, or corner shapes that resemble deformation. Several studio images include black borders or unusual viewpoints, while warehouse scenes contain clutter and overlapping parcels.

This analysis explains limitations only. It did not trigger relabeling, removal, threshold adjustment, retraining, or another MobileNet attempt against the same test set.

## Scientific closure

- Recall target achieved: **NO**.
- Threshold changed after test: **No**.
- Model retrained or fine-tuned after test: **No**.
- Preprocessing, labels, or dataset changed: **No**.
- Test-driven modification performed: **No**.

`transfer-mobile-v1` is closed. Any later improvement must be a new experiment with a new evaluation protocol.

## Artifacts

- Evaluation script: `scripts/evaluate_transfer_model_final.py`
- Metrics: `models/selected_transfer_model/final_test_metrics.json`
- Predictions: `models/selected_transfer_model/final_test_predictions.csv`
- Integrity: `models/selected_transfer_model/final_integrity_check.json`
- Subgroups: `models/selected_transfer_model/final_subgroup_results.json`
- Confusion matrix: `models/selected_transfer_model/final_confusion_matrix.png`
- ROC curve: `models/selected_transfer_model/final_roc_curve.png`
- PR curve: `models/selected_transfer_model/final_pr_curve.png`
- FN/FP contact sheets: `models/selected_transfer_model/final_false_negative_contact_sheet.jpg`, `final_false_positive_contact_sheet.jpg`
- Reproducibility: `models/selected_transfer_model/final_reproducibility_metadata.json`

Phase 11 stops here. Severity estimation was not started.
