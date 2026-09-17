# Phase 10B/10C — Frozen Transfer Candidate Comparison and Calibration

## Purpose

Phase 10A showed that partial fine-tuning degraded validation performance and rapidly overfit. Following the roadmap decision, Phase 10B/10C therefore compared the three stronger strict-freeze Experiment A candidates:

- ResNet50 frozen;
- EfficientNet-B0 frozen;
- MobileNetV3-Large frozen.

No model was trained, fine-tuned, or loaded. Calibration used saved validation probabilities only. The test split was not accessed.

The operational requirement was **damaged recall ≥90%**. For each candidate independently, the selected threshold maximized specificity among eligible thresholds, then precision, then F1, then proximity to 0.5.

## Why threshold calibration matters

The neural network produces a damaged probability. A threshold converts it into an operational decision.

```text
Probability = 0.72

Threshold 0.50:
0.72 >= 0.50 → damaged

Threshold 0.80:
0.72 < 0.80 → intact
```

Lowering the threshold usually detects more damaged parcels, increasing recall but reducing specificity. Raising it usually sends fewer parcels to review, increasing specificity but risking more damaged misses. Threshold 0.5 is conventional, not automatically optimal.

ROC-AUC and PR-AUC do not change during threshold calibration because the ordering of model probabilities is unchanged. ROC-AUC describes damaged-versus-intact ranking across thresholds; PR-AUC focuses on ranking quality for the damaged class.

## Calibrated thresholds

| Candidate | Default threshold | Calibrated threshold | Recall requirement achieved |
|---|---:|---:|---|
| ResNet50 frozen | 0.500000 | **0.386661082506** | YES — 90.05% |
| EfficientNet-B0 frozen | 0.500000 | **0.534759759903** | YES — 90.05% |
| MobileNetV3-Large frozen | 0.500000 | **0.434584259987** | YES — 90.31% |

ResNet and MobileNet needed lower thresholds to increase sensitivity. EfficientNet already exceeded the exact required number of detections at 0.5; its threshold could rise slightly, recovering two intact parcels without losing a damaged detection.

## Default versus calibrated results

| Model | Operating point | Recall | Specificity | Precision | F1 | Accuracy | Balanced accuracy | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ResNet50 | Default 0.5 | 86.73% | 88.50% | 93.66% | 90.07% | 87.33% | 87.62% | 23 | 52 |
| ResNet50 | **Calibrated** | **90.05%** | 84.00% | 91.69% | 90.86% | 88.01% | 87.03% | 32 | 39 |
| EfficientNet-B0 | Default 0.5 | 90.05% | 83.50% | 91.45% | 90.75% | 87.84% | 86.78% | 33 | 39 |
| EfficientNet-B0 | **Calibrated** | **90.05%** | 84.50% | 91.93% | 90.98% | 88.18% | 87.28% | 31 | 39 |
| MobileNetV3 | Default 0.5 | 86.99% | 90.50% | 94.72% | 90.69% | 88.18% | 88.74% | 19 | 51 |
| MobileNetV3 | **Calibrated** | **90.31%** | **86.00%** | **92.67%** | **91.47%** | **88.85%** | **88.15%** | **28** | **38** |

Calibrated confusion matrices, ordered intact then damaged:

- ResNet50: `[[168, 32], [39, 353]]`
- EfficientNet-B0: `[[169, 31], [39, 353]]`
- MobileNetV3-Large: `[[172, 28], [38, 354]]`

At the common operational requirement, MobileNet correctly recognizes the most intact parcels and detects one more damaged parcel than the other candidates. It produces four fewer false positives than ResNet, three fewer than EfficientNet, and one fewer false negative.

## Ranking, size, and latency

| Model | ROC-AUC | PR-AUC | Parameters | Checkpoint | RTX 5060 latency |
|---|---:|---:|---:|---:|---:|
| ResNet50 | 93.53% | 96.97% | 23,510,081 | 89.99 MiB | 2.210 ms/image |
| EfficientNet-B0 | 94.63% | 97.58% | **4,008,829** | **15.58 MiB** | 0.920 ms/image |
| MobileNetV3-Large | **94.89%** | **97.60%** | 4,203,313 | 16.23 MiB | **0.511 ms/image** |

MobileNet has the strongest ranking metrics and is fastest. EfficientNet is slightly smaller and nearly tied in AUC, making it a strong alternative. ResNet is much larger and slower while trailing both efficient architectures in ranking quality.

## Mandatory subgroup comparison

Subgroup performance was measured after global threshold selection and did not tune thresholds.

| Model | Operating point | Ordinary damaged recall | Open-box recall |
|---|---|---:|---:|
| ResNet50 | Default | 93.88% (322/343) | 36.73% (18/49) |
| ResNet50 | Calibrated | 95.92% (329/343) | 48.98% (24/49) |
| EfficientNet-B0 | Default | 95.92% (329/343) | 48.98% (24/49) |
| EfficientNet-B0 | Calibrated | 95.92% (329/343) | 48.98% (24/49) |
| MobileNetV3 | Default | 94.46% (324/343) | 34.69% (17/49) |
| MobileNetV3 | Calibrated | **95.92%** (329/343) | **51.02%** (25/49) |

All calibrated models detect ordinary visible damage well. Open boxes remain difficult because they can look structurally clean. MobileNet's calibrated threshold provides the best open-box result, but still misses 24/49 open-box validation images.

## Validation leader

**Current Phase 10B/10C validation leader: MobileNetV3-Large frozen at threshold `0.434584259987`.**

Selection evidence:

1. all candidates meet ≥90% recall;
2. MobileNet has the highest specificity: 86.00%;
3. it also has the highest precision and F1;
4. it has the highest ROC-AUC and PR-AUC;
5. it is the fastest candidate;
6. it has the best calibrated open-box recall.

No tie-break was required. EfficientNet remains close and is 0.65 MiB smaller, but its specificity is 1.50 points lower and it creates three more false positives at essentially the same recall.

Following the roadmap, this phase records the validation leader but does not perform the formal Phase 10D model freeze or authorize test evaluation. That remains the next separate phase.

## Scientific controls

- Validation inventories matched: 592 unique samples per candidate.
- Labels and frozen subgroup definitions matched exactly.
- Inputs were saved Experiment A validation prediction CSVs.
- No image, model checkpoint, dataset manifest, or test artifact was loaded.
- No training or fine-tuning occurred.
- Threshold candidates were every unique observed validation probability plus 0.5.
- Test accessed: **No**.

## Artifacts

- Script: `scripts/calibrate_transfer_frozen_candidates.py`
- Complete result: `models/transfer_learning_calibration/validation_calibration_results.json`
- Six-point comparison: `models/transfer_learning_calibration/calibrated_candidate_comparison.csv`
- Per-model threshold sweeps: `models/transfer_learning_calibration/*_threshold_calibration.csv`
- Threshold and recall/specificity plots: `models/transfer_learning_calibration/*.png`
- Reproducibility metadata: `models/transfer_learning_calibration/reproducibility_metadata.json`

Phase 10B/10C stops here. No Phase 10D freeze, final test evaluation, transfer fine-tuning, or new experiment was started.
