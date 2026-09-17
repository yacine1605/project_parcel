# Phase 10D — Best Transfer-Learning Model Freeze

## Purpose

Phase 10B/10C compared all strict-freeze transfer candidates at the same validation recall requirement. MobileNetV3-Large led specificity, precision, F1, ROC-AUC, PR-AUC, false-positive count, false-negative count, open-box recall, and latency.

Phase 10D turns that validation decision into a fixed inference contract before test evaluation. No model was trained or loaded, no probability was recomputed, no threshold was recalibrated, and the test split was not accessed.

## Frozen transfer system

| Component | Frozen value |
|---|---|
| Dataset | `parcel_binary_v2` |
| Architecture | MobileNetV3-Large |
| Torchvision constructor | `torchvision.models.mobilenet_v3_large` |
| Pretrained weights | `MobileNet_V3_Large_Weights.IMAGENET1K_V2` |
| Training strategy | Strictly frozen ImageNet backbone; only final one-logit head trained |
| Checkpoint | `models/transfer_learning/mobilenet_v3_large_frozen/best_model.pt` |
| Checkpoint SHA-256 | `f255600ad26289e7a3cc3b277926ffd91c8384a013a4203808b12d4ea77629d1` |
| Checkpoint size | 16.23 MiB |
| Total parameters | 4,203,313 |
| Trainable during Experiment A | 1,281 |
| Frozen threshold | **`0.434584259987`** |
| Freeze version | `transfer-mobile-v1` |

The checkpoint is referenced by path and cryptographic hash instead of copied. This avoids two model files silently diverging while still allowing later integrity verification.

## Frozen preprocessing

```text
Input image
  → convert to RGB
  → resize shorter side to 256, preserving aspect ratio
  → center crop 224×224
  → convert to tensor [3, 224, 224]
  → normalize with ImageNet mean/std
  → MobileNetV3-Large
  → one raw logit
  → sigmoid probability
  → threshold 0.434584259987
  → damaged / intact
```

ImageNet normalization:

- mean: `[0.485, 0.456, 0.406]`
- standard deviation: `[0.229, 0.224, 0.225]`

For a batch of 32 images, the input tensor shape is `[32, 3, 224, 224]`: 32 images, three RGB channels, and 224×224 spatial dimensions. The model produces logits with shape `[32]`, one raw score per image. Sigmoid converts logits into damaged probabilities.

The operational rule is:

```text
probability >= 0.434584259987 → damaged
probability <  0.434584259987 → intact
```

This threshold must not change after future test results are observed.

## Frozen validation evidence

| Metric | Result |
|---|---:|
| Accuracy | 88.85% |
| Damaged recall | **90.31%** |
| Specificity | **86.00%** |
| Damaged precision | **92.67%** |
| Damaged F1 | **91.47%** |
| Balanced accuracy | 88.15% |
| ROC-AUC | 94.89% |
| PR-AUC | 97.60% |
| TP / TN | 354 / 172 |
| FP / FN | 28 / 38 |

Confusion matrix: `[[172, 28], [38, 354]]`.

Frozen subgroup recall:

- ordinary damaged: 95.92% (329/343);
- open box: 51.02% (25/49).

Measured RTX 5060 GPU forward latency was 0.511 ms/image after tensor transfer.

## Integrity checks

The freeze script verified:

- architecture and dataset version match the selected experiment;
- training used train/validation only;
- test remained closed during training and calibration;
- the backbone passed the strict state-level freeze audit;
- calibration used validation only;
- the recall target remained the predefined 90%;
- MobileNetV3-Large is the recorded validation leader;
- calibrated recall satisfies the target;
- all source artifacts exist and are hashed.

Every check passed. The machine-readable manifest additionally records hashes for the checkpoint, experiment metadata, validation predictions, calibration result, and strict-freeze audit.

## What freezing means

The following choices are now closed for this transfer-learning model family:

- architecture;
- pretrained weights;
- backbone-freeze strategy;
- checkpoint;
- input resolution and preprocessing;
- normalization;
- probability conversion;
- operating threshold;
- dataset version;
- validation metrics and selection rationale.

Future test performance may be reported and analyzed, but it must not cause any of these values to change. A changed system would be a new experiment requiring a new evaluation protocol.

## Artifacts

- Freeze script: `scripts/freeze_best_transfer_model.py`
- Frozen manifest: `models/selected_transfer_model/selected_transfer_model.json`
- Freeze reproducibility: `models/selected_transfer_model/reproducibility_metadata.json`
- Referenced checkpoint: `models/transfer_learning/mobilenet_v3_large_frozen/best_model.pt`

Phase 10D is complete. No Phase 11 test evaluation was started.
