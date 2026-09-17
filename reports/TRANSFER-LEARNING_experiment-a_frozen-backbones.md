# SUPERSEDED — Preliminary BatchNorm-Adaptive Runs

> **Do not use this report for conclusions.** Parameter gradients were frozen, but the initial implementation left backbone BatchNorm/stochastic behavior in training mode. The runs were rejected and archived. The authoritative corrected report is `TRANSFER-LEARNING_experiment-a_strictly-frozen-backbones.md`.

# Transfer Learning Experiment A — Frozen ImageNet Backbones

## Outcome

Three standard torchvision architectures were evaluated as frozen-backbone feature extractors on `parcel_binary_v2`. Each model used official ImageNet-pretrained weights. All convolutional/backbone parameters had `requires_grad=False`; only a newly initialized one-logit classification head was trained.

Training and validation splits were unchanged. The test split was not accessed. No backbone was fine-tuned and no threshold was calibrated.

At threshold 0.5, **ResNet50 currently provides the strongest pretrained representation overall**: it has the highest accuracy, F1, specificity, balanced accuracy, ROC-AUC, and PR-AUC. EfficientNet-B0 has the highest damaged recall and open-box recall among the three. MobileNetV3-Large is the fastest and smallest-latency option.

## What transfer learning means

ImageNet pretraining teaches a CNN broadly useful visual filters from a large image dataset. The convolutional **backbone** turns an RGB image into learned features such as edges, textures, shapes, and object parts. The **classification head** maps those features to this project's output: one parcel-damage logit.

```text
224×224 RGB parcel image
        ↓
Frozen ImageNet backbone
(general visual features; no weight updates)
        ↓
New trainable linear head
        ↓
One raw logit
        ↓
Sigmoid during validation
        ↓
Probability threshold 0.5
        ↓
Damaged / intact
```

`requires_grad=False` prevents PyTorch from calculating gradients for backbone parameters. `loss.backward()` therefore calculates useful gradients only for the new head, and Adam updates only that head. `BCEWithLogitsLoss` combines sigmoid and binary cross-entropy safely; sigmoid is applied separately during validation to obtain probabilities.

## Shared protocol

| Setting | Value |
|---|---|
| Dataset | `parcel_binary_v2`, existing train/validation groups |
| Input | RGB; resize shorter side to 256; center crop 224×224 |
| Normalization | ImageNet channel mean and standard deviation |
| Training augmentation | None |
| Epochs | 10 |
| Batch size | 32 |
| Optimizer | Adam |
| Head learning rate | 0.001 |
| Loss | `BCEWithLogitsLoss` |
| Checkpoint rule | Minimum validation loss |
| Metrics threshold | 0.5, not calibrated |
| Seed | 42 |
| GPU | NVIDIA GeForce RTX 5060 |

The deterministic preprocessing is intentionally common across all candidates. This creates a clean pretrained-feature baseline before later fine-tuning or augmentation experiments.

## Main comparison

The scratch-CNN row uses its frozen Phase 9C validation operating point at threshold `0.400157928467`. Pretrained rows use the requested uncalibrated threshold 0.5, so their recall values should not be treated as final operating-point comparisons. ROC-AUC and PR-AUC are particularly informative here.

| Model | Recall | Specificity | Precision | F1 | ROC-AUC | PR-AUC | Trainable params | Training time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Scratch CNN, frozen reference | **90.05%** | 53.00% | 78.97% | 84.15% | 81.60% | 88.83% | 93,825 | 267.9 s |
| ResNet50 frozen | 85.20% | **90.50%** | **94.62%** | **89.66%** | **94.08%** | **97.22%** | 2,049 | 182.2 s |
| EfficientNet-B0 frozen | **85.97%** | 84.50% | 91.58% | 88.68% | 93.88% | 97.10% | 1,281 | 150.8 s |
| MobileNetV3-Large frozen | 82.40% | 86.50% | 92.29% | 87.06% | 93.27% | 96.89% | 1,281 | **129.1 s** |

All three pretrained models have dramatically higher ROC-AUC and PR-AUC than the scratch CNN. This is strong validation evidence that pretrained ImageNet representations rank damaged versus intact parcels more effectively, even before backbone fine-tuning.

## Complete validation metrics at threshold 0.5

| Metric | ResNet50 | EfficientNet-B0 | MobileNetV3-Large |
|---|---:|---:|---:|
| Accuracy | **86.99%** | 85.47% | 83.78% |
| Damaged precision | **94.62%** | 91.58% | 92.29% |
| Damaged recall | 85.20% | **85.97%** | 82.40% |
| Damaged F1 | **89.66%** | 88.68% | 87.06% |
| Specificity | **90.50%** | 84.50% | 86.50% |
| Balanced accuracy | **87.85%** | 85.23% | 84.45% |
| ROC-AUC | **94.08%** | 93.88% | 93.27% |
| PR-AUC | **97.22%** | 97.10% | 96.89% |
| TP / TN | 334 / 181 | 337 / 169 | 323 / 173 |
| FP / FN | **19 / 58** | 31 / **55** | 27 / 69 |

Confusion matrices, ordered intact then damaged:

- ResNet50: `[[181, 19], [58, 334]]`
- EfficientNet-B0: `[[169, 31], [55, 337]]`
- MobileNetV3-Large: `[[173, 27], [69, 323]]`

None reaches the 90% global recall target at the arbitrary 0.5 threshold. That is not a failure of the representations; threshold calibration is deliberately deferred.

## Frozen subgroup results

| Model | Ordinary damaged | Open box | Open-box support |
|---|---:|---:|---:|
| ResNet50 | **92.13%** (316/343) | 36.73% (18/49) | 49 |
| EfficientNet-B0 | 90.96% (312/343) | **51.02%** (25/49) | 49 |
| MobileNetV3-Large | 90.96% (312/343) | 22.45% (11/49) | 49 |

All pretrained representations distinguish ordinary visible damage much better than clean-looking operational open-box cases. EfficientNet-B0 is the strongest frozen representation for open boxes at threshold 0.5. Subgroups did not influence checkpoint selection.

## Parameters, size, and deployment measurements

| Model | Total params | Frozen params | Trainable params | Checkpoint | Best epoch | GPU latency |
|---|---:|---:|---:|---:|---:|---:|
| ResNet50 | 23,510,081 | 23,508,032 | 2,049 | 89.99 MiB | 9 | 2.245 ms/image |
| EfficientNet-B0 | 4,008,829 | 4,007,548 | 1,281 | **15.58 MiB** | 10 | 0.794 ms/image |
| MobileNetV3-Large | 4,203,313 | 4,202,032 | 1,281 | 16.23 MiB | 7 | **0.441 ms/image** |

Latency is GPU forward-pass time on the RTX 5060 after tensors were moved to the GPU. It excludes disk decoding and host-to-device transfer. MobileNetV3-Large offers the best raw inference speed; EfficientNet-B0 is slightly smaller on disk; ResNet50 trades much greater size and latency for the strongest validation discrimination.

## Architecture-specific results

### ResNet50 frozen

- Weights: `IMAGENET1K_V2`
- Trainable tensors: `fc.weight`, `fc.bias`
- Best validation loss: 0.2988 at epoch 9
- Main strength: best ranking metrics, F1, specificity, and lowest false positives
- Limitation: largest model and poor open-box recall

### EfficientNet-B0 frozen

- Weights: `IMAGENET1K_V1`
- Trainable tensors: `classifier.1.weight`, `classifier.1.bias`
- Best validation loss: 0.3008 at epoch 10
- Main strength: highest default recall and open-box recall, near-ResNet AUC at one-sixth the size
- Limitation: lower specificity than ResNet50

### MobileNetV3-Large frozen

- Weights: `IMAGENET1K_V2`
- Trainable tensors: `classifier.3.weight`, `classifier.3.bias`
- Best validation loss: 0.3254 at epoch 7
- Main strength: 0.441 ms/image latency with strong ranking metrics
- Limitation: lowest overall and open-box recall

## Current interpretation

ResNet50 currently looks strongest when predictive quality is primary. EfficientNet-B0 has the most attractive accuracy/size compromise and deserves close attention during later calibrated comparison. MobileNetV3-Large is the deployment-speed candidate.

No final winner is frozen in Experiment A because the pretrained models have not yet been compared at the shared ≥90% recall operating point, and fine-tuning is a later controlled experiment. The present conclusion is limited to validation and frozen ImageNet features.

## Artifacts

Each model directory contains `best_model.pt`, `experiment_metadata.json`, `training_history.csv`, `validation_predictions.csv`, `confusion_matrix.png`, and `training_curves.png`:

- `models/transfer_learning/resnet50_frozen/`
- `models/transfer_learning/efficientnet_b0_frozen/`
- `models/transfer_learning/mobilenet_v3_large_frozen/`

Shared artifacts:

- Training script: `scripts/train_transfer_frozen_backbone.py`
- Comparison CSV: `models/transfer_learning/experiment_a_comparison.csv`
- Reproducibility metadata: `models/transfer_learning/reproducibility_metadata.json`

Experiment A stops here. No threshold calibration, backbone fine-tuning, test evaluation, severity estimation, or additional model training was started.
