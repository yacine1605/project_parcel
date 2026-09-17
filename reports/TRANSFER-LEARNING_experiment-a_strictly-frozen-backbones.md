# Transfer Learning Experiment A — Strictly Frozen ImageNet Backbones

## Outcome

ResNet50, EfficientNet-B0, and MobileNetV3-Large were trained as frozen ImageNet feature extractors on `parcel_binary_v2`. Every backbone stayed in evaluation mode during head training, so parameters, BatchNorm statistics, and stochastic backbone behavior remained fixed. Only a new one-logit linear head learned.

The test split was not accessed. No threshold calibration or backbone fine-tuning was performed.

**MobileNetV3-Large currently has the strongest frozen representation by ranking:** ROC-AUC 94.89% and PR-AUC 97.60%. EfficientNet-B0 is the only candidate reaching ≥90% recall at threshold 0.5 and has the best open-box recall. MobileNet is fastest; EfficientNet is smallest.

## How frozen transfer learning works

ImageNet pretraining supplies broadly useful visual filters. The frozen **backbone** converts an image into fixed features; the new **classification head** maps them to one damage logit.

```text
RGB image → frozen ImageNet backbone → fixed features
          → trainable linear head → logit → sigmoid → probability
          → threshold 0.5 → damaged / intact
```

`requires_grad=False` blocks backbone parameter gradients. Keeping the backbone in `eval()` additionally freezes BatchNorm buffers and stochastic behavior. `BCEWithLogitsLoss` safely combines sigmoid and binary cross-entropy during training; sigmoid is applied separately during validation.

## Shared protocol

| Setting | Value |
|---|---|
| Dataset | `parcel_binary_v2`, unchanged train/validation groups |
| Input | RGB; resize shorter side 256; center crop 224×224 |
| Normalization | ImageNet mean/std |
| Random augmentation | None |
| Epochs / batch | 10 / 32 |
| Optimizer / LR | Adam / 0.001 |
| Loss | `BCEWithLogitsLoss` |
| Checkpoint | Minimum validation loss |
| Threshold | 0.5, uncalibrated |
| Seed / GPU | 42 / RTX 5060 |

## Comparison with frozen scratch CNN

The scratch row uses its calibrated Phase 9C threshold; pretrained rows use threshold 0.5. ROC-AUC and PR-AUC are therefore the fairest representation comparison.

| Model | Recall | Specificity | Precision | F1 | ROC-AUC | PR-AUC | Trainable params | Time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Scratch CNN reference | 90.05% | 53.00% | 78.97% | 84.15% | 81.60% | 88.83% | 93,825 | 267.9 s |
| ResNet50 frozen | 86.73% | 88.50% | 93.66% | 90.07% | 93.53% | 96.97% | 2,049 | 184.8 s |
| EfficientNet-B0 frozen | **90.05%** | 83.50% | 91.45% | **90.75%** | 94.63% | 97.58% | 1,281 | 158.1 s |
| MobileNetV3-Large frozen | 86.99% | **90.50%** | **94.72%** | 90.69% | **94.89%** | **97.60%** | 1,281 | **132.0 s** |

All pretrained models substantially exceed the scratch CNN in ROC-AUC and PR-AUC, supporting the hypothesis that pretrained features generalize better on validation.

## Complete validation metrics at threshold 0.5

| Metric | ResNet50 | EfficientNet-B0 | MobileNetV3-Large |
|---|---:|---:|---:|
| Accuracy | 87.33% | 87.84% | **88.18%** |
| Precision | 93.66% | 91.45% | **94.72%** |
| Recall | 86.73% | **90.05%** | 86.99% |
| F1 | 90.07% | **90.75%** | 90.69% |
| Specificity | 88.50% | 83.50% | **90.50%** |
| Balanced accuracy | 87.62% | 86.78% | **88.74%** |
| ROC-AUC | 93.53% | 94.63% | **94.89%** |
| PR-AUC | 96.97% | 97.58% | **97.60%** |
| TP / TN | 340 / 177 | 353 / 167 | 341 / 181 |
| FP / FN | 23 / 52 | 33 / **39** | **19** / 51 |

Confusion matrices: ResNet `[[177,23],[52,340]]`; EfficientNet `[[167,33],[39,353]]`; MobileNet `[[181,19],[51,341]]`.

## Frozen subgroups

| Model | Ordinary damaged | Open box |
|---|---:|---:|
| ResNet50 | 93.88% (322/343) | 36.73% (18/49) |
| EfficientNet-B0 | **95.92%** (329/343) | **48.98%** (24/49) |
| MobileNetV3-Large | 94.46% (324/343) | 34.69% (17/49) |

Open boxes remain much harder. Subgroups did not influence checkpoint selection.

## Parameters, size, and latency

| Model | Total / frozen / trainable params | Checkpoint | Best epoch/loss | GPU latency |
|---|---:|---:|---:|---:|
| ResNet50 | 23,510,081 / 23,508,032 / 2,049 | 89.99 MiB | 10 / 0.3049 | 2.210 ms/image |
| EfficientNet-B0 | 4,008,829 / 4,007,548 / 1,281 | **15.58 MiB** | 10 / 0.2768 | 0.920 ms/image |
| MobileNetV3-Large | 4,203,313 / 4,202,032 / 1,281 | 16.23 MiB | 10 / **0.2751** | **0.511 ms/image** |

Latency is GPU forward time after tensor transfer, excluding disk decoding and host-to-device transfer.

## Interpretation

- **Strongest representation:** MobileNetV3-Large, narrowly, based on best ROC-AUC/PR-AUC plus accuracy, specificity, precision, balanced accuracy, and latency.
- **Best recall/open-box candidate:** EfficientNet-B0; nearly tied AUC and smallest checkpoint.
- **ResNet50:** strong but dominated here by smaller candidates in ranking, speed, and size.

No final winner is frozen because candidates have not yet been compared at a calibrated common recall point, and fine-tuning belongs to a later experiment.

## Integrity note

Preliminary runs that froze parameters but permitted BatchNorm/stochastic training behavior were rejected before final reporting and archived under `models/transfer_learning/invalid_bn_adaptive_initial_runs/`. Results here are corrected strict-freeze reruns.

## Artifacts

- `scripts/train_transfer_frozen_backbone.py`
- `models/transfer_learning/resnet50_frozen/`
- `models/transfer_learning/efficientnet_b0_frozen/`
- `models/transfer_learning/mobilenet_v3_large_frozen/`
- `models/transfer_learning/experiment_a_comparison.csv`
- `models/transfer_learning/reproducibility_metadata.json`

Experiment A stops here. No calibration, fine-tuning, test evaluation, or severity work was started.
