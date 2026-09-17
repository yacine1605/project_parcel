# Phase 10A — Transfer Learning Experiment B: Partial Fine-Tuning

## What this phase tested

Transfer learning reuses visual knowledge learned from ImageNet. Experiment A kept that knowledge fixed and trained only a classification head. Phase 10A allowed selected late layers to adapt to parcel-specific features such as crushed geometry, holes, tears, open flaps, seams, tape, labels, and cardboard texture.

Each Phase 10A model started from its valid Experiment A checkpoint. No model was trained from scratch. The test split was not accessed, and threshold 0.5 was not calibrated.

**Result:** partial fine-tuning did not improve validation performance under this controlled protocol. All three models rapidly overfit; minimum validation loss selected epoch 1 or 2. The unsuccessful experiments are preserved rather than hidden.

## Educational architecture overview

### ResNet50

```text
Input → early convolution → layer1 → layer2 → layer3
      → layer4 → classification head
```

Residual connections help information and gradients flow through a deep network. Early convolution through layer3 stayed frozen. `layer4 + fc` were trainable because layer4 contains the highest-level ImageNet features most likely to benefit from parcel adaptation.

### EfficientNet-B0

EfficientNet balances depth, width, and image resolution rather than scaling only one dimension. Early feature blocks stayed frozen; `features.7`, `features.8`, and the classifier were trainable.

### MobileNetV3-Large

MobileNet is designed for efficient inference using lightweight operations, conceptually including depthwise-style spatial filtering and inexpensive channel mixing. Early features stayed frozen; `features.15`, `features.16`, and the classifier were trainable.

## Fine-tuning and learning rates

`requires_grad=False` means PyTorch does not calculate parameter updates for a weight. `requires_grad=True` permits learning during backpropagation.

Late-backbone parameters used learning rate `1e-4`; heads used `5e-4`. The pretrained network already contains useful knowledge, so the smaller backbone rate attempts to avoid destructive large updates. The head can move faster because it directly models the parcel task.

BatchNorm contains trainable affine scale/shift parameters plus running mean/variance buffers. Frozen blocks remained in evaluation mode, keeping all of these fixed. BatchNorm inside explicitly trainable late blocks deliberately updated affine parameters and running statistics. A state-level audit verified zero changed tensors or buffers outside declared trainable prefixes.

## Shared protocol

| Setting | Value |
|---|---|
| Dataset | `parcel_binary_v2`, unchanged train/validation split |
| Input | 224×224 RGB after resize-short-side 256 and center crop |
| Normalization | ImageNet mean/std |
| Augmentation | None |
| Epochs / batch size | 10 / 16 |
| Optimizer | Adam, two parameter groups |
| Loss | `BCEWithLogitsLoss` |
| Checkpoint | Minimum validation loss |
| Metrics threshold | 0.5, uncalibrated |
| Seed / GPU | 42 / RTX 5060 |

## Phase 10A validation results

| Metric | ResNet50 fine-tuned | EfficientNet-B0 fine-tuned | MobileNetV3 fine-tuned |
|---|---:|---:|---:|
| Accuracy | 85.47% | 87.16% | **87.67%** |
| Precision | **96.08%** | 95.14% | 94.68% |
| Recall | 81.38% | 84.95% | **86.22%** |
| F1 | 88.12% | 89.76% | **90.25%** |
| Specificity | **93.50%** | 91.50% | 90.50% |
| Balanced accuracy | 87.44% | 88.22% | **88.36%** |
| ROC-AUC | 93.68% | 93.70% | **94.13%** |
| PR-AUC | 97.25% | 97.14% | **97.33%** |
| TP / TN | 319 / 187 | 333 / 183 | 338 / 181 |
| FP / FN | **13 / 73** | 17 / 59 | 19 / **54** |
| Confusion matrix | `[[187,13],[73,319]]` | `[[183,17],[59,333]]` | `[[181,19],[54,338]]` |

None reached the ≥90% recall objective at threshold 0.5. Threshold calibration is intentionally deferred.

## Mandatory subgroup results

| Model | Ordinary damaged | Open box |
|---|---:|---:|
| ResNet50 fine-tuned | 91.25% (313/343) | 12.24% (6/49) |
| EfficientNet-B0 fine-tuned | 91.84% (315/343) | **36.73%** (18/49) |
| MobileNetV3 fine-tuned | **94.46%** (324/343) | 28.57% (14/49) |

Fine-tuning particularly harmed open-box recognition. The late layers fit ordinary visual damage strongly but did not learn the operational rule that a clean-looking open carton is damaged.

## Experiment A versus B

| Architecture | Training | Recall | Specificity | F1 | ROC-AUC | PR-AUC |
|---|---|---:|---:|---:|---:|---:|
| ResNet50 | Frozen | **86.73%** | 88.50% | **90.07%** | 93.53% | 96.97% |
| ResNet50 | Fine-tuned | 81.38% | **93.50%** | 88.12% | **93.68%** | **97.25%** |
| EfficientNet-B0 | Frozen | **90.05%** | 83.50% | **90.75%** | **94.63%** | **97.58%** |
| EfficientNet-B0 | Fine-tuned | 84.95% | **91.50%** | 89.76% | 93.70% | 97.14% |
| MobileNetV3 | Frozen | **86.99%** | 90.50% | **90.69%** | **94.89%** | **97.60%** |
| MobileNetV3 | Fine-tuned | 86.22% | 90.50% | 90.25% | 94.13% | 97.33% |

ResNet fine-tuning slightly improved ranking metrics and specificity but lost 5.36 recall points and 1.94 F1 points. EfficientNet lost 5.10 recall points and roughly 0.93 ROC-AUC points despite gaining specificity. MobileNet changed least but still declined in recall, F1, ROC-AUC, and PR-AUC.

## Overfitting evidence

- ResNet train loss fell to 0.009 by epoch 2; best validation epoch was 1.
- EfficientNet train loss fell to 0.066 by epoch 2; best validation epoch was 1.
- MobileNet train loss fell to 0.045 by epoch 2; best validation epoch was 2.
- Later validation losses generally rose while training losses approached zero.

This is classic overfitting: the model becomes extremely accurate on training examples without improving unseen validation images. The trainable scopes—especially 63.66% of ResNet and 51.96% of MobileNet—contain enough capacity to memorize a limited dataset quickly.

## Parameters and performance

| Model | Total | Frozen | Trainable | Trainable % | Best epoch/loss | Time | Latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| ResNet50 | 23,510,081 | 8,543,296 | 14,966,785 | 63.66% | 1 / 0.4200 | 197.7 s | 1.751 ms/image |
| EfficientNet-B0 | 4,008,829 | 2,878,156 | 1,130,673 | 28.20% | 1 / 0.3364 | 143.1 s | 0.755 ms/image |
| MobileNetV3 | 4,203,313 | 2,019,072 | 2,184,241 | 51.96% | 2 / 0.4334 | 145.2 s | **0.445 ms/image** |

Latency is validation GPU forward time after tensor transfer and is not used for model selection in this phase.
Checkpoint sizes remain architecture-dependent: ResNet50 89.99 MiB, EfficientNet-B0 15.58 MiB, and MobileNetV3-Large 16.23 MiB.

## Scientific conclusion

Partial fine-tuning, as predeclared here, **did not help**. The strictly frozen backbones remain stronger candidates for later validation selection and calibration. This does not prove that all fine-tuning is ineffective; a future separate experiment could test a narrower scope, stronger regularization, or smaller learning rate. Doing so now would be a new experiment, not a correction to Phase 10A.

Phase 10A stops here. No threshold calibration, Phase 10B selection, test evaluation, or severity work was started.

## Artifacts

- Script: `scripts/train_transfer_partial_finetune.py`
- ResNet: `models/transfer_learning_finetuned/resnet50_partial_finetune/`
- EfficientNet: `models/transfer_learning_finetuned/efficientnet_b0_partial_finetune/`
- MobileNet: `models/transfer_learning_finetuned/mobilenet_v3_large_partial_finetune/`
- Comparison: `models/transfer_learning_finetuned/phase10a_comparison.csv`
- Frozen-scope audit: `models/transfer_learning_finetuned/frozen_scope_audit.json`
- Reproducibility: `models/transfer_learning_finetuned/reproducibility_metadata.json`
