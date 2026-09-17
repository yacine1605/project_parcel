# Phase 9A — Educational Custom CNN Baseline

## Outcome

A small custom CNN was trained from scratch with PyTorch on frozen `parcel_binary_v2`. Only train and validation were used. This is a new model-family experiment after the closed Phase 8 classical evaluation; the test split was not loaded or evaluated.

The first controlled baseline deliberately uses no random augmentation. The checkpoint was selected by minimum validation loss, declared before training. Epoch 13 of 20 was selected.

| Setting | Value |
|---|---|
| Framework | PyTorch 2.13.0 + CUDA 13.0 |
| GPU | NVIDIA GeForce RTX 5060 |
| Train / validation images | 2,765 / 592 |
| Input | 224×224 RGB |
| Batch size | 32 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss | `BCEWithLogitsLoss` |
| Validation probability threshold | 0.5 |
| Trainable parameters | 93,825 |
| Training time | 267.9 seconds |

## Validation result

| Metric | Result |
|---|---:|
| Validation loss | 0.5011 |
| Accuracy | 77.20% |
| Damaged precision | 83.73% |
| Damaged recall | 81.38% |
| Damaged F1 | 82.54% |
| Specificity | 69.00% |
| ROC-AUC | 81.60% |
| PR-AUC | 88.83% |
| False positives | 62 |
| False negatives | 73 |

Confusion matrix (rows=true and columns=predicted, intact then damaged): `[[138, 62], [73, 319]]`.

At threshold 0.5, ordinary-damaged recall is 86.30% (296/343), while open-box recall is 46.94% (23/49). Like HOG, the CNN finds visually explicit structural damage more readily than clean-looking boxes that are operationally damaged solely because they are open.

# How the custom CNN works

```text
RGB image [3, 224, 224]
        ↓
Convolution Block 1 finds edges, corners, and textures
        ↓
Convolution Block 2 combines them into cardboard boundaries and folds
        ↓
Convolution Block 3 learns higher-level holes, tears, deformation, and openings
        ↓
Global average pooling summarizes 128 learned feature maps
        ↓
One output logit
        ↓
Sigmoid probability during evaluation
        ↓
Probability threshold 0.5
        ↓
Damaged / intact
```

The observed tensor shapes were:

```text
Input batch:                 [2, 3, 224, 224]
After Conv Block 1:          [2, 32, 112, 112]
After Conv Block 2:          [2, 64, 56, 56]
After Conv Block 3:          [2, 128, 28, 28]
After global average pool:   [2, 128]
Final logits:                [2]
```

A logit is an unrestricted raw network score. The model does not apply sigmoid internally because `BCEWithLogitsLoss` combines sigmoid and binary cross-entropy in a numerically stable calculation. During validation, sigmoid converts each logit into a probability from 0 to 1. Threshold 0.5 then converts that probability into a damaged or intact prediction.

During training, `model.train()` enables Dropout and BatchNorm training behavior. Each batch follows eight visible steps: load data, move tensors to GPU, clear old gradients, forward pass, calculate loss, backpropagate, update weights, and calculate metrics. `optimizer.zero_grad()` is necessary because PyTorch otherwise accumulates gradients. `loss.backward()` computes parameter gradients; `optimizer.step()` uses them to update weights.

During validation, `model.eval()` disables Dropout and makes BatchNorm use its stored statistics. `torch.no_grad()` disables gradient tracking because validation does not learn or update weights, saving memory and computation.

## HOG features versus learned CNN features

HOG uses a manually designed rule: summarize local gradient directions. It cannot change what visual evidence it extracts. The CNN instead learns convolution filters from the training images. Early filters can learn generic edges and textures; deeper filters can combine them into parcel-specific patterns.

On the same v2 validation split at each model's default threshold:

| Metric | Phase 8D HOG+SVM | Phase 9A CNN | Change |
|---|---:|---:|---:|
| Accuracy | 72.64% | 77.20% | +4.56 pp |
| Damaged precision | 80.26% | 83.73% | +3.46 pp |
| Damaged recall | 77.81% | 81.38% | +3.57 pp |
| Damaged F1 | 79.02% | 82.54% | +3.52 pp |
| Specificity | 62.50% | 69.00% | +6.50 pp |
| ROC-AUC | 77.83% | 81.60% | +3.76 pp |

This supports the hypothesis that learned visual features are more useful than fixed HOG features on validation. It is not yet a final test conclusion. The HOG calibrated threshold and CNN default threshold also serve different operating points and must not be confused.

## Scientific interpretation and next step

Training curves fluctuate substantially, which is expected for a small CNN trained from scratch on a limited and visually diverse dataset. The improving train loss with unstable validation loss suggests overfitting and sensitivity to the validation distribution. No hyperparameter was changed after viewing these results.

The next controlled experiment should add only mild, realistic training augmentation—small rotation, translation, zoom, brightness, and contrast—while leaving validation deterministic. It should retain the same architecture, optimizer, learning rate, loss, epoch budget, seed, and checkpoint rule so the augmentation effect can be measured fairly. Test must remain closed during that comparison.

## Artifacts

- Educational training script: `scripts/train_custom_cnn_phase9a.py`
- Frozen validation-selected checkpoint: `models/phase9a_custom_cnn/best_model.pt`
- Experiment metadata: `models/phase9a_custom_cnn/experiment_metadata.json`
- Training history: `models/phase9a_custom_cnn/training_history.csv`
- Validation predictions: `models/phase9a_custom_cnn/validation_predictions.csv`
- Training curves: `models/phase9a_custom_cnn/training_curves.png`
