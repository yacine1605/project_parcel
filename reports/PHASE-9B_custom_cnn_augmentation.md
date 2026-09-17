# Phase 9B — Custom CNN With Realistic Training Augmentation

## Outcome

The exact Phase 9A custom CNN was trained from scratch with mild augmentation applied only to training images. Architecture, input size, split, labels, seed, optimizer, learning rate, loss, batch size, epoch budget, validation preprocessing, probability threshold, and minimum-validation-loss checkpoint rule were unchanged. The test split was not accessed.

Phase 9B achieved **91.58% damaged recall**, up 10.20 percentage points from Phase 9A, and reduced false negatives from 73 to 33. This gain cost 29.50 points of specificity and 59 additional false positives. Augmentation therefore changed the operating trade-off strongly toward recall; it did not improve general discrimination or stability.

## Exact training-only augmentations

| Augmentation | Setting | Warehouse rationale |
|---|---|---|
| Rotation | ±8° | Small camera/conveyor orientation changes |
| Translation | Up to 5% horizontally and vertically | Mild framing variation without removing most of the parcel |
| Scale | 0.95–1.05 | Small camera-distance and parcel-size variation |
| Brightness | ±15% | Mild warehouse illumination/exposure changes |
| Contrast | ±15% | Mild camera and lighting contrast changes |
| Horizontal flip | Probability 0.5 | Left/right orientation does not change damage semantics |

Vertical flips, extreme rotations, aggressive crops, and strong color changes were excluded. Validation preprocessing remained deterministic resize, tensor conversion, and normalization only.

## Frozen training configuration

| Setting | Value |
|---|---|
| Model | Phase 9A `ParcelDamageCNN`, 93,825 parameters |
| Initialization | Random; no pretrained weights |
| Input | 224×224 RGB |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss | `BCEWithLogitsLoss` |
| Batch size | 32 |
| Epochs | 20 |
| Seed | 42 |
| Checkpoint rule | Minimum validation loss |
| Best epoch | **19** |
| Best validation loss | **0.5374** |
| Training time | 410.5 seconds |
| Device | NVIDIA GeForce RTX 5060 |

The model outputs one raw logit per image. `BCEWithLogitsLoss` combines sigmoid and binary cross-entropy during loss calculation. During validation, sigmoid converts logits into damaged probabilities, and threshold 0.5 creates damaged/intact predictions.

## Validation metrics at threshold 0.5

| Metric | Phase 9B result |
|---|---:|
| Accuracy | 73.99% |
| Damaged precision | 74.79% |
| Damaged recall | **91.58%** |
| Damaged F1 | 82.34% |
| Specificity | **39.50%** |
| Balanced accuracy | 65.54% |
| ROC-AUC | 79.25% |
| PR-AUC | 86.99% |
| True positives | 359 |
| True negatives | 79 |
| False positives | **121** |
| False negatives | **33** |

Confusion matrix (rows=true and columns=predicted, intact then damaged): `[[79, 121], [33, 359]]`.

## Frozen subgroup evaluation

| Subgroup | Support | Detected | Recall |
|---|---:|---:|---:|
| Ordinary damaged | 343 | 325 | **94.75%** |
| Open box mapped to damaged | 49 | 34 | **69.39%** |

Open-box recall improved materially from 46.94% to 69.39%, a gain of 22.45 percentage points. It remains lower than ordinary-damaged recall, showing that clean-looking open parcels are still difficult for a damage-appearance model.

## Direct Phase 9A versus 9B comparison

| Metric | Phase 9A no augmentation | Phase 9B augmentation | Change |
|---|---:|---:|---:|
| Accuracy | 77.20% | 73.99% | −3.21 pp |
| Damaged precision | 83.73% | 74.79% | −8.94 pp |
| Damaged recall | 81.38% | **91.58%** | **+10.20 pp** |
| Damaged F1 | 82.54% | 82.34% | −0.20 pp |
| Specificity | 69.00% | **39.50%** | **−29.50 pp** |
| Balanced accuracy | 75.19% | 65.54% | −9.65 pp |
| ROC-AUC | 81.60% | 79.25% | −2.34 pp |
| PR-AUC | 88.83% | 86.99% | −1.84 pp |
| False negatives | 73 | 33 | −40 |
| False positives | 62 | 121 | +59 |
| Ordinary-damaged recall | 86.30% | 94.75% | +8.45 pp |
| Open-box recall | 46.94% | 69.39% | +22.45 pp |

Phase 9B is preferable only if recall is prioritized so strongly that the large false-alarm increase is acceptable. Phase 9A provides the better overall discrimination and recall/specificity balance. Accuracy alone is not used to decide this.

## Training curves and overfitting analysis

Augmentation slowed convergence: the best checkpoint moved from epoch 13 to epoch 19, and training time increased from 267.9 to 410.5 seconds. Final training loss was higher with augmentation (0.4881 versus 0.4271), which is expected because randomized inputs make the training task harder.

However, the evidence does **not** show that augmentation reduced overfitting overall:

- best validation loss worsened from 0.5011 to 0.5374;
- the train/validation loss gap at the selected epoch increased from 0.0299 to 0.0389;
- validation-loss variability across epochs was nearly unchanged (standard deviation 0.0835 versus 0.0816);
- validation recall/specificity continued to swing substantially between epochs;
- ROC-AUC and PR-AUC both declined.

The augmentation regularized training by keeping training loss higher, but this did not translate into more stable or better-ranked validation predictions. Its clearest benefit was sensitivity: it substantially reduced damaged misses, including open boxes. Its clearest cost was poor specificity.

## Scientific conclusion

- Did augmentation improve damaged recall? **Yes**, by 10.20 points.
- Did augmentation improve open-box recall? **Yes**, by 22.45 points.
- Did it improve the recall/specificity trade-off? **No**; specificity fell 29.50 points and balanced accuracy fell 9.65 points.
- Did it reduce overfitting? **Not demonstrably** under the fixed 20-epoch protocol.
- Was test accessed? **No**.

Threshold 0.5 has not been calibrated or changed. Phase 9B stops here; no transfer learning or test evaluation was started.

## Artifacts

- Training script: `scripts/train_custom_cnn_phase9b_augmented.py`
- Best checkpoint: `models/phase9b_custom_cnn/best_model.pt`
- Metadata: `models/phase9b_custom_cnn/experiment_metadata.json`
- History: `models/phase9b_custom_cnn/training_history.csv`
- Validation predictions: `models/phase9b_custom_cnn/validation_predictions.csv`
- Training curves: `models/phase9b_custom_cnn/training_curves.png`
- Augmentation examples: `models/phase9b_custom_cnn/augmentation_examples.png`
