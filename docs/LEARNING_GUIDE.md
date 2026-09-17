# Parcel Damage Detection — Learning Guide

This guide is an educational index for the project. Read it in order if you want
to understand why the work progresses from data preparation to classical machine
learning, neural networks, deployment, and robustness testing.

The scripts preserve the scientific separation between training, validation, and
test data. Training learns parameters, validation selects configurations and
operating thresholds, and the protected test split provides one final estimate of
generalization.

## 1. Dataset preparation and leakage control

Before training a model, every image needs a label and a fixed split. Related or
duplicate images must not appear in different splits, because a model could appear
accurate by recognizing an almost identical image it saw during training.

Read:

- `scripts/prepare_parcel_binary_v1.py`
- `scripts/build_parcel_binary_v2.py`

Learn:

- what train, validation, and test splits mean;
- why exact hashes and perceptual hashes are different;
- why related images receive a shared group ID;
- how manifests make dataset decisions reproducible;
- why `parcel_binary_v2` is frozen and must not be rebuilt casually.

## 2. YOLO object detection

YOLO answers: **where is visible damage, and what known type does it resemble?**
It returns bounding boxes, class names, and confidence values.

Read:

- `scripts/train_exp001.py` through the selected YOLO experiment scripts
- `prototype/inference_yolo.py`

Learn:

- a bounding box approximates an object's location;
- confidence represents prediction confidence, not physical severity;
- Intersection over Union (IoU) is overlap area divided by combined area;
- mAP50 evaluates matches at IoU 0.50;
- mAP50–95 averages across stricter IoU thresholds and is harder.

## 3. HOG handcrafted features

HOG means Histogram of Oriented Gradients. It summarizes local edge directions:

```text
image -> brightness gradients -> orientation histograms -> 8,100 features
```

Read:

- `scripts/extract_hog_features_v2.py`

Learn how a manually designed feature extractor converts each image into a fixed
numeric vector before classification.

## 4. Linear SVM classification

The SVM learns a linear decision boundary in HOG feature space. Its decision score
is signed distance-like evidence, not an initial probability.

Read:

- `scripts/train_hog_svm_v2.py`
- `scripts/calibrate_hog_svm_threshold_v2.py`

Learn why the scaler is fitted on training data only, how `LinearSVC` separates
classes, and how an operating threshold changes recall and specificity without
retraining the SVM.

## 5. Custom CNN

A convolutional neural network learns image features automatically. Early layers
often learn edges and textures; deeper layers combine them into parcel-specific
patterns such as tears, holes, deformation, and openings.

Read:

- `scripts/train_custom_cnn_phase9a.py`

Learn about tensors, convolution, pooling, BatchNorm, Dropout, logits,
`BCEWithLogitsLoss`, forward passes, gradients, and validation.

## 6. Data augmentation

Training-only augmentation creates modest realistic variations such as small
rotations, translations, zoom, brightness, and contrast changes. Validation stays
deterministic so experiments are compared on the same observations.

Read:

- `scripts/train_custom_cnn_phase9b_augmented.py`

Learn why augmentation may improve robustness but does not guarantee a better
recall/specificity trade-off. Phase 9B improved raw recall while worsening
specificity and ranking metrics, so its outcome is preserved rather than hidden.

## 7. Transfer learning

An ImageNet-pretrained backbone already recognizes general visual structure. A
new classification head converts those features into one parcel-damage logit.

```text
ImageNet visual knowledge -> frozen backbone -> trained parcel-specific head
```

Read:

- `scripts/train_transfer_frozen_backbone.py`

Learn what pretrained weights, a backbone, a classification head, and
`requires_grad=False` mean.

## 8. Partial fine-tuning

Partial fine-tuning keeps early general features frozen while allowing later
features and the classification head to adapt to parcel imagery.

Read:

- `scripts/train_transfer_partial_finetune.py`
- `reports/PHASE-10A_partial_fine_tuning.md`

Learn why a lower learning rate protects useful pretrained knowledge and how
BatchNorm behavior must be deliberate. This project also demonstrates an
important scientific lesson: partial fine-tuning overfit and degraded validation
performance, so the strictly frozen models remained stronger candidates.

## 9. Threshold calibration

A threshold converts a probability or score into a label:

```text
probability 0.72 >= threshold 0.43 -> damaged
```

Lower thresholds generally increase recall and reduce specificity. Higher
thresholds generally reduce recall and increase specificity. Calibration changes
the operating decision, not neural-network weights.

Read:

- `scripts/calibrate_transfer_frozen_candidates.py`
- `scripts/calibrate_and_select_custom_cnn.py`

## 10. Final evaluation and error analysis

Metrics say how often errors occur; error analysis helps explain why. The final
test set can be inspected only after the system is permanently frozen and cannot
then be reused for iterative tuning.

Read:

- `scripts/evaluate_transfer_model_final.py`
- `scripts/analyze_final_classifier_errors.py`

Learn the confusion-matrix layout `[[TN, FP], [FN, TP]]`, why false negatives are
especially costly, and why open boxes were the main limitation.

## 11. Multi-stage inference integration

The current application first localizes parcels, then classifies each crop as
open or closed. Open boxes go directly to REVIEW. Closed crops receive the
damaged/intact classifier, and damage YOLO runs only for crops classified as
damaged. This teaches both the efficiency benefit and the failure-propagation
risk of a gated pipeline.

Read:

- `prototype/inference_classifier.py`
- `prototype/inference_parcel_detector.py`
- `prototype/inference_parcel_state.py`
- `prototype/inference_yolo.py`
- `prototype/inspection_pipeline.py`

Learn how crop coordinates map back to the original image, why skipped models
must be recorded as `not run`, and how combined evidence produces conservative
ACCEPT or REVIEW decisions without inventing severity or a fusion probability.

## 12. SQLite inspection history

SQLite stores structured inspection records, not model weights or training data.
One inspection can have zero or many linked YOLO detections. The database also
stores the full evidence JSON so the authoritative per-parcel state is preserved.

Read:

- `prototype/database.py`

Learn about tables, rows, primary keys, foreign keys, one-to-many relationships,
INSERT, SELECT, transactions, commit, and rollback.

## 13. Streamlit dashboard

Streamlit creates an interactive Python web interface. It reruns the script after
many widget interactions, so session state preserves the latest result and stops
accidental duplicate database inserts.

Read:

- `dashboard/app.py`
- `dashboard/dashboard_helpers.py`

Learn about uploads, buttons, images, metric cards, dataframes, session state,
inspection history, the REVIEW queue, and database-backed analytics.

## 14. Controlled robustness testing

Robustness testing keeps the frozen system fixed while controlled conditions such
as lighting, angle, distance, parcel appearance, and damage appearance change.

Read:

- `scripts/capture_parcel_robustness_images.py`
- `scripts/evaluate_parcel_robustness.py`

Learn why human ground truth is required, why the same physical parcels should be
repeated across conditions, and why YOLO mAP cannot be calculated without
human-annotated bounding boxes.

## Suggested learning path

Start with dataset construction, then compare HOG/SVM with the custom CNN. Move to
transfer learning only after the basic PyTorch training loop is clear. Study
threshold calibration before final evaluation, and finish with integration,
persistence, dashboarding, and robustness measurement.
