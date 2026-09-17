# Open/closed parcel crop classifier — Colab training

This package trains the crop-level classifier used after the generic parcel detector. It does not train or replace either YOLO model or the existing damaged/intact MobileNet.

## Files to upload

1. Open `Parcel_State_Classifier_Training.ipynb` in Google Colab.
2. Select **Runtime > Change runtime type > T4 GPU**.
3. Run every cell in order.
4. When prompted, upload `parcel_state_classification_dataset.zip`.

The reviewed data keeps its original train/validation split:

```text
parcel_state_classification_v1/
├── train/closed_box/
├── train/open_box/
├── valid/closed_box/
└── valid/open_box/
```

The notebook uses task-specific on-the-fly augmentation, ImageNet MobileNetV3-Small preprocessing, balanced sampling, label smoothing, staged fine-tuning, early stopping, and per-class metrics. The open/closed threshold is selected on a stratified calibration subset of the training split; the original validation split is then evaluated once. It produces:

- `parcel_state_classifier_best.pt` — checkpoint, class mapping, preprocessing metadata, and calibrated threshold.
- `parcel_state_classifier_training_results.zip` — history, metrics, confusion matrix, and checkpoint.

Download both files and place them back in this directory. The project pipeline can then be updated to load the classifier checkpoint after generic parcel localization.

Augmentation includes mild perspective, rotation, translation, scale, illumination/color variation, blur, horizontal reflection, and limited random erasing. Vertical flips and severe cropping are intentionally excluded because they can distort or hide the visual evidence that distinguishes an open box from a closed box.

The validation set is still small and class-imbalanced. Treat its metrics as preliminary and later add independently captured phone-image validation data. On-the-fly augmentation improves robustness but does not create genuinely new box designs, camera environments, or packaging domains.
