# Parcel-state YOLO training in Colab

This package trains the separate first-stage detector required by the inference
pipeline. Training happens only in Google Colab; no local training is performed.

## Required dataset

The prepared `parcel_state_dataset.zip` contains YOLO polygon annotations that
Ultralytics converts to detection boxes, with this exact class order:

```yaml
names:
  0: closed_box
  1: open_box
```

The derived dataset uses only source train/validation images. `Hand` annotations
are removed and hand-only images remain as background negatives. No source test
split is accessed. Future data should add more phone scenes, multiple parcels,
partial parcels, lighting variation, and independently captured validation data.

## Colab steps

1. Open `Parcel_State_YOLO11n_Training.ipynb` in Google Colab.
2. Select **Runtime > Change runtime type > T4 GPU**.
3. Run cells in order.
4. Upload `parcel_state_dataset.zip` when prompted.
5. Download both outputs:
   - `parcel_state_best.pt`
   - `parcel_state_training_results.zip`

Return those files to `colab/parcel_state_training/`. After validation, the
checkpoint will be installed as `models/parcel_state/best.pt`.
