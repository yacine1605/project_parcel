# Parcel detector training in Google Colab

This bundle trains the parcel-localization model required by the inspection
pipeline. The source dataset is Roboflow Universe `parcel-detect` version 3,
published by Test under CC BY 4.0:
<https://universe.roboflow.com/test-o3fl8/parcel-detect-p7xvl/dataset/3>.

## Files to upload

1. Open `Parcel_Detector_YOLO11n_Training.ipynb` in Google Colab.
2. Select **Runtime > Change runtime type > T4 GPU**.
3. Run the notebook cells in order.
4. When prompted, upload `parcel_detector_dataset_v3.zip`.

The notebook checks the 2,486 training images, 622 validation images, and the
single `Boxes` class before training. It writes a corrected runtime YAML because
the Roboflow export references a test split that was not included.

## Files to return

Download and return both notebook outputs:

- `parcel_detector_best.pt` — required by the application.
- `parcel_detector_training_results.zip` — metrics, plots, and training record.

The checkpoint will be installed locally as:

```text
runs/PARCEL-DET-001_yolo11n_v3/weights/best.pt
```

Do not rename or modify the dataset classes. The pipeline expects exactly one
class named `Boxes`.
