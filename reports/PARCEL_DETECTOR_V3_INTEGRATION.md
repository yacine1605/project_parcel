# Parcel detector v3 integration

## Source and license

The parcel-localization dataset is Roboflow Universe `parcel-detect`, version 3,
published by Test under CC BY 4.0:
<https://universe.roboflow.com/test-o3fl8/parcel-detect-p7xvl/dataset/3>.

The local raw export is
`datasets/raw/parcel-detect.v3-parcel-black_pad-320-.yolov11`. It contains 2,486
training images and 622 validation images. Its 5,207 annotations all use the
single `Boxes` class. Every image has a label file, and the audit found no
malformed coordinates. The export does not contain the test directory declared
by its original YAML, so training uses only its supplied train and validation
splits.

## Pipeline change

The inference order is now:

```text
input image
  -> parcel YOLO
  -> largest detected parcel + 4% crop padding
  -> MobileNet damaged/intact classification on crop
  -> damage YOLO localization on the same crop
  -> damage boxes translated back to original-image coordinates
```

If no parcel is localized, both condition models run on the full frame only as
fallback evidence and the final decision is forced to `REVIEW`. A localization
failure can therefore never produce an automatic `ACCEPT`.

When an image contains multiple parcels, the current database schema can store
only one image-level classifier result. The prototype consequently selects the
largest detected parcel and records all parcel boxes. True per-parcel decisions
will require a versioned database migration.

## Training

Use `scripts/train_parcel_detector.py`. The script corrects the Roboflow YAML at
runtime without modifying the raw export and saves the selected checkpoint at
`runs/PARCEL-DET-001_yolo11n_v3/weights/best.pt`.

The completed Google Colab run used YOLO11n pretrained weights, 320 px images,
50 epochs, batch size 64, seed 42, and the supplied 2,486/622 train/validation
split. Best validation mAP50–95 occurred at epoch 49:

| Precision | Recall | mAP50 | mAP50–95 |
|---:|---:|---:|---:|
| 0.88819 | 0.87144 | 0.92992 | 0.67031 |

The installed checkpoint SHA-256 is
`5ffb658923312b26521fed8035cfa36131a4968d773e5a83042ec91588ae9c0f`.
It matches the archived `weights/best.pt` byte-for-byte and exposes exactly one
class, `{0: "Boxes"}`.

These measurements are validation results from the source dataset, not an
independent warehouse test. The supplied dataset has no background-only images,
so false-positive behavior on scenes without parcels still requires external
testing.
