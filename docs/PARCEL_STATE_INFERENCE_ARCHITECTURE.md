# Parcel-state inference architecture

## Processing order

```text
Phone image
  -> generic parcel YOLO (Boxes + bounding boxes)
  -> each parcel crop
  -> crop-level state classifier (closed_box/open_box)
       open_box   -> stop and report open_box
       closed_box -> frozen MobileNet (intact/damaged)
           intact  -> report closed_intact
           damaged -> frozen damage YOLO (type + location)
```

The generic detector is the only full-image model and is preserved because it
generalizes better to phone images and white mailer boxes. The state model is
not used for primary localization. MobileNet and damage YOLO never receive the full phone image. Both receive the
same padded parcel crop, preserving MobileNet's frozen resize, center-crop, and
ImageNet normalization. Parcel crops reduce background influence, standardize
downstream inputs, improve phone-image robustness, and allow every parcel in an
image to be processed separately.

Open parcels skip MobileNet and damage YOLO. Closed-intact parcels skip damage
YOLO. A damaged MobileNet result remains damaged even when YOLO finds no region;
the result includes `classified_damaged_but_no_damage_region_detected`.

## Models and thresholds

| Component | Configuration |
|---|---|
| Generic parcel weights | `runs/PARCEL-DET-001_yolo11n_v3/weights/best.pt` |
| Generic parcel confidence | `GENERIC_PARCEL_CONF = 0.25` in `prototype/inference_parcel_detector.py` |
| Crop-level state weights | `models/parcel_state_classifier/best.pt` |
| Required parcel classes | `closed_box`, `open_box` |
| Open/closed threshold | Calibrated value stored in the classifier checkpoint; optional `PARCEL_STATE_OPEN_THRESHOLD` environment override |
| Crop padding | `PARCEL_CROP_PADDING = 0.04` in `prototype/inference_parcel_detector.py` |
| MobileNet threshold | Loaded unchanged from `models/selected_transfer_model/selected_transfer_model.json` |
| Damage YOLO confidence | `PROTOTYPE_YOLO_CONFIDENCE = 0.25` in `prototype/inference_yolo.py` |

The state-model path is project-relative and can also be supplied directly
to `load_parcel_state_classifier(checkpoint_path=...)`. At startup, its class order
is validated.

The installed augmented crop classifier was selected at epoch 4 and uses an
open-box probability threshold of `0.22`. The threshold was selected on a
stratified calibration subset of the training data, after which the original
validation split was evaluated once. Its validation macro-F1 is `0.8019`.
Closed-box precision/recall/F1 are `0.8571`/`0.6000`/`0.7059`; open-box
precision/recall/F1 are `0.8462`/`0.9565`/`0.8980`. Compared with the prior
checkpoint, it reduces safety-critical open-to-closed errors from 3 to 2 but
increases closed-to-open review routing from 1 to 8. These results use only 66
source-validation crops, not an independent phone/warehouse test set, so
deployment confidence is not yet established. The prior checkpoint is preserved
at `models/parcel_state_classifier/best_pre_augmentation.pt` for rollback.

## Coordinates and result schema

Crop-relative damage boxes are translated back by adding the padded crop's
`x1` and `y1` offsets to both corners. The structured output contains a
`parcels` list; each entry preserves parcel box, state and confidence, MobileNet
status/probability when applicable, damage boxes/classes/confidences, warnings,
and per-stage timing. `status` is `no_parcel_detected` when the list is empty,
and no downstream model runs in that case.

## Run inference

One phone image:

```powershell
.\venv\Scripts\python.exe scripts\inspect_phone_images.py path\to\phone.jpg
```

A folder (models load once for the whole batch):

```powershell
.\venv\Scripts\python.exe scripts\inspect_phone_images.py path\to\phone_images --output prototype_outputs\batch
```

Add `--debug` to save `original.jpg`, every `parcel_N_crop.jpg`,
`parcel_state_predictions.jpg`, and `final_predictions.jpg`. Debugging is off by
default.

## Compatibility

The SQLite schema predates multi-parcel state results. The pipeline retains its
legacy aggregate fields for compatibility and now stores explicit
`classifier_executed` and `damage_detector_executed` flags plus the complete
evidence JSON. The authoritative detail is the per-parcel `parcels` list; skipped
models are shown as `not run`, never as inferred intact predictions. A normalized
parcel table can be added later if per-parcel fields need direct SQL queries.
