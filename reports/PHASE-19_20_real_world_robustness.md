# Real-World Robustness Testing — Capture Readiness Phase

## Purpose

Ordinary validation measures performance on a fixed dataset. Real-world
robustness testing asks whether the same frozen system remains reliable when
warehouse conditions change, including lighting, camera angle, distance, parcel
appearance, and damage appearance.

This phase found **no existing controlled robustness image collection or scenario
manifest**. Consequently, empirical bright/dark, distance, and angle metrics
cannot yet be reported scientifically. This report records a measurement-ready
protocol and deliberately leaves the result table empty rather than repurposing
ordinary train, validation, or final-test images.

## Frozen system under evaluation

- Classifier: frozen MobileNetV3-Large, `transfer-mobile-v1`
- Classifier threshold: `0.434584259987`
- Detector: frozen YOLO11n, experiment `EXP-002`
- Combined policy: ACCEPT only when MobileNet says intact and YOLO has zero
  detections; otherwise REVIEW
- Models execute independently

No weights, preprocessing rules, thresholds, or decision rules are changed by the
capture or evaluation tools.

## Educational distinction

Robustness requires changing one controlled environmental factor and observing
the frozen model response. For example, photographing parcel `P001` under normal,
bright, and dark lighting is more informative than using three unrelated parcels,
because the parcel condition remains constant while lighting changes.

The new manifest stores human-known ground truth. A model prediction must never be
copied into the ground-truth column.

## New reproducibility artifacts

- `scripts/capture_parcel_robustness_images.py`
- `scripts/evaluate_parcel_robustness.py`
- `datasets/robustness/parcel_robustness_v1/manifest.csv`
- `models/robustness/phase19_20_artifacts/readiness.json`

The evaluator will later create:

- `robustness_predictions.csv`
- `scenario_metrics.csv`
- `robustness_summary.json`
- annotated result images

The evaluator currently refuses to run because the manifest has zero captures.
This is the intended scientific safeguard.

## Controlled capture design

Use the same physical parcels across conditions wherever practical. A useful pilot
should contain both intact and damaged parcels and deliberately include open boxes,
because open-box recognition is a known system limitation.

| Dimension | Controlled values |
|---|---|
| Lighting | bright, normal, dark |
| View | front, side, top, 45 degrees |
| Distance | 50 cm, 100 cm, 150 cm |
| Parcel appearance | brown, white, printed, labeled/taped, varied size |
| Damage appearance | small, medium, large, open, deformation, hole where safe |

Recommended minimum pilot:

- at least 10 physical parcels, including at least 5 damaged and 5 intact;
- at least 3 repeat captures per tested condition;
- the same parcel IDs reused across controlled condition changes;
- open-box cases represented at several flap openings and viewing angles;
- approximately 120–180 images for an initial factorial subset rather than trying
  every possible combination immediately.

This is a pilot recommendation, not a claim that 120 images establishes production
reliability. A production study needs more parcels, repeated warehouse sessions,
multiple cameras, and confidence intervals planned around the required error rate.

## Capture procedure

Before each capture:

1. Assign a stable ID to the physical parcel.
2. Record damaged or intact from direct inspection.
3. Set and record one lighting, angle, and measured distance condition.
4. Record parcel appearance and visible damage appearance.
5. Keep unrelated conditions constant when testing one factor.
6. Press Space to capture, or Q to cancel without changing the manifest.

Example command:

```powershell
.\venv\Scripts\python.exe scripts\capture_parcel_robustness_images.py `
  --parcel-id P001 `
  --label damaged `
  --lighting normal `
  --view-angle front `
  --distance-cm 100 `
  --parcel-appearance brown_cardboard `
  --damage-appearance open_box `
  --damage-size medium
```

The tool creates unique image names and appends one manifest row. It never
overwrites the frozen training, validation, or test datasets.

## Evaluation procedure

After captures and human-label review:

```powershell
.\venv\Scripts\python.exe scripts\evaluate_parcel_robustness.py
```

For each image, the evaluator records MobileNet probability/label, YOLO detections,
the conservative prototype decision, and separate latency values. It reports
classification accuracy, damaged precision, damaged recall, specificity, F1, FP,
FN, and average latency overall and by scenario dimension.

Metric meanings:

- **Recall:** of all truly damaged parcels, how many were detected?
- **Precision:** of all parcels predicted damaged, how many were truly damaged?
- **Specificity:** of all truly intact parcels, how many were left intact?
- **F1:** a balance between precision and recall.

High recall reduces missed damage. High specificity reduces unnecessary manual
inspection.

## YOLO measurement boundary

The manifest can record expected damage class names, but class presence does not
say where the damage is. True YOLO precision, recall, mAP50, and mAP50–95 require
human bounding boxes for the robustness images. The evaluator therefore does not
mislabel image-level class agreement as detector mAP.

If localization robustness is required, annotate a separate copy of these captures
with detection boxes under a versioned dataset such as
`parcel_robustness_detection_v1`.

## Results status

| Scenario | Recall | Precision | Specificity | F1 | YOLO mAP | Latency |
|---|---:|---:|---:|---:|---:|---:|
| Normal | Pending captures | Pending | Pending | Pending | Requires boxes | Pending |
| Bright | Pending captures | Pending | Pending | Pending | Requires boxes | Pending |
| Dark | Pending captures | Pending | Pending | Pending | Requires boxes | Pending |
| 45 degrees | Pending captures | Pending | Pending | Pending | Requires boxes | Pending |
| Small defect | Pending captures | Pending | Pending | Pending | Requires boxes | Pending |
| New box type | Pending captures | Pending | Pending | Pending | Requires boxes | Pending |

## Current conclusion and stop condition

The project is technically ready to collect and evaluate controlled robustness
images, but it is not scientifically possible to claim robustness performance
without those physical captures and human ground truth. No synthetic transformations
were presented as real warehouse measurements, no final-test images were reused,
and no model was retrained or modified.

The next action is physical data collection using the documented protocol. Model
and system comparison should not proceed as though robustness testing were complete
until those measurements exist.
