# Parcel Inspection Project — Presentation Guide

This guide describes the current four-stage application. Phase-numbered reports preserve earlier experiments and may describe an older two-model architecture.

## Suggested seven-minute presentation

### 1. Problem and scope

“Manual parcel inspection is repetitive. This project investigates whether several specialized computer-vision models can preserve useful evidence and route uncertain cases to a human reviewer.”

State immediately that this is an educational research prototype, not a certified warehouse system.

### 2. Data and experimental method

- Damage classes: `minor_damage`, `compressed`, `hole`, and `wet`.
- Parcel-state classes: `closed_box` and `open_box`.
- Dataset versions and splits were frozen for the reported experiments.
- The damage-classifier threshold was calibrated on validation data before final evaluation.
- Checkpoint hashes and group-aware data controls support reproducibility.

Confidence is model certainty, not physical damage severity.

### 3. Current architecture

```text
phone image
  -> generic parcel YOLO
  -> one crop per detected parcel
  -> open/closed MobileNetV3-Small
       open   -> REVIEW
       closed -> damaged/intact MobileNetV3-Large
                    intact  -> closed_intact
                    damaged -> damage YOLO -> type and location
```

Explain the important engineering tradeoff: gating avoids unnecessary downstream inference, but a false negative in an earlier stage can suppress later evidence. That is why missing localization and unresolved states go to `REVIEW`, and why the combined policy still needs independent evaluation.

### 4. Evidence and decisions

Each parcel record contains its original-image box, crop box, state prediction, damage classification when executed, localized damage when executed, warnings, and timing. Explicit execution flags distinguish “not run” from a genuine intact or zero-detection result.

| Evidence | Decision |
|---|---|
| Every detected parcel is closed and classified intact | `ACCEPT` |
| Open parcel | `REVIEW` |
| Damaged classification, with or without a localized region | `REVIEW` |
| No parcel, invalid crop, or unknown state | `REVIEW` |

The prototype never issues `REJECT`.

### 5. Component results

- Damage MobileNetV3-Large final test: recall 89.29%, precision 89.06%, F1 89.17%, specificity 78.50%, ROC-AUC 93.35% on 592 images.
- Damage YOLO11n validation: precision 0.584, recall 0.561, mAP50 0.588, and mAP50–95 0.296 on 444 images / 681 instances.
- Parcel-state classifier validation: macro-F1 0.8019 on 66 reviewed crops.

Do not combine these component measurements into a system-accuracy claim. They use different datasets and protocols, and the four-stage policy has not received an independent end-to-end evaluation.

### 6. Live demonstration

Build and test the portable bundle before presenting:

```powershell
.\venv\Scripts\python.exe scripts\build_colab_demo.py
.\venv\Scripts\python.exe scripts\test_colab_bundle.py
```

Use the prepared examples to discuss:

1. a closed damaged parcel with localized evidence;
2. an open parcel that stops downstream damage inspection;
3. a first-stage parcel-detector miss routed to review;
4. a state-classifier domain-shift error routed conservatively to review.

These examples illustrate behavior; they do not prove accuracy.

### 7. Conclusion

“The main contribution is not a claim of perfect detection. It is an evidence-preserving, testable pipeline that connects data auditing, multiple model types, conservative policy logic, persistence, and a usable dashboard while documenting its limitations.”

## Claims to avoid

- “It is production ready.”
- “No detection means the parcel is safe.”
- “Confidence is damage severity.”
- “The four-stage system has the accuracy of its best component.”
- “The demo examples are an evaluation set.”
- “Colab latency is representative warehouse latency.”

## Presentation checklist

- Run the automated checks and bundle smoke test.
- Load all four components before screen sharing.
- Keep the portable ZIP available in case GitHub access fails.
- Explain at least one successful path and one failure path.
- Show the per-parcel evidence or dashboard history, including a `not run` field.
- Credit the demo-image datasets using `colab/ATTRIBUTION.md`.
