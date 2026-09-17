# Phase 13 — Final System Architecture: Classifier + YOLO Integration

> **Historical design record.** This report explains the earlier two-model
> architecture. The current runnable application uses parcel localization,
> open/closed crop classification, and conditional damage inspection. See
> `docs/PARCEL_STATE_INFERENCE_ARCHITECTURE.md` for the current contract.

## Purpose and boundaries

This phase connects completed research components into a prototype design. It does not train, fine-tune, calibrate, or evaluate a model. MobileNetV3-Large, its threshold `0.434584259987`, all Phase 11 results, and every YOLO experiment remain unchanged.

The project contains **two different computer-vision tasks**:

1. **Image-level classification:** MobileNet answers, “Is this whole parcel damaged or intact?” Its output is one damaged probability and then `damaged` or `intact`. It does not naturally produce a bounding box.
2. **Object detection/localization:** YOLO answers, “Where is visible damage, and what type is it?” A detection can contain a bounding box, damage class, and confidence.

YOLO is therefore not redundant with MobileNet. Classification summarizes the image; detection locates and describes visible defects.

## Evidence that shapes the design

The frozen MobileNet final test result was 89.29% recall, 78.50% specificity, 89.06% precision, 89.17% F1, and 93.35% ROC-AUC. Ordinary damaged recall was strong at 97.96%, but open-box recall was only 28.57%. Open boxes caused 35 of 42 false negatives, and Phase 12 showed many of those errors were confident.

This matters because a sequential screening system propagates first-stage errors. If MobileNet says `intact` and the software skips YOLO, YOLO never gets an opportunity to find evidence. The current classifier must therefore **not be a hard gate** for all further inspection unless an independent mechanism reliably handles open/closed state.

The completed YOLO program established a selected recall-first YOLO11n reference on frozen detection dataset V3. Its validation measurements were 1.3 ms inference plus 0.9 ms postprocessing and 0.1 ms preprocessing per image. The frozen MobileNet Phase 11 measurement was 0.899 ms/image for GPU model-forward execution after transfer. These measurements come from different pipelines and exclude different costs; they are useful context, not a measured combined end-to-end time.

## Pipeline A — YOLO only

```text
Camera
  |
  v
YOLO detector
  |
  v
Damage class + bounding box + confidence
  |
  v
Operational decision
```

### Strengths

- One inference model makes the deployed logic easier to maintain.
- Localization and visible damage type are directly available.
- Bounding boxes help a warehouse operator understand why an item was flagged.
- No classifier error can suppress the detector.

### Weaknesses

- An image-level damaged/intact outcome must be derived from detector outputs, such as whether any qualifying detection exists. The current project has not validated a final fusion or detector-to-image rule on the binary task.
- Detection depends on the coverage and quality of bounding-box annotations and known damage classes.
- A no-detection result may mean intact, missed damage, or damage outside the detector ontology.
- The completed damage detector was not designed as a dedicated open/closed-state model, so open-box coverage cannot be assumed.

## Pipeline B — MobileNet then YOLO

```text
Camera
  |
  v
MobileNet damaged probability
  |
  +-- predicted damaged --> YOLO --> location + type
  |
  +-- predicted intact ----> stop
```

### Strengths

- MobileNet can act as a fast screening stage.
- YOLO runs on fewer images when most parcels are intact, which may reduce average GPU work.
- The image-level outcome is explicit, while suspicious images receive localization.

### Weaknesses

- A classifier false negative becomes a pipeline false negative because downstream detection never runs. This is **false-negative propagation**.
- The measured 28.57% open-box recall makes the `intact → stop` branch unsafe for the present system.
- Sequential dependencies make diagnosis harder: a missing box could result from the gate or detector.
- Average latency depends on the actual damaged referral rate; no deployment measurement currently establishes the savings.

This pipeline is not recommended as a hard-gated prototype.

## Pipeline C — MobileNet and YOLO in parallel

```text
                         +--> MobileNet --> image-level evidence ---+
Camera --> one image ----|                                      |--+--> evidence record --> decision
                         +--> YOLO ------> boxes/types/confidence --+
```

“Parallel” means neither model decides whether the other is allowed to run. It can mean simultaneous GPU scheduling if engineering permits, or simply independent execution on the same image. The safety property is independence, not literal concurrency.

### Strengths

- MobileNet provides a strong whole-image signal while YOLO preserves localization.
- A MobileNet false negative does not suppress detector evidence.
- Conflicting evidence is visible and explainable: for example, MobileNet may say intact while YOLO marks a hole.
- Both raw outputs can be stored for later audit and real-world validation.
- It is the safest current foundation for adding independent open/closed-state evidence later.

### Weaknesses

- Two models increase GPU work, integration code, version management, and monitoring.
- Combined latency has not yet been measured end to end.
- Model outputs require a fusion policy. **No arbitrary automatic rule is selected in this phase.**
- Conflicts may initially require a conservative `REVIEW` state rather than an automatic accept/reject action.

## Deployment trade-off comparison

Entries marked “inferred” are system-design expectations rather than measured production results.

| Property | YOLO only | Classifier then YOLO | Parallel classifier + YOLO |
|---|---|---|---|
| Models executed per parcel | 1 | 1, then sometimes 2 | 2 |
| Explicit image classification | Must be derived from detections | Yes | Yes |
| Damage localization | Yes | Only after positive classifier gate | Yes |
| Failure propagation | Detector miss directly affects decision | High: classifier miss suppresses YOLO | Lower: neither model suppresses the other |
| Open-box risk | Unknown; no dedicated state model | **High**, because classifier open-box recall is 28.57% | Still unresolved, but evidence is not suppressed |
| Explainability | Strong bounding-box explanation | Mixed; intact decisions have no YOLO evidence | Strongest evidence record, including disagreements |
| GPU load | Lowest of the two-model options (inferred) | Referral-rate dependent (inferred) | Highest of the three (inferred) |
| Expected latency | One detector path (inferred) | Lowest average only if referrals are uncommon (inferred) | Two inference paths; scheduling dependent (inferred) |
| Measured component timing | YOLO11n V3 validation: 0.1 ms preprocess, 1.3 ms inference, 0.9 ms postprocess | MobileNet forward 0.899 ms plus YOLO only when referred; combined not measured | Same component evidence; combined not measured |
| Complexity/maintenance | Lowest | Medium, with gate policy | Highest, with fusion and two model versions |
| Warehouse usability | Good localization, less explicit no-damage evidence | Efficient but unsafe with current gate | Best safety/audit foundation; disagreements can go to review |

## Open box is not conventional visible damage

An open parcel can have clean cardboard, no hole, no tear, no crushed texture, and no discoloration. Its signal instead comes from flap geometry, visible interior, opening width, viewing angle, and structural state. Combining it with conventional damage under one binary label asks one head to recognize visually different concepts.

Future—not Phase 13—options include:

```text
Shared visual backbone
  +--> damaged/intact head
  +--> open/closed head
  +--> damage-type detector
```

or:

```text
YOLO visible-damage detector
          +
dedicated open/closed classifier
```

A separate open/closed model would require dedicated labels, controlled open-versus-closed data, group-separated validation, threshold calibration, and a new untouched evaluation protocol. It must not be tuned against the completed Phase 11 test set.

## Recommended next prototype architecture

**Recommend Pipeline C: independently run MobileNet and YOLO, preserve both outputs, and route unresolved or conflicting evidence to review.**

This is the safest next prototype because it:

1. avoids the known dangerous hard gate;
2. preserves YOLO localization for operator explanation;
3. retains MobileNet's strong whole-image recognition of ordinary visible damage;
4. keeps each model's role conceptually understandable;
5. enables measurement of real end-to-end latency and disagreement frequency before automatic fusion is frozen.

The recommendation does **not** define an unvalidated numerical fusion rule. During prototype validation, the interface should display both outputs and conservatively treat disagreement as evidence requiring review. An operational rule may be designed only using new validation/field-study evidence, never the completed final test predictions.

## Proposed prototype data flow

```text
Camera frame
    |
    v
Parcel capture + deterministic preprocessing
    |
    +-----------------------+
    |                       |
    v                       v
MobileNet classifier       YOLO detector
probability + label        boxes + classes + confidence
    |                       |
    +-----------+-----------+
                v
          Combined evidence record
                |
                +--> future independent open/closed-state evidence
                |
                v
          Severity module (future, only with valid labels)
                |
                v
          ACCEPT / REVIEW / REJECT
                |
                v
          SQLite inspection history
                |
                v
          Streamlit dashboard
```

### What each stage does

1. **Camera/capture:** associates a parcel image with package ID and timestamp.
2. **Preprocessing:** prepares separate correctly sized and normalized inputs for MobileNet and YOLO; their pipelines must not be silently mixed.
3. **MobileNet:** produces an image-level damaged probability and frozen label.
4. **YOLO:** returns zero or more localized visible-damage hypotheses.
5. **Evidence record:** stores raw outputs, model versions, thresholds, timing, and any disagreement before a business decision.
6. **Open-box evidence:** remains a future independent component because neither current output is proven sufficient for this state.
7. **Severity:** is future work and must use genuine severity annotations; confidence is not severity.
8. **Operational decision:** maps validated evidence to accept, review, or reject. The prototype should prefer review when important evidence conflicts.
9. **Database/dashboard:** preserve inspection history and present understandable evidence to warehouse staff.

## Next roadmap step

The next roadmap phase is **Phase 14: Severity Estimation audit**. Before implementing a severity model, the project must verify whether existing annotations support physical severity labels. If they do not, Phase 14 should stop and propose an annotation strategy rather than fabricate severity from confidence or bounding-box size.

## Closure

No model was loaded, retrained, recalibrated, or modified. No checkpoint, threshold, YOLO experiment, label, prediction, or final-test result was overwritten. This phase records a system-design decision only.
