# Phase 16 — Real-World Parcel Inspection Prototype

> **Historical implementation report.** The two-model pipeline described here
> has since been replaced by the four-stage parcel-level pipeline documented in
> `docs/PARCEL_STATE_INFERENCE_ARCHITECTURE.md`.

## Outcome

A reusable prototype now accepts a local/uploaded image or optional webcam frame and independently runs the frozen MobileNetV3-Large classifier and selected EXP-002 YOLO11n detector. Neither model controls whether the other runs. No model was trained, fine-tuned, recalibrated, or modified.

## Frozen artifacts and roles

### MobileNet classifier

- Checkpoint: `models/transfer_learning/mobilenet_v3_large_frozen/best_model.pt`
- SHA-256: `f255600ad26289e7a3cc3b277926ffd91c8384a013a4203808b12d4ea77629d1`
- Frozen threshold: `0.43458425998687744`
- Preprocessing: RGB, shorter-side resize to 256, center crop 224, ImageNet normalization
- Output: one raw damaged logit, sigmoid probability, then damaged/intact label

MobileNet answers: **Does the complete image look damaged or intact?** It does not naturally provide a damage location.

### YOLO detector

- Checkpoint: `runs/EXP-002_yolo11n_ontology_corrected/weights/best.pt`
- Input size: 640
- Classes: `minor_damage`, `compressed`, `hole`, `wet`
- Prototype display confidence: `0.25`, matching the completed detector's descriptive error-analysis threshold
- Output: bounding boxes, nominal damage classes, and detection confidence

YOLO answers: **Where is visible damage, and what type does it resemble?** Confidence expresses detection certainty, not severity.

Ultralytics `8.4.129` was installed into the project virtual environment, matching EXP-002. No checkpoint was downloaded or replaced.

## Independent execution

```text
                         +--> MobileNet --> damaged probability + label
Input RGB parcel image --|
                         +--> YOLO ------> boxes + types + confidence
                                              |
                                              v
                                    combined evidence record
```

YOLO runs on every image. There is deliberately no `if classifier == damaged` gate. This is essential because final-test open-box recall for MobileNet was only 28.57%; an `intact` result cannot safely suppress independent inspection.

## Inference stages

`predict_damage_classification()` performs RGB conversion, exact frozen preprocessing, ImageNet normalization, tensor creation, batch creation, device transfer, `model.eval()` inference under `torch.no_grad()`, sigmoid conversion, and frozen-threshold comparison.

Expected classifier shapes are:

```text
image tensor: [3, 224, 224]
batch tensor: [1, 3, 224, 224]
raw logit:    [1]
```

`model.eval()` disables training-time Dropout behavior and uses saved BatchNorm statistics. `torch.no_grad()` prevents gradient storage because inference has no loss, backpropagation, optimizer, or update.

`predict_damage_regions()` passes the RGB image independently to YOLO. Ultralytics handles preprocessing, inference, and non-maximum suppression. Each `xyxy` box stores left, top, right, and bottom original-image coordinates. Zero detections is a normal result.

## Combined evidence record

Every inspection records:

- UTC timestamp and image ID;
- classifier label, probability, threshold, and raw logit;
- YOLO count, types, boxes, and confidences;
- classifier preprocessing and forward timing;
- YOLO preprocessing, inference, postprocessing, and complete-call timing;
- total pipeline time;
- prototype decision and plain-language reason;
- explicit `null` fields for severity and fusion probability.

No fake combined probability or severity score is calculated.

## Conservative prototype policy

This is an explainable **prototype policy**, not a validated production rule:

| MobileNet | YOLO detections | Result |
|---|---:|---|
| intact | 0 | `ACCEPT` |
| damaged | 0 | `REVIEW` |
| intact | 1 or more | `REVIEW` |
| damaged | 1 or more | `REVIEW` |

Disagreement always produces `REVIEW`. The prototype never outputs `REJECT`; severity and warehouse rejection policy are not scientifically established.

## Example development-image inference

The demonstration used a `parcel_damage_v3` validation image containing a visible hole. No final-test image was used.

| Output | Value |
|---|---|
| Device | NVIDIA GeForce RTX 5060 |
| MobileNet label | damaged |
| MobileNet probability | 0.9869 |
| Frozen threshold | 0.4346 |
| YOLO detection count | 1 |
| YOLO type/confidence | hole / 0.5084 |
| Prototype decision | REVIEW |

### Measured cold first-call timing

| Component | Time |
|---|---:|
| Classifier preprocessing/transfer | 3.23 ms |
| Classifier forward | 240.51 ms |
| YOLO preprocessing | 25.47 ms |
| YOLO inference | 13.31 ms |
| YOLO postprocessing | 3.20 ms |
| Complete YOLO call | 399.45 ms |
| Complete combined pipeline | 658.16 ms |

Checkpoint loading is excluded. The first inference includes one-time CUDA/kernel and Ultralytics predictor initialization effects, explaining why classifier forward time is much larger than the earlier warmed 0.899 ms batch measurement. Drawing occurs after the recorded pipeline timestamp and is excluded. This is a cold-start prototype observation, not a production benchmark. Webcam steady-state latency should later be measured over warmed repeated frames with percentile reporting.

Pure model-forward latency measures neural-network execution only. Complete end-to-end latency may include decoding, preprocessing, tensor transfer, non-maximum suppression, evidence creation, drawing, display, and storage. These scopes must not be mixed.

## Image annotation

The saved output displays MobileNet probability and label, frozen threshold, YOLO boxes/types/confidence, detection count, prototype decision, and combined processing time.

## Webcam mode

`prototype/webcam_demo.py` implements:

```text
open camera → capture BGR frame → convert to RGB
→ run MobileNet → run YOLO → draw evidence
→ convert RGB to BGR → display → q quits
```

The webcam was **not automatically opened during Phase 16**, avoiding unexpected camera activation or a blocking GUI. The code validates availability and frame capture and releases the camera in a `finally` block.

Run interactively:

```powershell
$env:YOLO_CONFIG_DIR = Join-Path (Get-Location) "Ultralytics"
.\venv\Scripts\python.exe -m prototype.webcam_demo --camera 0
```

Press `q` to exit.

## Single-image usage

```powershell
$env:YOLO_CONFIG_DIR = Join-Path (Get-Location) "Ultralytics"
.\venv\Scripts\python.exe -m prototype.single_image_demo path\to\parcel.jpg
```

JPG, JPEG, PNG, BMP, and WebP are supported. Missing files, unsupported extensions, unreadable images, missing checkpoints, class-map mismatch, and unavailable webcams produce clear errors.

## Limitations

- The ACCEPT/REVIEW policy has not undergone warehouse validation.
- MobileNet is weak on open boxes, and YOLO has no open/closed-state class.
- YOLO detects only four known damage classes and may miss unknown defects.
- The two tasks have different dataset and annotation scopes.
- A single view may hide damage; multiple cameras/views may be needed.
- No calibrated fusion rule, severity estimate, or reject policy exists.
- First-frame latency is not representative of warmed throughput.
- Lighting, viewpoint, distance, and parcel diversity require systematic testing.

## Artifacts

- `prototype/inference_classifier.py`
- `prototype/inference_yolo.py`
- `prototype/inspection_pipeline.py`
- `prototype/single_image_demo.py`
- `prototype/webcam_demo.py`
- `reports/figures/phase16_example_annotated.jpg`
- `models/prototype/phase16_example_result.json`
- `models/prototype/phase16_reproducibility.json`

## Closure

No model was retrained. No weight, checkpoint, threshold, preprocessing contract, dataset label, segmentation annotation, severity label, or final test result was changed. SQLite and Streamlit were not started.
