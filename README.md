# Parcel Inspection: A Four-Stage Computer Vision Prototype

This educational project explores how several specialized vision models can cooperate in a conservative parcel-inspection workflow. It localizes every parcel in a phone image, classifies each crop as open or closed, checks closed parcels for damage, and optionally localizes visible defect regions.

The system produces only two operational decisions:

- `ACCEPT` when every detected parcel is closed and classified as intact.
- `REVIEW` for open parcels, visible damage, missing localization, invalid crops, or any unresolved evidence.

It never issues an automatic rejection and never treats model confidence as physical damage severity. This is a research and learning project, not a validated warehouse safety system.

## Current architecture

```text
phone image
    |
    v
generic parcel detector (YOLO11n)
    |
    +--> one padded crop per parcel
             |
             v
      open/closed classifier (MobileNetV3-Small)
             |
             +--> open ------> REVIEW; downstream damage models are not run
             |
             +--> closed ----> damaged/intact classifier (MobileNetV3-Large)
                                      |
                                      +--> intact  --> closed_intact
                                      |
                                      +--> damaged --> damage detector (YOLO11n)
                                                           |
                                                           +--> boxes and damage types
```

The structured result records which models actually ran. A skipped model is displayed as `not run`; it is not silently converted into an intact prediction.

For the detailed contracts and thresholds, see [Parcel-state inference architecture](docs/PARCEL_STATE_INFERENCE_ARCHITECTURE.md).

## Why this project is educational

The repository demonstrates more than model training:

- dataset auditing, ontology correction, and leakage-aware splitting;
- classical HOG/SVM and scratch-CNN baselines;
- transfer learning and validation-only threshold calibration;
- object detection and coordinate conversion between crops and full images;
- conservative human-in-the-loop policy design;
- SQLite persistence with one-to-many detection records and full JSON evidence;
- a Streamlit inspection, history, review, and analytics interface;
- reproducible smoke tests and checkpoint integrity checks;
- honest documentation of model failures and domain-shift limitations.

## Quick start

### 1. Create an environment

Python 3.11 or newer is recommended. Install a PyTorch/torchvision pair appropriate for your CPU or CUDA environment using the [official PyTorch selector](https://pytorch.org/get-started/locally/), then install the application dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-colab.txt
python -m pip install -r requirements-dashboard.txt
```

The four inference checkpoints listed below must be present. The curated Git repository is configured to include them.

### 2. Inspect one image

```powershell
.\.venv\Scripts\python.exe -m prototype.single_image_demo path\to\parcel.jpg
```

For a folder of phone images, with optional intermediate debug images:

```powershell
.\.venv\Scripts\python.exe scripts\inspect_phone_images.py path\to\images --debug
```

### 3. Run the dashboard

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard\app.py
```

The dashboard stores successful uploads under `data/dashboard_images/`. Saving an inspection is a separate user action. SQLite records model-execution flags plus the full per-parcel evidence JSON, so an open parcel is not misreported as a MobileNet `intact` result.

## Model artifacts

| Stage | Artifact | Purpose |
|---|---|---|
| Parcel localization | `runs/PARCEL-DET-001_yolo11n_v3/weights/best.pt` | Finds generic parcels in the full image |
| Parcel state | `models/parcel_state_classifier/best.pt` | Classifies parcel crops as `closed_box` or `open_box` |
| Damage classification | `models/transfer_learning/mobilenet_v3_large_frozen/best_model.pt` | Classifies closed crops as damaged or intact |
| Damage localization | `runs/EXP-002_yolo11n_ontology_corrected/weights/best.pt` | Localizes `minor_damage`, `compressed`, `hole`, and `wet` |

`scripts/build_colab_demo.py` verifies the SHA-256 fingerprint of every checkpoint before creating the portable demo bundle.

## Run the checks

These checks do not retrain a model or access the frozen final test set:

```powershell
.\.venv\Scripts\python.exe -m compileall -q prototype dashboard scripts
.\.venv\Scripts\python.exe scripts\test_parcel_state_pipeline.py
.\.venv\Scripts\python.exe scripts\test_parcel_state_review.py
.\.venv\Scripts\python.exe scripts\test_inspection_database.py
.\.venv\Scripts\python.exe scripts\test_streamlit_dashboard.py
.\.venv\Scripts\python.exe scripts\build_colab_demo.py
.\.venv\Scripts\python.exe scripts\test_colab_bundle.py
```

## Repository guide

| Path | What to learn there |
|---|---|
| `prototype/` | Reusable model loading, inference, policy, annotation, and persistence code |
| `dashboard/` | Streamlit user interface and upload validation |
| `scripts/` | Dataset preparation, training, evaluation, audit, and smoke-test entry points |
| `colab/` | Portable inference demo and documented training notebooks |
| `docs/LEARNING_GUIDE.md` | Guided tour through the machine-learning concepts |
| `docs/PARCEL_STATE_INFERENCE_ARCHITECTURE.md` | Authoritative current inference design |
| `reports/` | Chronological experiment reports; earlier reports describe historical architectures |

The root README and the parcel-state architecture document describe the current application. Phase-numbered reports are retained as an experiment history and should not be interpreted as the latest runtime contract.

## Reported component results

These measurements come from different datasets and protocols; they must not be combined into a single system-accuracy claim.

| Component | Evaluation | Result |
|---|---|---|
| Damage MobileNetV3-Large | Frozen final test, 592 images | Recall 89.29%, precision 89.06%, F1 89.17%, specificity 78.50%, ROC-AUC 93.35% |
| Damage YOLO11n (`EXP-002`) | Validation, 444 images / 681 instances | Precision 0.584, recall 0.561, mAP50 0.588, mAP50-95 0.296 |
| Parcel-state classifier | Validation, 66 reviewed crops | Macro-F1 0.8019; preliminary because the set is small |

The complete four-stage policy has not been evaluated on an independent warehouse dataset.

## Known limitations

- The generic parcel detector can miss parcels outside its training domain.
- The state classifier was evaluated on only 66 validation crops and can route closed parcels to review.
- Gating damage YOLO behind the damaged/intact classifier can suppress localization after a classifier false negative.
- No detection or high confidence is proof of physical safety.
- The stored `ACCEPT`/`REVIEW` policy is a prototype policy, not an operational certification.
- Latency depends on hardware, CUDA warm-up, image size, and system load.

These limitations are deliberately visible in the demo and evidence records.

## Data and attribution

Prepared examples are derived from CC BY 4.0 datasets and are documented in [colab/ATTRIBUTION.md](colab/ATTRIBUTION.md). Large raw datasets, generated runs, local databases, virtual environments, and caches are excluded from Git.

Before accepting outside contributions or reuse, add an explicit software license chosen by the repository owner.
