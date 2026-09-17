# Phase 18 — Streamlit Warehouse Inspection Dashboard

> **Historical dashboard report.** The interface now drives the newer
> parcel-localization and state-gated pipeline. For current behavior, use the
> root `README.md` and `docs/PARCEL_STATE_INFERENCE_ARCHITECTURE.md`.

## Purpose and learning objective

Streamlit is a Python framework that creates an interactive web interface from
ordinary Python code. In this project it provides the front end for uploading a
parcel, running the already frozen models, displaying their evidence, optionally
saving the completed inspection, and reviewing SQLite history and analytics.

The dashboard contains **no training logic**. Its data flow is:

```text
Streamlit interface
        |
        v
Existing Phase 16 combined inspection pipeline
        |
        +--> frozen MobileNetV3-Large image classifier
        |
        +--> frozen YOLO11n damage detector
        |
        v
Transparent ACCEPT / REVIEW prototype decision
        |
        v
Existing Phase 17 SQLite database (only after Save Inspection)
```

MobileNet and YOLO always run independently. An intact MobileNet prediction
cannot suppress YOLO, which is especially important because open-box recall was
a known classifier weakness.

## Artifacts

- `dashboard/app.py` — four-page Streamlit application
- `dashboard/dashboard_helpers.py` — upload validation, unique image storage,
  and view formatting around existing project functions
- `scripts/test_streamlit_dashboard.py` — fast page smoke test
- `requirements-dashboard.txt` — pinned Streamlit dependency
- Existing database reused: `data/parcel_inspections.db`

No Phase 16 or Phase 17 artifact was overwritten.

## Dashboard pages

### Parcel Inspection

The main page accepts JPG, JPEG, and PNG uploads and an optional package ID. The
user explicitly clicks **Inspect Parcel** before inference runs. It shows:

- original and annotated images;
- MobileNet label, damaged probability, and frozen 43.46% threshold;
- YOLO damage classes, bounding boxes, and confidence values;
- the existing ACCEPT/REVIEW decision and its reason;
- classifier-forward, YOLO-inference, and total-pipeline latency.

Classifier probability and YOLO confidence are explicitly described as model
confidence, not physical severity. No YOLO detection is also explicitly described
as insufficient evidence to guarantee that a parcel is intact.

### Inspection History

The history page displays recent inspection ID, timestamp, package ID,
classifier result, probability, YOLO count, decision, and total latency. Selecting
one inspection shows its stored images and linked detection rows. Missing stored
images produce a readable warning rather than breaking the page.

### Review Queue

The queue uses the existing `get_review_inspections()` query and lists parcel ID,
timestamp, YOLO classes, and classifier probability for all REVIEW records.

### Analytics

Analytics use only genuine SQLite rows and provide:

- total inspections;
- ACCEPT count;
- REVIEW count;
- average total pipeline latency;
- YOLO damage-class counts;
- daily inspection counts.

The interface does not invent REJECT, severity, or production KPIs. When only one
day exists, it labels the chart as a count rather than claiming a meaningful trend.

## Streamlit reruns and duplicate prevention

Streamlit reruns the script from top to bottom after many widget interactions.
`st.session_state` therefore preserves the latest result, its paths, and whether
it has already been saved. A separate **Save Inspection** button performs the
database insert once. After insertion, the saved flag and SQLite row ID remain in
session state, so routine rerenders do not duplicate that inspection.

Heavy model objects use `st.cache_resource`, allowing checkpoints to be loaded
once per Streamlit process rather than once per widget interaction. Caching does
not change model weights or output.

## Safe image handling

Uploads are decoded through Pillow and converted to RGB. Empty, corrupt, and
unsupported files receive concise errors. Original and annotated images are saved
as PNG files under separate folders in `data/dashboard_images/`, using a UTC
timestamp plus a UUID. Existing images cannot be silently overwritten. SQLite
stores their paths rather than large image blobs.

## Prototype policy

The dashboard displays the unchanged Phase 16 policy:

| Classifier evidence | YOLO evidence | Decision |
|---|---|---|
| intact | zero detections | ACCEPT |
| damaged | one or more detections | REVIEW |
| disagreement | either direction | REVIEW |

This is an explainable prototype rule, not a validated production policy. It
never outputs REJECT and creates no synthetic fusion probability.

## Verification

Streamlit 1.62.0 was installed in the existing virtual environment. Verification
used development data only; no frozen final-test image was opened.

1. Python syntax compilation passed for both dashboard modules.
2. Streamlit's `AppTest` opened all four pages without an application exception.
3. A training-split development image was passed through the same upload helper
   used by the UI.
4. Frozen MobileNet and YOLO both executed independently on the RTX 5060.
5. Original and annotated images received unique paths.
6. The result was saved to a temporary SQLite database and read back correctly.
7. The temporary database reported exactly one inspection, confirming save and
   analytics integration without changing the operational database.
8. The existing operational database powered history, REVIEW, and analytics views.
9. Missing-image handling is implemented through an existence-check helper and
   a user-facing warning.

Development-check example (not a test-set result):

```text
Decision: REVIEW
MobileNet damaged probability: 96.03%
YOLO detections: 1
Total first-run pipeline time: 495.78 ms
Device: NVIDIA GeForce RTX 5060 (CUDA)
```

The timing includes model/GPU warm-up and the complete combined call. It is not a
certified production benchmark.

## Running the dashboard

From the project root in PowerShell:

```powershell
.\venv\Scripts\python.exe -m streamlit run dashboard\app.py
```

To install the dashboard dependency in a fresh copy of the existing environment:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements-dashboard.txt
```

To rerun the fast page smoke test:

```powershell
.\venv\Scripts\python.exe scripts\test_streamlit_dashboard.py
```

## Known prototype limitations

- This is a local single-user prototype, not an authenticated warehouse service.
- Session state is browser-session memory, not durable workflow state.
- ACCEPT/REVIEW has not been validated as a production decision policy.
- Open-box state still lacks an independent specialized model.
- The dashboard supports uploaded images but does not embed the Phase 16 live
  webcam loop; webcam inference remains available through the existing demo.
- Latency varies with warm-up, hardware state, and concurrent load.
- Files are stored locally; retention, access control, and backup policies are not
  yet implemented.
- Screenshots were not generated automatically because the headless smoke test
  validates functionality without launching a GUI browser.

## Scientific boundary

No model was retrained, fine-tuned, recalibrated, or otherwise changed. The frozen
MobileNet threshold and YOLO inference configuration remain unchanged. Phase 18
adds only a user interface and database presentation layer.
