# Manual parcel-state review

This workflow converts the raw YOLO polygon annotations into reviewable parcel crops. It does not edit the raw dataset and does not train a model.

## Prepare the review set

```powershell
python scripts/prepare_parcel_state_classification_review.py
```

The prepared crops and persistent review manifest are written to:

```text
datasets/processed/parcel_state_classification_review/
├── candidates/train/
├── candidates/valid/
└── manifest.csv
```

Train and validation membership is preserved. Hand-only source images appear as full-image candidates with `exclude` as a suggestion, but the reviewer makes the final decision.

## Review images

```powershell
python -m streamlit run dashboard/parcel_state_labeling.py --server.port 8502
```

Open `http://localhost:8502`, then choose **Closed box**, **Open box**, or **Exclude** for each image. Decisions are saved immediately, survive an app restart, and may be changed from the **All** view.

Accepted images are copied into classifier-ready folders:

```text
datasets/processed/parcel_state_classification_v1/
├── train/
│   ├── closed_box/
│   └── open_box/
└── valid/
    ├── closed_box/
    └── open_box/
```

Excluded images remain recorded in `manifest.csv` but are not copied into either classifier class.
