# Phase 8C — Parcel Binary V2 Construction and Freeze

## Outcome

`parcel_binary_v2` was built from the Phase 8B inclusion manifest and is **frozen for experiments**. The source inventory, `parcel_binary_v1`, review artifacts, YOLO datasets, and prior experiment outputs were not modified.

Open boxes retain `phase8b_final_label=open_box` in the manifest and map to binary `damaged` under the warehouse-QC policy. The 199 excluded source rows remain documented in the Phase 8B manifest and were not copied.

## Frozen split

Groups were assigned deterministically with seed 42, targeting 70/15/15 separately by binary class. Every leakage-control `group_id` occurs in one split only.

| Split | Damaged | Intact | Total | Share |
|---|---:|---:|---:|---:|
| Train | 1,830 | 935 | 2,765 | 70.02% |
| Validation | 392 | 200 | 592 | 14.99% |
| Test | 392 | 200 | 592 | 14.99% |
| **Total** | **2,614** | **1,335** | **3,949** | **100%** |

The dataset contains 1,167 groups. The overall damaged:intact image ratio is 1.96:1 and must be reported during modeling.

## Freeze audit

- Manifest rows: 3,949
- Copied image files: 3,949
- Unique image IDs: 3,949
- Unique output paths: 3,949
- Readability failures: 0
- Group IDs spanning splits: 0
- Exact image hashes spanning splits: 0

The test assignment is now closed. Feature extraction, model selection, class weighting, and threshold calibration must use train and validation only. Test may be evaluated once after the complete model and decision threshold are selected.

## Artifacts

- Dataset: `datasets/processed/parcel_binary_v2/`
- Full provenance manifest: `datasets/processed/parcel_binary_v2/manifests/dataset_manifest.csv`
- Machine-readable freeze audit: `datasets/processed/parcel_binary_v2/manifests/audit_summary.json`
- Reproducible builder: `scripts/build_parcel_binary_v2.py`

The builder refuses to overwrite an existing v2 directory and deletes partial output on failure. Any future label, policy, grouping, or split change requires a new dataset version.

## Next experiment

Extract HOG features for v2, retrain/select the linear-SVM family using train/validation only, and calibrate its decision threshold on validation for a predefined damaged-recall target (for example, at least 90%). Report the resulting specificity and precision tradeoff before opening the frozen test split.
