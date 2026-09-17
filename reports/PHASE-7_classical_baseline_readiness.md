# Phase 7 — Classical Baseline Readiness

Status: **BLOCKED ON GENUINE INTACT IMAGES**

## Inventory result

The frozen detection dataset cannot directly support a damaged-versus-intact
classifier: all 2,944 V3 images contain at least one damage annotation and none
has an empty label file.

The two original Roboflow exports were also inspected using `Box` and `Invoice`
as neutral annotations and every other class as damage:

| Source | Split | Neutral-only | Damage-bearing |
|---|---|---:|---:|
| Damage Package Detection v7 | train | 1 | 105 |
| Damage Package Detection v7 | valid | 0 | 18 |
| Damage Package Detection v7 | test | 1 | 12 |
| damaged_package_roboflow_v3 | train | 1 | 2,114 |
| damaged_package_roboflow_v3 | valid | 0 | 460 |
| damaged_package_roboflow_v3 | test | 0 | 448 |

These are not sufficient negatives. The apparent neutral candidates may also be
the same source image across exports. They must not be multiplied through
augmentation or treated as a usable intact class.

## Required intact collection

Collect at least 300 genuinely intact parcel images before the first HOG + SVM
baseline. Prefer 500 or more. Include:

- brown, white, and printed cardboard;
- different package sizes and aspect ratios;
- labels, tape, barcodes, and plastic wrap;
- bright, normal, and dim lighting;
- front, side, top, and oblique views;
- varied backgrounds and camera distances.

An intact image must show no visible crushing, dent, tear/opening, puncture,
moisture staining, or open flap. Ambiguous images should be excluded or marked
for review, not forced into the negative class.

## Leakage-safe capture and split policy

Assign a `package_group_id` to every physical parcel. Multiple views of the same
parcel must remain in one split. Split by package group—not by image—using an
approximate 70/15/15 train/validation/test ratio. Compute exact and perceptual
hashes before freezing the classification dataset.

Do not reuse the current detector test set for Phase 7 model selection. Create a
new classification test split and keep it closed until the HOG/SVM configuration
is selected on validation.

## Intake manifest

Use `datasets/raw/intact_collection_v1/manifest.csv`. Every row records:

`image_path,package_group_id,capture_session,lighting,view,distance_cm,review_status,notes`

Only rows with `review_status=approved_intact` may enter the negative class.

## Unblock criteria

- At least 300 approved intact images;
- at least 30 distinct physical package groups;
- no package group crosses splits;
- exact and near-duplicate review completed;
- damage and intact class counts documented;
- scikit-learn installed in the project environment;
- HOG + SVM training and evaluation then proceeds with accuracy, precision,
  recall, F1, PR-AUC, and confusion matrix.

No classical-model metric is reported at this stage because constructing
synthetic or mislabeled negatives would make the certification comparison
scientifically invalid.
