# Phase 7A — Binary Dataset Audit

## Decision

**PASS for leakage-controlled baseline training, with unresolved label-quality limitations.** The source split was discarded. A deterministic group-stratified split (seed 42) was created without using the old YOLO held-out test set or modifying `parcel_damage_v3`.

The intact labels are **not visually certified**. Contact-sheet review found open boxes in the nominal intact class. All retained intact images therefore remain in a manual-review manifest; seven confirmed examples are flagged `open_box`. Metrics must be interpreted as performance against noisy source labels, not audited physical truth.

## Source inventory

| Source split | Damaged | Intact | Total |
|---|---:|---:|---:|
| train | 2,278 | 1,466 | 3,744 |
| valid | 149 | 153 | 302 |
| test | 51 | 51 | 102 |
| **Total** | **2,478** | **1,670** | **4,148** |

All 4,148 images are readable 640×640 JPEGs. The two unexpected non-image files are `README.dataset.txt` and `README.roboflow.txt`; both are metadata and were not copied as images. The source damaged:intact ratio is 1.48:1 (59.7% damaged), a moderate imbalance.

## Duplicate and grouping audit

- Corrupt/unreadable images: **0**.
- Files participating in exact-duplicate sets: **344** (168 hash sets: 141 damaged, 27 intact).
- Redundant byte-identical copies excluded: **175** (148 damaged, 27 intact). Canonical copies were retained.
- Conservative pHash candidate pairs (Hamming distance ≤3 and similar aspect ratio): **534**. These were grouped for leakage control, not automatically deleted or used to change labels.
- Roboflow augmentation-sibling files identified from the pre-`.rf.<hash>` name: **3,471**.
- Leakage-control groups after merging signals: **1,172**.
- Cross-class perceptual groups excluded as ambiguous label conflicts: **14 images** (11 damaged, 3 intact).
- Total excluded: **189**; total retained: **3,959**.

Generic filenames such as `images-1`, `images-8`, and numeric names recur across unrelated web images. There are 78 filename keys occurring in both classes (574 images), so filenames alone were deliberately not treated as cross-class identity evidence. Within a class, same-key images were conservatively kept together. Exact hashes and strong pHash evidence can merge across filenames/classes. Multiple views of the same physical parcel cannot be reliably inferred from this scraped dataset; only augmentation siblings and strong visual matches are controlled.

## New split

| New split | Damaged | Intact | Total | Share |
|---|---:|---:|---:|---:|
| train | 1,623 | 1,148 | 2,771 | 70.0% |
| valid | 348 | 246 | 594 | 15.0% |
| test | 348 | 246 | 594 | 15.0% |
| **Total** | **2,319** | **1,640** | **3,959** | **100%** |

Retained damaged:intact ratio is 1.41:1 in every split. Automated verification found **zero group IDs crossing splits** and **zero retained mixed-class groups**. The Phase 7 test split is a new classification split and not the old YOLO test set.

## Intact-class visual audit

Seventeen contact sheets cover all 1,640 retained intact images. The manifest records 1,633 as `pending_manual_review` and seven as `flagged_manual_review/open_box`. Open boxes are policy-sensitive: an open but otherwise undamaged box may be operationally unacceptable, yet it is not necessarily structural damage. The dataset has no trustworthy metadata for dents/compression, holes, tears, wet damage, crushed corners, open boxes, or multiple views; those cannot be certified automatically from folder labels.

Artifacts:

- `datasets/processed/parcel_binary_v1/manifests/dataset_manifest.csv`
- `datasets/processed/parcel_binary_v1/manifests/manual_review_manifest.csv`
- `datasets/processed/parcel_binary_v1/manifests/near_duplicate_pairs.csv`
- `datasets/processed/parcel_binary_v1/manifests/intact_review_sheets/`
- `datasets/processed/parcel_binary_v1/manifests/audit_summary.json`

## Exclusion policy and limitations

Nothing questionable was silently deleted: all source rows remain in the main manifest with status and reason. Raw data is unchanged. The derived dataset excludes only unreadable files (none), redundant exact copies, and mixed-label leakage groups.

The split is credible against the dominant known leakage mechanism—the documented three rotated Roboflow variants—and conservative pHash matches. It cannot guarantee physical-parcel independence for undocumented multiple views. More importantly, the intact class requires expert manual adjudication. This baseline may proceed for exploratory certification, but it is not suitable as final operational evidence until manual review is completed and the dataset is rebuilt.
