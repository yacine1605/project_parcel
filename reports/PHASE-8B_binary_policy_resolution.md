# Phase 8B — Binary-Label Policy Resolution

## Decision

For the primary warehouse-QC binary task, an **open box is unacceptable and maps to the binary `damaged` class**. The adjudication label `open_box` remains preserved in metadata so operational policy can be changed or analyzed later without relabeling source images.

No model was trained. No raw images, Phase 7 datasets, frozen YOLO datasets, saved models, or existing experiment artifacts were modified. Phase 8B produced metadata and review sheets only.

## Blinded second review

All **287 Phase 8A ambiguous images** from **82 groups** were shuffled with seed 808 into 12 review sheets. The sheets displayed only image pixels, `group_id`, and `image_id`; they did not expose Phase 8A labels, flags, notes, confidence, filenames, or proposed actions. Every image—not only one group representative—was reviewed before assigning a final group label.

| Second-review outcome | Groups | Images |
|---|---:|---:|
| intact | 66 | 241 |
| damaged | 2 | 8 |
| open_box | 9 | 28 |
| irreducibly ambiguous | 5 | 10 |
| **Total** | **82** | **287** |

Only the 10 irreducibly ambiguous images are newly excluded. They depict non-parcel/uncertain objects or views where parcel condition remains insufficient after reviewing all group members. No unresolved group was forced into a binary class.

## Final adjudication and binary mapping

Across all retained Phase 7 candidates after second review:

| Preserved final adjudication | Images | Warehouse-QC binary class |
|---|---:|---|
| damaged | 2,391 | damaged |
| open_box | 223 | damaged |
| intact | 1,335 | intact |
| **Included total** | **3,949** | — |

The final binary candidate pool is therefore:

| Binary class | Groups | Images | Share |
|---|---:|---:|---:|
| damaged / unacceptable | 635 | 2,614 | 66.2% |
| intact / acceptable | 532 | 1,335 | 33.8% |
| **Total** | **1,167** | **3,949** | **100%** |

The damaged:intact ratio is 1.96:1. Future training should report this imbalance and evaluate class weighting or sampling using train/validation only.

## Exclusions

| Reason | Images |
|---|---:|
| Redundant byte-identical duplicate retained from Phase 7 exclusion policy | 175 |
| Conflicting classes within one leakage-control group | 14 |
| Irreducibly ambiguous after blinded second review | 10 |
| **Total excluded from the 4,148-row source inventory** | **199** |

At group level, five irreducibly ambiguous groups and one all-excluded mixed-label group are ineligible. Exact-duplicate exclusions generally share a group with a retained canonical image, so they do not represent additional excluded identities.

## Provenance by existing splits

The original Roboflow split is retained only as provenance:

| Source split | Damaged | Intact | Total |
|---|---:|---:|---:|
| train | 2,397 | 1,152 | 3,549 |
| valid | 157 | 142 | 299 |
| test | 60 | 41 | 101 |

The Phase 7 split is also metadata only after adjudication:

| Phase 7 split | Damaged | Intact | Total |
|---|---:|---:|---:|
| train | 1,876 | 889 | 2,765 |
| valid | 371 | 222 | 593 |
| test | 367 | 224 | 591 |

Neither table should be copied blindly into the frozen v2 dataset. Construct `parcel_binary_v2` from eligible groups with deterministic seed 42, approximately 70/15/15, stratified as closely as group sizes allow. Every `group_id` must occur in exactly one new split. Freeze the test assignment before feature extraction, model selection, or threshold tuning.

## Required construction artifacts

- Final group decisions: `datasets/processed/parcel_binary_v2_review/phase8b/manifests/final_group_decisions.csv`
- Image-level inclusion/exclusion manifest: `datasets/processed/parcel_binary_v2_review/phase8b/manifests/parcel_binary_v2_inclusion_manifest.csv`
- Class-count audit: `datasets/processed/parcel_binary_v2_review/phase8b/manifests/phase8b_class_count_audit.json`
- Blinded decisions: `datasets/processed/parcel_binary_v2_review/phase8b/manifests/second_review_group_decisions.csv`
- Blinded review sheets: `datasets/processed/parcel_binary_v2_review/phase8b/blinded_second_review_sheets/`

The image-level manifest preserves `phase8a_label` and `phase8b_final_label`; open-box rows retain `phase8b_final_label=open_box` while `phase8b_binary_qc_class=damaged`. Inclusion is explicit in `phase8b_include`; excluded rows include a reason.

## Remaining limitations

The second review was blinded to prior decisions but was not an independent dual-review by a parcel-damage specialist. Hidden damage and undocumented physical-parcel identities cannot be recovered from scraped imagery. Before certification-grade v2 freezing, independently quality-control a random sample of accepted intact, damaged, and open-box groups, plus all five irreducibly ambiguous groups if additional source-resolution imagery becomes available.
