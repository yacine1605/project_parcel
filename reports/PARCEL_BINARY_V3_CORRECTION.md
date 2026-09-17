# Parcel Binary V3 — Open-box label correction

## Outcome

`parcel_binary_v3` was constructed from frozen `parcel_binary_v2` without
modifying v2. Manual group decisions from the open-box audit were propagated only
after all 223 members of the 63 audited groups were checked for visual group
consistency.

## Corrections

- Audited groups: 63
- Audited images: 223
- Groups changed from binary damaged to intact: 39
- Images changed from binary damaged to intact: 163
- Reviewed open-box groups retained as damaged: 20 groups / 50 images
- Reviewed minor-damage groups retained as damaged: 4 groups / 10 images

## V3 inventory

| Split | Damaged | Intact | Total |
|---|---:|---:|---:|
| Train | 1,732 | 1,033 | 2,765 |
| Validation | 354 | 238 | 592 |
| Test | 365 | 227 | 592 |
| **Total** | **2,451** | **1,498** | **3,949** |

Integrity checks passed:

- 3,949 manifest rows and 3,949 image files;
- 3,949 unique output paths;
- 1,167 unique groups;
- zero missing files or target hash mismatches;
- zero group cross-split leakage;
- zero exact-hash cross-split leakage.

## Evaluation boundary

The v2 test results and error contact sheets influenced the decision to audit
these labels. V3 therefore preserves the old splits only for provenance and
development comparison. Its `test` directory is not a new untouched final test
for a retrained model. Any new final performance claim requires independently
collected external test data with reviewed ground truth.

## Artifacts

- Dataset: `datasets/processed/parcel_binary_v3/`
- Manifest: `datasets/processed/parcel_binary_v3/manifests/dataset_manifest.csv`
- Audit summary: `datasets/processed/parcel_binary_v3/manifests/audit_summary.json`
- Builder: `scripts/build_parcel_binary_v3.py`
- Manual decisions: `reports/open_box_label_audit/open_box_group_audit.csv`
