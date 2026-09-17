# Phase 8A — Nominal Intact-Class Adjudication Audit

## Scope and result

Phase 8A audited the **1,640 retained nominal-intact images** from `parcel_binary_v1`, representing **622 Phase 7 leakage-control groups**. No model was trained. Raw data, `parcel_binary_v1`, `parcel_damage_v3`, saved models, and existing YOLO experiments were not modified.

All 622 groups received a decision. Decisions were propagated to every image in the group, producing zero pending rows and zero within-group label inconsistencies.

| Adjudicated label | Groups | Images | Share of nominal intact images |
|---|---:|---:|---:|
| intact | 466 | 1,094 | 66.7% |
| damaged | 20 | 64 | 3.9% |
| open_box | 54 | 195 | 11.9% |
| ambiguous | 82 | 287 | 17.5% |
| **Total** | **622** | **1,640** | **100%** |

The source `undamagedpackages` label is therefore not sufficiently clean to copy directly into `parcel_binary_v2`: **546 images (33.3%)** are not certified intact under the Phase 8A policy.

## Explicit label policy

- **intact:** The parcel appears closed/sealed and has no visible dent, compression, hole, tear, wet damage, crushed corner, or open flap.
- **damaged:** Visible structural or material damage exists, including dent/compression, crush, hole/puncture, tear, moisture damage/staining, or a materially crushed corner.
- **open_box:** One or more flaps/lids are visibly open, or the package is presented as an open carton, without sufficient evidence to call structural damage. Opening alone is not silently mapped to damage.
- **ambiguous:** The condition cannot be decided confidently because of resolution, occlusion, illustration/rendering, incomplete view, borderline deformation, conflicting views, or unsafe grouping.

Tape, labels, barcodes, printed graphics, handles manufactured into a box, and ordinary seams are not damage by themselves. A visibly open parcel remains `open_box` even when its cardboard appears structurally sound.

## Review method

The review unit was the Phase 7 `group_id`, not an isolated augmented image. Twenty-five contact sheets showed one representative for every nominal-intact group. Clear closed parcels with no visible defect and at most three consistent source variants were accepted as `intact` with medium confidence. Clear open-flap/lid cases and visible structural damage received high-confidence explicit overrides.

Phase 7 deliberately over-grouped same-class generic filenames to prevent leakage. A group containing more than three images may combine unrelated web images, so one representative cannot safely adjudicate every member. Such groups were assigned `ambiguous` with `group_collision_risk` unless the visible group had a clear shared open/damage finding. Other uncertain compositions, renders, partial views, and visually borderline parcels were also assigned `ambiguous` rather than forced into `intact`.

## Split distribution of adjudicated nominal-intact images

| Existing Phase 7 split | Intact | Damaged | Open box | Ambiguous |
|---|---:|---:|---:|---:|
| train | 680 | 52 | 168 | 248 |
| valid | 215 | 3 | 17 | 11 |
| test | 199 | 9 | 10 | 28 |

These split names are provenance only. `parcel_binary_v2` should be rebuilt by eligible group, deterministically with seed 42, instead of copying these counts blindly. The test split must remain closed after the v2 split is frozen.

## Manifest and v2 construction guidance

The reviewed manifest is:

`datasets/processed/parcel_binary_v2_review/manifests/reviewed_dataset_manifest.csv`

It retains every Phase 7 manifest field and adds:

- `phase8a_label`
- `phase8a_confidence`
- `phase8a_visual_flags`
- `phase8a_review_notes`
- `phase8a_reviewer`
- `phase8a_status`
- `parcel_binary_v2_action`

Recommended actions are explicit:

- `include_as_intact`: 1,094 adjudicated intact images;
- `reclassify_as_damaged`: 64 nominal-intact images with visible damage;
- `include_as_damaged`: retained Phase 7 damaged-source images (out of Phase 8A review scope);
- `exclude_pending_open_box_policy`: 195 open-box images;
- `exclude_pending_manual_re_review`: 287 ambiguous images;
- `retain_exclusion`: previously excluded Phase 7 rows.

Do not merge `open_box` into damaged or intact without an operational policy decision. For a strict damaged-vs-intact v2 baseline, the defensible default is to exclude both `open_box` and `ambiguous`, while preserving them in the manifest for future multi-class work or expert review.

## Limitations

This was systematic contact-sheet adjudication, not independent dual-review by parcel-damage specialists. Medium-confidence intact decisions are based on visible surfaces; hidden damage cannot be ruled out. Some Phase 7 filename groups are conservative collision groups, which is why they were not forced into a label. Physical-parcel identity and undocumented multiple-view relationships remain unavailable in the scraped source metadata.

Before freezing `parcel_binary_v2`, a second reviewer should prioritize the 287 ambiguous images, all open-box policy decisions, and a random quality-control sample of accepted intact and reclassified damaged groups. Inter-reviewer agreement should be recorded if certification requires human-label reliability evidence.
