# Parcel Damage Dataset v1 — Phase 3 Audit

Generated: `2026-08-26T21:18:44.569966+00:00`
Dataset: `C:\Code\python\Project Card board\datasets\processed\parcel_damage_v1`

## Freeze decision

**NOT FROZEN — STOP BEFORE TRAINING**

- The 11 marginally out-of-bounds boxes are explicitly accepted for Dataset V1 and are not blockers.
- Class imbalance is intentional for EXP-001 and is not a blocker.
- The broad `minor_damage` ontology is accepted as a V1 limitation and is not a blocker.
- Visual review confirmed 3 true cross-split duplicate pairs. Removal requires approval and a corrected dataset version.

## Classes

`nc: 4`

| ID | Name | Instances | Images | Instance share |
|---:|---|---:|---:|---:|
| 0 | minor_damage | 251 | 249 | 5.54% |
| 1 | compressed | 274 | 272 | 6.05% |
| 2 | hole | 3011 | 1889 | 66.50% |
| 3 | wet | 992 | 635 | 21.91% |

## Split validation

| Split | Images | Labels | Missing | Orphan | Empty | Corrupt | Boxes | Zero-box | Multi-class | Avg boxes/image |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 2081 | 2081 | 0 | 0 | 0 | 0 | 3177 | 0 | 65 | 1.527 |
| valid | 446 | 446 | 0 | 0 | 0 | 0 | 685 | 0 | 18 | 1.536 |
| test | 420 | 420 | 0 | 0 | 0 | 0 | 666 | 0 | 15 | 1.586 |

## Bounding boxes

Small `<1%`, medium `1–25%`, large `≥25%` of image area.

| Class | Mean area | Median area | Min | Max | Small | Medium | Large |
|---|---:|---:|---:|---:|---:|---:|---:|
| minor_damage | 0.473795 | 0.512689 | 0.008031 | 0.878201 | 1 | 58 | 192 |
| compressed | 0.342823 | 0.337537 | 0.008411 | 0.702319 | 1 | 85 | 188 |
| hole | 0.076387 | 0.034859 | 0.000910 | 0.821045 | 348 | 2466 | 197 |
| wet | 0.220913 | 0.169766 | 0.006760 | 0.872859 | 6 | 619 | 367 |

## Image resolutions

- **train:** width 640–640 px; height 640–640 px. Common: 640x640 (2081)
- **valid:** width 640–640 px; height 640–640 px. Common: 640x640 (446)
- **test:** width 640–640 px; height 640–640 px. Common: 640x640 (420)

## Leakage check

- Exact duplicate groups: 0
- Exact cross-split groups: 0
- Potential near-duplicate cross-split pairs (64-bit dHash distance ≤5, excluding exact pairs): 23
- Visual verdicts: 3 true duplicates and 20 false positives. See `near_duplicate_review.md`.
- No files were automatically deleted.

## Annotation issues

| Type | Count |
|---|---:|
| box_outside_image | 11 |

Full file-level findings are preserved in `dataset_v1_stats.json`.

## Figures

- `figures/class_distribution.png`
- `figures/bbox_area_distribution.png`
- `figures/image_resolutions.png`
- Per-class and special-case labeled sample grids under `figures/`.

## Known limitations

- The automated audit cannot verify semantic label correctness, lighting difficulty, background clutter, or package orientation; sample grids require human inspection.
- dHash is a screening method. Flagged near duplicates need visual confirmation, and visually similar images can evade it.
- `minor_damage` is broader than the intended target ontology. This is accepted for V1 and will not be relabeled before EXP-001.
- Class imbalance is intentional for the first baseline: hole 66.50%, wet 21.91%, compressed 6.05%, minor_damage 5.54%.
- Eleven marginally out-of-bounds boxes are accepted for V1.
