# Preliminary open-box label audit

## Scope

The frozen `parcel_binary_v2` manifest contains 223 included images labelled
`open_box`, representing 63 leakage-control groups. This audit reviewed one
representative image from every group. It did not modify the frozen manifest,
images, splits, checkpoints, predictions, or reported metrics.

## Manual representative review results

| Representative review label | Groups | Images represented |
|---|---:|---:|
| open_box | 20 | 50 |
| intact | 39 | 163 |
| minor_damage | 4 | 10 |
| damaged | 0 | 0 |
| ambiguous | 0 | 0 |
| **Total** | **63** | **223** |

After manual correction, only 20/63 groups (31.7%) are labelled open boxes from
the representative view. They account for 50/223 images (22.4%), although that
image-weighted percentage is strongly affected by augmentation group sizes.
The remaining 43/63 groups (68.3%) are labelled intact or minor damage.

## Interpretation

This is strong evidence of systematic label noise in the preserved `open_box`
subgroup. Consequently, the published 28.57% MobileNet open-box test recall
cannot be interpreted purely as failure to recognize genuinely open boxes.
Incorrect and ambiguous ground truth contributes to the measured result.

All 223 members were subsequently displayed in 12 group-ordered sheets and
visually checked. Every group was internally consistent: its members depicted the
same source scene or an augmentation/near-duplicate. The reviewed group label can
therefore be propagated to every member without splitting a leakage-control group.

## Artifacts

- `open_box_group_audit.csv`: one row per group with the preliminary label.
- `open_box_groups_01.jpg` through `open_box_groups_04.jpg`: representative sheets.
- `open_box_all_members_01.jpg` through `open_box_all_members_12.jpg`: all 223 images.
- `open_box_group_member_validation.csv`: member-level consistency result per group.
- `scripts/audit_open_box_groups.py`: reproducible artifact generator.

## Required next step

Construct a new versioned dataset from the reviewed group decisions. Do not alter
frozen `parcel_binary_v2`. Before training, run integrity checks for row counts,
file hashes, group-exclusive splits, and corrected class totals.
