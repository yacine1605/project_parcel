# EXP-002 Readiness Review

Status: **BLOCKED BEFORE TRAINING — compressed ontology review required**

## Decision

Do not start the proposed class-balancing experiment yet. The validation error
examples show that the `compressed` class is not visually consistent enough to
justify duplicating or augmenting its current examples.

## Evidence

- Some `compressed` boxes represent broad parcel deformation.
- Some represent a local puncture or tear and are plausibly predicted as `hole`.
- Box scope varies from the local damaged region to nearly the whole parcel.
- Of 30 compressed false negatives at the fixed operating point, 9 are assigned
  another damage class and 8 are localization failures. This pattern is
  consistent with class-boundary and box-policy ambiguity, not only scarcity.
- Compressed objects are already large (35.66% median normalized area), so image
  resolution is not the evidence-led next variable.

## Required review

Run `scripts/build_compressed_ontology_review.py`, then review every row in
`reports/compressed_ontology_review/compressed_review_checklist.csv` against the
generated contact sheets. Use decisions such as `keep_compressed`,
`relabel_hole`, `relabel_minor_damage`, `relabel_wet`, `exclude_unverifiable`, and
`needs_domain_review`. Tears that create an opening map to `hole`; superficial
tears without an opening map to `minor_damage`. Confirm one bounding-box policy:
local visible defect versus whole damaged parcel.

The frozen `parcel_damage_v2` dataset must not be edited. If corrections are
approved, copy it to `parcel_damage_v3`, apply logged changes there, rerun the
full dataset audit, and freeze V3 before training.

The guarded migration entry point is `scripts/create_dataset_v3_from_review.py`.
It runs validation only by default and refuses to create V3 while checklist rows
are blank, invalid, or marked `needs_domain_review`. After the review is complete,
run it with `--apply`; it will never overwrite an existing V3 directory.

For the visual review, run `scripts/review_compressed_labels.py`. The interface
opens each annotated source image, resumes at the first unfinished row, saves
every decision immediately, and supports C/H/M/X/N keyboard shortcuts. It edits
only the checklist and has no dataset-write code.

## EXP-002 after resolution

Once the ontology review passes, EXP-002 should change one variable only:
targeted balancing of the corrected `compressed` training examples. Keep
YOLO11n, 640 px, 75 epochs, batch 16, seed 42, pretrained weights, and default
Ultralytics augmentation identical to EXP-001. Select the result on validation;
do not inspect the test split until the configuration is fixed.
