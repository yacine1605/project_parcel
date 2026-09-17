# Phase 14 — Severity Estimation Annotation-Support Audit

## Outcome

| Audit question | Answer |
|---|---|
| Does the current dataset contain explicit severity labels? | **NO** |
| Can reliable severity labels be derived without guessing? | **NO** |
| Is manual annotation required? | **YES** |
| Should a severity model be trained now? | **NO — STOP** |

The current data supports damage classification, detection, localization, nominal damage type, and reviewed open-box state. It does not contain ground truth for how serious the physical damage is. No severity model was trained.

## What severity means

Severity answers:

> How serious is the physical damage to the parcel?

Examples include cosmetic, minor, significant, and severe. Severity is not model confidence, detection confidence, classification probability, box area alone, or the number of detections alone.

A model can be 99% confident that a tiny hole exists; the `99%` describes certainty, not physical seriousness. Conversely, a model can have low confidence in a genuinely severe but unfamiliar failure. Bounding-box geometry describes where an annotator drew a rectangle, not whether contents are safe.

## Classification, damage type, and ordinal severity

Binary classification uses two nominal labels:

```text
intact / damaged
```

Damage detection uses nominal type classes:

```text
minor_damage / compressed / hole / wet
```

Nominal means the names are different categories without a universal numerical order. `hole` is not mathematically “greater than” `wet`.

A proposed severity scale is ordinal:

```text
0 < 1 < 2 < 3 < 4
```

Ordinal means the levels have an order from less serious to more serious, even though the distance between levels is not necessarily equal. A damage type cannot automatically be converted to one ordinal value: a tiny sealed puncture and a large rupture are both holes but may have different consequences.

## Audit scope

The audit inspected:

- `parcel_damage_v3` class mapping and YOLO annotation schema;
- YOLO **train and validation** labels only;
- `parcel_binary_v2` manifest schema and train/validation rows only;
- frozen dataset documentation, ontology review records, class mappings, and source documentation;
- existing review flags and notes.

The final classification test images, test labels, and test predictions were not accessed. The YOLO test labels were also explicitly excluded from the area analysis.

## Existing severity-related signals

| Possible signal | Exists? | Reliable? | Direct or proxy? | Notes |
|---|---|---|---|---|
| Explicit severity label | No | No | Missing | No severity/grade column or 0–4 annotation exists. |
| Damage type | Yes | Suitable as reviewed nominal ontology | Context/proxy | Four detector classes; the same type can span multiple severity levels. |
| Bounding-box area | Yes | Geometry is measurable but annotation-dependent | Proxy only | Useful as one possible feature; not physical seriousness. |
| Number of damage regions | Yes | Annotation-dependent | Proxy only | Multiple regions may reflect repeated small defects or annotation style. |
| Open-box state | Yes | Reviewed subgroup metadata | State/context | Open/closed state is not itself a complete severity grade. |
| Damage localization | Yes | Available for four YOLO classes | Context/proxy | Indicates where annotated damage appears. |
| Damaged/intact label | Yes | Frozen binary task label | Nominal class | Does not grade damaged samples. |
| Review notes/visual flags | Partly | Created for binary adjudication | Context only | Examples include deformation and open flap; not a complete severity rubric. |
| Operational accept/review/reject | No | No | Missing | No warehouse disposition ground truth exists. |
| Structural-integrity grade | No | No | Missing | No direct assessment of safe usability. |
| Contents-exposure extent | No | No | Missing | Not systematically annotated. |
| Wetness/opening/deformation extent | No | No | Missing | Type/location may exist, but extent and consequence are not graded. |

The detector class `minor_damage` contains a severity-like word, but it is one nominal category beside `compressed`, `hole`, and `wet`. It does not assign a comparable severity level to every parcel and cannot serve as a complete severity target.

## Bounding-box area audit

For normalized YOLO annotations:

```text
relative_box_area = normalized_width × normalized_height
```

This is approximately the fraction of the image covered by the rectangular annotation. It is not necessarily the fraction of the parcel surface that is physically damaged, because a rectangle includes background and undamaged pixels inside its boundaries.

The audit read 3,858 development boxes: 3,177 from train and 681 from validation. Test annotations were excluded.

| Damage type | Boxes | Median area | Middle 50% | 10th–90th percentile |
|---|---:|---:|---:|---:|
| minor_damage | 266 | 0.449 | 0.235–0.649 | 0.124–0.746 |
| compressed | 181 | 0.353 | 0.233–0.487 | 0.135–0.560 |
| hole | 2,572 | 0.035 | 0.017–0.085 | 0.009–0.192 |
| wet | 839 | 0.161 | 0.065–0.342 | 0.033–0.487 |

The distributions differ but overlap, and the class counts are highly imbalanced. Holes dominate the annotation count, while compressed has only 181 boxes. These are **box counts**, not independent physical parcels or severity-label counts.

Bounding-box area may become one input feature after severity ground truth exists. It cannot create that ground truth. A small puncture may expose or compromise contents; a large loose rectangle may surround mild deformation; and annotation conventions differ by damage type.

## Damage type and operational risk

Different types can suggest different risks, but type does not determine seriousness:

- a small deformation may be cosmetic, while severe crushing may destroy structural integrity;
- a small hole may or may not expose contents;
- wetness may be localized or extensive and may affect different contents differently;
- an open box may have a narrow flap gap or fully exposed contents;
- multiple small regions do not necessarily outweigh one critical rupture.

Therefore rules such as `hole = severe`, `wet = severe`, or `open_box = level 3` would be unsupported label invention. Operational stakeholders must define consequence-based criteria first.

## Available sample counts versus severity counts

Development metadata contains 2,222 damaged and 1,135 intact images. These represent 480 damaged and 427 intact leakage-control groups. It also includes 174 open-box images from 44 groups.

These counts do **not** translate into severity counts:

| Proposed level | Current defensible count |
|---|---:|
| 0 — intact | Not yet a severity annotation; 1,135 development images are binary intact candidates |
| 1 — cosmetic | Unknown |
| 2 — minor | Unknown |
| 3 — significant | Unknown |
| 4 — severe | Unknown |

All damaged images remain ungraded. Dataset imbalance, ambiguity, and per-level damage-type representation cannot be measured until a manual pilot exists. Augmented images and multiple views must not be counted as independent severity examples.

## Proposed human annotation rubric—not adopted

This rubric is a candidate for stakeholder review and pilot testing. It has not changed any current label.

### 0 — Intact

- no visible damage;
- normally closed parcel;
- no evidence of compromised structure or exposed contents.

### 1 — Cosmetic

- superficial mark or slight dent;
- appearance affected;
- closure and structural integrity appear unaffected;
- contents not exposed.

### 2 — Minor

- clear but localized deformation or small defect;
- parcel appears usable;
- contents not exposed;
- handling integrity appears retained.

### 3 — Significant

- substantial deformation, puncture, wetness, or opening;
- structural or handling integrity is questionable;
- manual inspection or repacking is warranted.

### 4 — Severe

- major rupture or large opening;
- contents exposed;
- severe crushing or extensive wetness;
- parcel no longer appears safely usable.

Annotators should record evidence separately: structural deformation, puncture/rupture, opening extent, visible interior or contents, wetness extent, damaged surface extent, structural-integrity loss, likely handling risk, and uncertainty. `Cannot determine` and `multiple views needed` must be valid outcomes rather than forcing guesses.

## Annotation feasibility and sample-size plan

Manual annotation is necessary. A scientifically defensible sequence is:

1. Have warehouse/quality stakeholders refine the operational meanings of accept, review, repack, and reject.
2. Write an annotation handbook containing boundary examples and rules for occlusion, multiple defects, contents exposure, and uncertain images.
3. Select a group-controlled pilot of roughly **200–300 independent parcel groups**, stratified across damage types, open state, intact hard negatives, viewpoints, and apparent extent. This is a planning range, not a statistical guarantee.
4. Use at least two independent blinded annotators per pilot sample.
5. Measure ordinal agreement using weighted Cohen's kappa for two annotators, plus the confusion matrix between levels. If more annotators are used, select an appropriate multi-rater ordinal statistic.
6. Adjudicate disagreements and revise unclear rubric boundaries before scaling.
7. Use pilot prevalence and learning curves to set the main collection size. As a practical planning floor, seek about **200 independent groups per retained severity level** if feasible; otherwise merge poorly distinguishable adjacent levels or collect more data rather than train on tiny classes.
8. Preserve multiple views under one group ID, freeze train/validation/test groups, and create a new versioned severity dataset. The present final test set must not become its development set.

The existing 907 development groups across binary labels may be insufficient for five balanced ordinal levels after grouping, ambiguity removal, and rare-level stratification. New targeted collection is likely necessary, especially for severe, cosmetic, open-box, and multi-damage boundary cases.

## Future modeling strategy—only after annotation validation

If the new labels demonstrate adequate agreement and sample support:

1. **Ordinal classification is the preferred first approach.** It explicitly respects `intact < cosmetic < minor < significant < severe` and can penalize large grading errors more meaningfully than unrelated classes.
2. **Multi-class classification is a useful baseline.** It predicts one of five classes but treats them more independently, so the ordered relationship is not built into the formulation.
3. **Rule-assisted severity may be evaluated separately.** Human-approved rules could combine predicted type, localized extent, open state, contents exposure, and operational policy. Area or confidence must never be the sole rule.

Model evaluation should include per-level recall/precision, weighted agreement with ground truth, ordinal error distance, severe-case recall, and confusion between adjacent levels. Any model selection must use the new validation split only.

## Scientific conclusion and next action

The current project **cannot scientifically support severity-model training**. Damage type, box geometry, region count, localization, binary condition, and open state are useful contextual signals, but all are proxies or nominal labels. Reliable severity cannot be derived without human judgment and a documented policy.

The next action is annotation design—not modeling: stakeholder review of the proposed rubric, a double-annotated group-controlled pilot, agreement analysis, adjudication, targeted data collection, and a newly frozen dataset version. Stop before training until those requirements are satisfied.

## Artifacts

- `scripts/audit_severity_support.py`
- `models/severity/phase14_annotation_support.json`
- `models/severity/phase14_candidate_rubric.json`
- `models/severity/phase14_box_area_summary.csv`
- `reports/figures/phase14_box_area_distributions.png`

## Closure

No severity model was created or trained. No classifier, detector, checkpoint, threshold, dataset label, annotation, or final test result was modified.
