# Phase 15 — Segmentation Feasibility and Annotation Audit

## Decision

| Question | Answer |
|---|---|
| Do genuine segmentation annotations exist? | **YES, PARTIALLY**—in a preserved pre-conversion polygon archive |
| Can a general four-class segmentation model be trained scientifically today? | **NO** |
| Is manual or assisted annotation required? | **YES** |
| Was a segmentation model trained? | **NO** |

The project has useful historical polygon annotations, but coverage is almost entirely `wet`. They can seed a new annotation effort; they do not support a scientifically balanced four-class segmentation experiment yet.

## Classification, detection, and segmentation

### Classification

Classification answers:

> Is this parcel damaged?

It produces an image-level result such as `damaged`. It does not say where the evidence is located.

### Object detection

Detection answers:

> Where approximately is the damage, and what type might it be?

YOLO currently returns rectangles described by center coordinates, width, and height, plus a class and confidence. A bounding box is useful for localization, but it usually contains both damaged and unaffected pixels.

### Segmentation

Segmentation answers:

> Which pixels belong to the damaged region?

A mask or polygon follows the visible boundary more closely. This can describe irregular shape and area more precisely than an enclosing rectangle.

**Semantic segmentation** assigns a class to each pixel, such as `wet` or `background`. Separate wet regions of the same class may merge conceptually. **Instance segmentation** also distinguishes individual regions, allowing two separate holes to have two separate masks.

## Segmentation is not severity

Segmentation may estimate visible damage area, shape, and coverage. It does not determine physical seriousness:

```text
damage area != severity
```

A small puncture can expose contents, while a large superficial stain or deformation may have a different operational consequence. Phase 14's severity conclusion remains unchanged. No severity label was inferred from polygons or area.

## Annotation inventory

### Active V3 detector labels

Every active `parcel_damage_v3` development annotation is a five-value YOLO detection row:

```text
class_id center_x center_y width height
```

| Split | Label files | Box rows | Polygon rows |
|---|---:|---:|---:|
| Train | 2,081 | 3,177 | 0 |
| Validation | 444 | 681 | 0 |

The current training pipeline therefore uses bounding boxes, not segmentation masks.

### Preserved polygon archive

`parcel_damage_v3/labels_before_polygon_conversion` contains original YOLO polygon rows that were preserved before conversion to detection boxes. A polygon row stores a class followed by at least three normalized `(x, y)` vertex pairs.

The audit validated coordinate range, vertex count, and nonzero polygon area on development splits only:

| Split | Polygon rows | Files containing polygons | Damage-type coverage |
|---|---:|---:|---|
| Train | 109 | 96 | 107 wet, 2 hole |
| Validation | 29 | 24 | 29 wet |
| **Development total** | **138** | **120** | **136 wet, 2 hole** |

Polygons contain 5–99 vertices, with a median of 33. Sample overlays visually confirm that they trace irregular wet and hole regions instead of enclosing rectangles.

Equivalent archives exist under V1, V2, and V3 because dataset versions preserved history. They are copies of the same annotation source and must not be added together as independent masks.

No raster-mask collection, COCO segmentation export, semantic-mask folder, or other independent segmentation dataset was found. The polygon format is genuine segmentation-style annotation, but a full boundary-quality and provenance review has not yet been completed.

## Why rectangular masks would be invalid

Filling a bounding box produces:

```text
bounding rectangle → rectangular foreground block
```

That teaches a model to reproduce annotation rectangles rather than damage boundaries. The filled region includes unaffected cardboard, labels, shadows, and possibly background. Such outputs could not honestly be presented as true damage segmentation.

The project must never convert the 3,858 development boxes into filled masks and call them segmentation ground truth. The Phase 15 figure illustrates this: the red rectangle contains more area than the existing cyan polygon.

## Per-class feasibility

| Target | Current polygon support | Feasibility decision |
|---|---:|---|
| Wet | 136 development polygons | Best pilot candidate after QA and group deduplication |
| Hole | 2 | Far too little; substantial manual annotation required |
| Compressed | 0 | Manual/assisted boundary annotation required |
| Minor visible damage | 0 | Manual annotation required; boundary policy may be difficult |
| Open box | 0 surface masks | Treat separately from damaged-surface segmentation initially |

Wet regions often have visually traceable extent, so they are the strongest starting class. Hole boundaries can also be meaningful, but two development examples are insufficient. Compressed damage may require a policy defining whether annotators trace only visibly deformed surfaces, crease lines, or the full affected face. “Minor damage” is broad and may have uncertain boundaries.

Open-box state is structural: flap configuration, visible interior, and closure state may matter more than a damaged surface. It should not automatically be added to a damage-mask ontology. A future separate open/closed classifier, flap/interior segmentation task, or structural keypoint task could be evaluated under its own policy.

## Proposed annotation policy—not applied

For every selected image, annotators should:

1. outline only visibly damaged pixels;
2. exclude unaffected cardboard inside the old detection rectangle;
3. separate distinct damaged regions where practical;
4. exclude labels, tape, seams, and shadows unless those materials are themselves damaged or the approved class policy explicitly includes them;
5. label each region with the approved nominal damage type;
6. mark uncertain or occluded boundaries instead of guessing;
7. record source image, physical-parcel/group identity, annotator, review status, and provenance;
8. obtain a second review for ambiguous or high-impact masks.

The policy needs class-specific guidance. For wetness, define visible stain boundaries; for holes, outline the opening/ruptured material under an explicit convention; for crushing, define whether creases or the whole deformed surface are included.

### Recommended storage format

Use **YOLO segmentation polygons** for the first pilot because the project already uses Ultralytics, annotations remain compact and inspectable, and existing archived polygons can be preserved without approximation. Maintain a lossless COCO polygon export for interoperability if practical.

For a U-Net experiment, genuine polygons may be rasterized into binary masks at training resolution. Rasterizing a real human polygon is a normal representation conversion; filling an old detection rectangle is not.

All annotations must be stored in a new immutable dataset version:

```text
parcel_damage_seg_v1
```

Do not insert polygons into frozen `parcel_damage_v3`.

## Pilot size and quality plan

First, manually review all 138 archived development polygons, link augmented siblings and multiple views to stable group IDs, and estimate the number of independent sources.

Two reasonable pilot scopes are:

- **Wet-focused feasibility pilot:** approximately 150–250 group-independent wet-region images. Valid archived polygons may contribute after QA and deduplication, with new annotations filling coverage gaps in lighting, viewpoint, stain size, cardboard color, and boundary ambiguity.
- **Four-class feasibility pilot:** approximately 75–100 independent image groups per class, or 300–400 groups total. This is an annotation-planning range, not proof that the resulting data is sufficient for a final model.

Use train/validation splits by source parcel or near-duplicate group. Double-review at least a representative subset and report polygon agreement, for example intersection-over-union between annotators and boundary disagreements. Increase the dataset based on learning curves, class imbalance, and validation uncertainty rather than treating the pilot range as a guaranteed sample-size calculation.

## Damage-area concept

With genuine masks, a later system could count:

```text
damage_pixels = pixels inside all approved damage masks
parcel_pixels = pixels belonging to the visible parcel

damage_ratio = damage_pixels / parcel_pixels
```

Dividing by total image pixels can be misleading when the box occupies only a small part of a warehouse scene. The more meaningful conceptual ratio is:

```text
visible damaged parcel area / visible parcel area
```

Calculating that denominator requires a parcel mask or another reliable parcel-boundary method, which the current annotations do not provide. No damage-ratio dataset was fabricated in this phase.

## Future model options

### U-Net—recommended educational first baseline

U-Net uses an encoder to learn what visual patterns are present and a decoder to restore spatial detail, producing one class prediction per pixel. Skip connections pass fine boundary information from early layers to the decoder. A small U-Net on a carefully reviewed wet-region pilot is the clearest way to learn pixel-level training, mask losses, and intersection-over-union metrics.

### YOLO segmentation—recommended ecosystem comparison

YOLO segmentation combines object localization, class prediction, and instance masks. It fits the existing Ultralytics workflow and may simplify integration with the completed detector. It should be compared only after a frozen segmentation dataset exists.

### Mask R-CNN—later advanced comparison

Mask R-CNN detects objects and predicts a separate mask for each instance. It is powerful and interpretable but has a more complex training and deployment pipeline, so it is not the best first educational experiment.

No model was implemented or trained in Phase 15.

## Value beyond current YOLO detection

Segmentation could provide:

- more accurate irregular damage boundaries;
- visible damaged-area and shape measurements;
- separation of damage pixels from unaffected pixels inside a box;
- overlays that are more precise for warehouse explanation;
- potential features for a future, independently annotated severity system.

It would not replace the need for damage type, open/closed state, operational policy, or severity ground truth.

## Final answers

1. **Do genuine segmentation labels currently exist?** Yes, partially: 138 development polygons in a preserved archive, almost all wet.
2. **Can segmentation be trained scientifically today?** Not as a general four-class damage model. A wet-only feasibility experiment is plausible only after QA, group deduplication, new dataset creation, and likely additional annotation.
3. **Is manual annotation required?** Yes, especially for minor damage, compressed regions, and holes.
4. **Which types are suitable?** Wet is best supported; holes are visually suitable but under-annotated; compressed and minor damage require clear boundary policies and new masks.
5. **Should open box be included?** Not initially as a damaged-surface class. Treat open/closed state separately unless a specific flap/interior mask task is defined.
6. **What dataset version should be created?** `parcel_damage_seg_v1`.
7. **What is the best educational first model?** A small U-Net, initially on a quality-reviewed target such as wet-region segmentation. Compare YOLO segmentation afterward for ecosystem integration.
8. **What value does segmentation add?** Pixel-accurate shape, coverage, and explanation beyond rectangular YOLO boxes—without turning area into severity.

## Artifacts

- `scripts/audit_segmentation_support.py`
- `models/segmentation/phase15_annotation_support.json`
- `reports/figures/phase15_bbox_vs_polygon_examples.png`

## Closure

No fake rectangular masks, severity labels, segmentation dataset, or trained segmentation model were created. No detector, classifier, checkpoint, threshold, frozen annotation, or final test result was modified.
