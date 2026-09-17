# Proposed Damage Ontology and Annotation Policy

Status: **APPROVED CLASS POLICY — four-class ontology retained**

## Finding

The 232 images containing `compressed` are not a clean balancing pool. Source
filenames include 43 references to `tear`, four to `hole` or `puncture`, and one
to `broken`. These names are review hints rather than ground truth, but the
contact sheets visually confirm that multiple damage concepts were merged into
`compressed`. Bounding boxes also alternate between the local defect and most
of the parcel.

## Class definitions

| Class | Include | Exclude |
|---|---|---|
| `compressed` | Visible structural deformation: crushed corner, collapsed wall, buckling, or material displaced inward/outward | A surface opening without meaningful deformation |
| `hole` | Puncture or opening through the parcel surface, including a local tear that creates an opening | Crease or dent with intact material |
| `minor_damage` | Cosmetic/local damage that is neither structural deformation nor an opening | Generic fallback for uncertain severe damage |
| `wet` | Visible liquid staining, saturation, or water damage | Shadows, dark printing, and plastic-wrap reflections |

No dedicated `tear` class will be added. A tear that creates a visible opening
or puncture is `hole`. A superficial tear without an opening and without
structural deformation is `minor_damage`. If a torn region also shows clear
crushing or collapse, choose the dominant visible defect consistently, or use
multiple boxes only when the defects are spatially distinct.

## Bounding-box policy

Annotate the smallest rectangular region containing the visible damage and
enough surrounding material to identify it. Do not annotate the complete parcel
unless the complete visible parcel is deformed. When a puncture and surrounding
crushing are independently visible, annotate both classes; otherwise choose the
dominant visible defect consistently.

## Review decisions

Use one of:

- `keep_compressed`
- `relabel_hole`
- `relabel_minor_damage`
- `relabel_wet`
- `exclude_unverifiable`
- `needs_domain_review`

Filename flags in the checklist only prioritize inspection. No decision may be
made from a filename without visual confirmation.

## Versioning gate

Do not edit `parcel_damage_v2`. After approval and completion of the checklist:

1. Copy V2 to `parcel_damage_v3`.
2. Apply reviewed label and box changes with a machine-readable change log.
3. Keep the existing four-class `data.yaml` unchanged.
4. Run the complete syntax, distribution, visual, and leakage audit.
5. Freeze V3, then run EXP-002 with one controlled balancing variable.
