# Phase 12 — Final Classification Error Analysis

## Purpose and scientific boundary

Metrics tell us **how often** the model is wrong. Error analysis helps us understand **why** the model is wrong.

This phase is descriptive only. It reads the 592 probabilities and predictions permanently recorded in Phase 11. It does not load the model, run inference, change preprocessing, move the frozen threshold, modify labels, or retrain anything. The source prediction file has SHA-256 `00318a9e9ed48b6980fa4a0662be3a1abf7e3a3fa05d085dff26d9942e7e0222`.

Test-error inspection may be used to document limitations and motivate a **future experiment with a new evaluation protocol**. Changing this system and evaluating it again on the same test set would contaminate the final result.

## Prediction margin

For every saved prediction, the analysis calculates:

```text
prediction margin = damaged probability - frozen threshold
```

The threshold remains `0.434584259987`. A probability of `0.42` has a margin near `-0.0146`, making it a borderline false negative. A probability of `0.08` has a margin near `-0.3546`, indicating a much more confident failure. Positive margins are classified as damaged; negative margins are classified as intact.

## Frozen error counts

The analysis reproduces the permanently recorded confusion matrix without generating new predictions:

| Outcome | Count |
|---|---:|
| True positives | 350 |
| True negatives | 157 |
| False positives | 43 |
| False negatives | 42 |

The four derived tables contain the requested sample ID, path, group ID, labels, probability, threshold, margin, subgroup, and open-box flag. The consolidated table also includes outcome and visually reviewed failure tags.

## False-negative analysis

### Breakdown

| False-negative subgroup | Misses | Fraction of all FN | Subgroup support | Subgroup recall |
|---|---:|---:|---:|---:|
| Open box | 35 | 83.33% | 49 | 28.57% |
| Ordinary damaged | 7 | 16.67% | 343 | 97.96% |
| **All damaged** | **42** | **100%** | **392** | **89.29%** |

Open boxes explain why the classifier missed the predefined 90% test-recall target. Detecting only one additional damaged sample would have raised recall from `350/392 = 89.29%` to `351/392 = 89.54%`, which would still be below 90%; two additional detections would have produced `352/392 = 89.80%`; three would have produced `353/392 = 90.05%`. Thirty-five of the 42 misses came from the open-box subgroup.

### Open-box misses

The visual review shows several recurring forms:

- clean cardboard with no tear, hole, crushing, or discoloration;
- a narrow dark opening that resembles a seam or shadow;
- flaps separated only slightly or partly hidden by perspective;
- hands, tape dispensers, or packing activity partially obscuring the parcel;
- multiple open boxes in cluttered stacks, reducing the prominence of any one opening;
- oblique or partial views where the visible box still appears structurally normal.

“Open” is partly a semantic and structural condition rather than a classic surface-damage texture. A box can be open without torn fibers, crushed geometry, holes, or other cues that a CNN can associate with conventional visible damage. The missed open boxes have a median damaged probability of only `0.144`; 22 of 35 are at least `0.20` below the threshold, and only 3 are within `0.10`. This is therefore not mainly a small threshold-boundary problem. The learned representation frequently treats these images as convincingly intact.

Correctly detected open boxes more often show a large visible interior, widely separated or raised flaps, strong opening geometry, or a prominent dark gap. Even so, two of the 14 correct open-box detections are very close to the threshold, emphasizing substantial within-subgroup variation.

### Ordinary visible-damage misses

Only 7 of 343 ordinary damaged parcels were missed. They show combinations of subtle or localized deformation, cluttered stacks, strong branding, partial views, and atypical packaging. Three cases remain partly uncertain because the thumbnail evidence does not support a single confident cause.

These ordinary misses are much more borderline than the open-box misses: their median damaged probability is `0.390`, and 4 of 7 lie within `0.10` of the frozen threshold. Ordinary visible-damage recognition is consequently strong; the 97.96% subgroup recall supports that conclusion.

## False-positive analysis

The 43 false positives contain intact features that resemble evidence of damage learned by the model. Recurrent observations include:

- shipping tape, broad tape strips, and dense labels;
- normal lid seams, folds, corners, and flap boundaries;
- branding, printing, and unusual colored packaging;
- strong perspective edges or non-frontal orientations;
- stacked boxes and warehouse/background clutter;
- shadows or contrast boundaries that resemble openings or deformation.

The false-positive median probability is `0.611`. Twelve of 43 are within `0.10` above the threshold, while 18 are at least `0.20` above it. Thus, some intact parcels are borderline referrals, but a meaningful subset are confident false alarms. Lowering or raising the threshold is not part of this phase and would not address the underlying representation consistently.

## Qualitative failure taxonomy

Tags are descriptive observations rather than proven causal explanations. Multiple tags are allowed, so counts overlap and must not be summed as mutually exclusive categories.

| Visual tag | Tagged error images |
|---|---:|
| Perspective or orientation | 44 |
| Open-box semantic ambiguity | 35 |
| Packaging or branding | 30 |
| Background or clutter | 28 |
| Seam or corner confusion | 20 |
| Tape or label confusion | 19 |
| Partial parcel visibility | 13 |
| Shadow or lighting | 9 |
| Subtle or localized damage | 7 |
| Other or uncertain | 3 |

The taxonomy is stored by sample ID in `error_taxonomy.json`. Group-related crops receive consistent interpretations so near-duplicate views are not presented as independent discoveries.

## Confidence and error distributions

| Outcome | Median probability | Middle 50% probability range |
|---|---:|---:|
| True positive | 0.993 | 0.959–0.999 |
| True negative | 0.115 | 0.057–0.210 |
| False positive | 0.611 | 0.500–0.700 |
| False negative | 0.174 | 0.079–0.302 |

Across all 85 errors:

- 4 were within `±0.02` of the threshold;
- 13 were within `±0.05`;
- 19 were within `±0.10`;
- 42 were within `±0.20`;
- 43 were at least `0.20` from the threshold.

Only 22.35% (`19/85`) are within `±0.10`, while 50.59% (`43/85`) are at least `0.20` away. Errors are therefore **not mostly borderline**. Open-box false negatives are especially confident. The high ROC-AUC and PR-AUC remain compatible with this result: overall ranking is strong, but a coherent label subgroup has visual semantics that the representation handles poorly.

## Final interpretation

1. **Why was the 90% recall target missed?** Open boxes dominated the missed damaged parcels. Three more detected damaged samples would have reached 90%, while 35 open-box samples were missed.
2. **What fraction of false negatives were open boxes?** `35/42 = 83.33%`.
3. **Is ordinary visible-damage recognition strong?** Yes. Ordinary-damaged recall was 97.96%, with only 7 misses among 343 samples.
4. **What creates most false positives?** Normal perspective edges, tape and labels, seams and corners, branding, and clutter repeatedly resemble learned damage cues.
5. **Are mistakes mostly borderline or confident?** Not mostly borderline. Only 19 of 85 errors fall within `±0.10`; 43 are at least `0.20` from the threshold.
6. **What limitation is architectural?** A global image classifier must compress the full scene into one label and lacks an explicit representation of flap geometry, parcel state, and localized structural relationships. Backgrounds and normal high-contrast structures can influence that global decision.
7. **What limitation comes from the label definition?** Mapping `open_box` into the same positive class as visually damaged parcels combines two related but visually different concepts: surface/shape damage and an open/closed operational state.
8. **What future data would most likely help?** A future protocol should collect diverse, group-controlled open and closed box pairs across opening widths, flap angles, camera viewpoints, lighting, box colors, tape states, and packing scenes. It should also include intact hard negatives containing heavy tape, labels, branding, strong seams, stacked boxes, and shadows. New collections need new held-out evaluation groups; the current test set must not become a tuning set.

Potential future research could compare a dedicated open/closed-state head, multi-task learning, or localized detection of flaps/openings. These are proposals only—not changes to `transfer-mobile-v1`.

## Artifacts

- `scripts/analyze_final_classifier_errors.py`
- `models/selected_transfer_model/error_analysis.csv`
- `models/selected_transfer_model/false_negatives.csv`
- `models/selected_transfer_model/false_positives.csv`
- `models/selected_transfer_model/true_positives.csv`
- `models/selected_transfer_model/true_negatives.csv`
- `models/selected_transfer_model/error_taxonomy.json`
- `reports/figures/phase12_false_negatives.png`
- `reports/figures/phase12_false_positives.png`
- `reports/figures/phase12_open_box_errors.png`
- `reports/figures/phase12_open_box_correct.png`
- `reports/figures/phase12_probability_distributions.png`

## Closure

Phase 12 did not load or modify model weights, change the threshold, regenerate predictions, alter labels, or remove samples. Phase 11 remains the permanent final evaluation. All improvement ideas are reserved for future experiments with a new evaluation protocol.
