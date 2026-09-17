# V3 label-correction diagnostic — frozen MobileNet

## Boundary

The unchanged `transfer-mobile-v1` checkpoint and unchanged frozen threshold were
run on the 592 preserved validation images using corrected V3 labels. This is a
post-hoc diagnostic of label effects, not model selection, threshold calibration,
or a fresh final evaluation.

## Comparison

| Metric | Original V2 validation labels | Corrected V3 validation labels | Change |
|---|---:|---:|---:|
| Damaged recall | 90.31% | 94.35% | +4.04 pp |
| Specificity | 86.00% | 79.83% | -6.17 pp |
| Precision | 92.67% | 87.43% | -5.24 pp |
| Accuracy | 88.85% | 88.51% | -0.34 pp |
| F1 | 91.47% | 90.76% | -0.71 pp |
| ROC-AUC | 94.89% | 96.50% | +1.61 pp |
| PR-AUC | 97.60% | 97.89% | +0.29 pp |

Corrected confusion matrix, rows intact/damaged and columns intact/damaged:

```text
[[190, 48],
 [ 20,334]]
```

## Interpretation

The original open-box label noise made genuinely intact-looking boxes count as
damaged positives. Correcting those labels removes difficult false negatives from
the damaged class, increasing recall, but turns many model damage predictions on
those images into intact-class false positives, reducing specificity and
precision. The correction therefore changes the apparent operating trade-off; it
does not simply improve every metric.

No threshold should be selected from this post-hoc run. A new V3 model may use
train and validation for development, but a final claim requires newly collected
external test data because the prior test analysis motivated the correction.

Artifacts are under `models/v3_label_correction_diagnostic/`.
