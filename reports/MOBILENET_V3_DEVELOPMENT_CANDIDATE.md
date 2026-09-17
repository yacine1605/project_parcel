# MobileNet V3 corrected-label development candidate

`transfer-mobile-v3-dev-1` is frozen as a development candidate trained on
`parcel_binary_v3`. It is not a production or final-test model.

| Metric | Original V2 candidate | V3 development candidate |
|---|---:|---:|
| Validation recall | 90.31% | 90.11% |
| Validation specificity | 86.00% | 93.70% |
| Validation precision | 92.67% | 95.51% |
| Validation F1 | 91.47% | 92.73% |
| ROC-AUC | 94.89% | 97.31% |
| PR-AUC | 97.60% | 98.32% |

The V3 candidate preserves approximately 90% recall while improving validation
specificity by 7.70 percentage points and precision by 2.84 points. This is not a
controlled final comparison because V3 corrects labels after prior V2 analysis.

The frozen threshold is `0.6363311409950256`. No preserved V3 test image was
loaded during training, selection, or calibration. A new independently collected
and reviewed external test set is required before replacing `transfer-mobile-v1`
in the prototype or making final performance claims.
