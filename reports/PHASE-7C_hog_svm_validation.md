# Phase 7C — HOG + SVM Validation

## Selection protocol

Eight linear SVM configurations compared `C ∈ {0.001, 0.01, 0.1, 1.0}` with `class_weight ∈ {None, balanced}`. `StandardScaler(with_mean=False)` and `LinearSVC` were placed in a pipeline; scaling was fitted on the 2,771 training examples only. Selection maximized damaged-class recall, then F1, precision, and ROC-AUC. No test metrics were accessed.

Class weighting was evaluated because the retained training distribution is 58.6% damaged / 41.4% intact, but it did not improve the selected result.

## Frozen configuration

- Kernel/model: linear (`LinearSVC`)
- `C`: **0.001**
- `class_weight`: **None**
- `random_state`: 42
- Training time: **2.28 s**

## Validation metrics

| Metric | Result |
|---|---:|
| Accuracy | 74.41% |
| Damaged precision | 79.17% |
| Damaged recall | 76.44% |
| Damaged F1 | 77.78% |
| ROC-AUC | 79.73% |

Confusion matrix, rows=true and columns=predicted, ordered `[intact, damaged]`:

| | Predicted intact | Predicted damaged |
|---|---:|---:|
| True intact | 176 | 70 |
| True damaged | 82 | 266 |

The selected model is saved at `models/hog_svm_phase7.joblib`. Candidate metrics and validation predictions are under `models/phase7_artifacts/`. The 82 validation false negatives demonstrate that the default decision threshold is not yet strong enough for a high-recall operational gate.
