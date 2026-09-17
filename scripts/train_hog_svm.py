"""Select one recall-first linear SVM using train/validation only."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def metrics(y: np.ndarray, pred: np.ndarray, score: np.ndarray) -> dict[str, object]:
    precision, recall, f1, _ = precision_recall_fscore_support(y, pred, average="binary", pos_label=1, zero_division=0)
    return {"accuracy": accuracy_score(y, pred), "precision": precision, "recall": recall, "f1": f1, "roc_auc": roc_auc_score(y, score), "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path("datasets/processed/parcel_binary_v1/features"))
    parser.add_argument("--model", type=Path, default=Path("models/hog_svm_phase7.joblib"))
    parser.add_argument("--results", type=Path, default=Path("models/phase7_artifacts/validation_results.json"))
    args = parser.parse_args()
    train = np.load(args.features / "hog_train.npz")
    valid = np.load(args.features / "hog_valid.npz")
    X_train, y_train = train["X"], train["y"]
    X_valid, y_valid = valid["X"], valid["y"]
    candidates = []
    for class_weight in (None, "balanced"):
        for C in (0.001, 0.01, 0.1, 1.0):
            pipe = Pipeline([("scale", StandardScaler(with_mean=False)), ("svm", LinearSVC(C=C, class_weight=class_weight, dual="auto", max_iter=20000, random_state=42))])
            started = time.perf_counter()
            pipe.fit(X_train, y_train)
            elapsed = time.perf_counter() - started
            pred = pipe.predict(X_valid)
            score = pipe.decision_function(X_valid)
            result = {"kernel": "linear", "C": C, "class_weight": class_weight, "training_seconds": elapsed, **metrics(y_valid, pred, score)}
            candidates.append((result, pipe, pred, score))
            print(json.dumps(result))
    # Recall-first, then F1, precision, ROC-AUC, and lower C as deterministic ties.
    selected = max(candidates, key=lambda item: (item[0]["recall"], item[0]["f1"], item[0]["precision"], item[0]["roc_auc"], -item[0]["C"]))
    result, model, pred, score = selected
    bundle = {"pipeline": model, "selected_config": {k: result[k] for k in ("kernel", "C", "class_weight")}, "validation_metrics": result, "positive_class": "damaged", "label_mapping": {"intact": 0, "damaged": 1}, "frozen_after_validation": True}
    args.model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.model)
    payload = {"selection_policy": "maximize damaged recall, then F1, precision, ROC-AUC; no test metrics used", "selected": result, "candidates": [c[0] for c in candidates]}
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (args.results.parent / "phase7_validation_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["image_id", "true_label", "predicted_label", "decision_score"])
        writer.writerows(zip(valid["image_ids"], y_valid, pred, score))
    print("SELECTED", json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
