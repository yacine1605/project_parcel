"""One-time evaluation of the frozen Phase 7 SVM on the untouched test split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support, roc_auc_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path("datasets/processed/parcel_binary_v1/features/hog_test.npz"))
    parser.add_argument("--model", type=Path, default=Path("models/hog_svm_phase7.joblib"))
    parser.add_argument("--output", type=Path, default=Path("models/phase7_artifacts/test_results.json"))
    args = parser.parse_args()
    bundle = joblib.load(args.model)
    if not bundle.get("frozen_after_validation"):
        raise RuntimeError("Refusing test evaluation: model is not marked frozen after validation")
    data = np.load(args.features)
    X, y, ids = data["X"], data["y"], data["image_ids"]
    model = bundle["pipeline"]
    pred, score = model.predict(X), model.decision_function(X)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(y, pred, average="binary", pos_label=1, zero_division=0)
    result = {
        "accuracy": accuracy_score(y, pred), "precision": precision, "recall": recall, "f1": f1,
        "specificity": tn / (tn + fp), "roc_auc": roc_auc_score(y, score),
        "confusion_matrix": cm.tolist(), "false_negatives": int(fn), "false_positives": int(fp),
        "per_class": classification_report(y, pred, labels=[0, 1], target_names=["intact", "damaged"], output_dict=True, zero_division=0),
        "selected_config": bundle["selected_config"], "test_samples": int(len(y)),
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (args.output.parent / "phase7_test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["image_id", "true_label", "predicted_label", "decision_score", "error_type"])
        for iid, truth, guess, value in zip(ids, y, pred, score):
            error = "false_negative" if truth == 1 and guess == 0 else "false_positive" if truth == 0 and guess == 1 else ""
            writer.writerow([iid, int(truth), int(guess), float(value), error])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
