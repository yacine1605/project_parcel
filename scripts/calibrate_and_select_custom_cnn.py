"""Phase 9C: validation-only threshold calibration and scratch-CNN selection.

No images, checkpoints, or test data are loaded. The script uses only the saved
Phase 9A/9B validation probabilities produced by the frozen best checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score


RECALL_TARGET = 0.90


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def calculate_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, object]:
    predictions = probabilities >= threshold
    tn = int(((labels == 0) & ~predictions).sum())
    fp = int(((labels == 0) & predictions).sum())
    fn = int(((labels == 1) & ~predictions).sum())
    tp = int(((labels == 1) & predictions).sum())
    divide = lambda numerator, denominator: float(numerator / denominator) if denominator else 0.0
    recall = divide(tp, tp + fn)
    specificity = divide(tn, tn + fp)
    precision = divide(tp, tp + fp)
    f1 = divide(2 * precision * recall, precision + recall)
    return {
        "threshold": float(threshold), "accuracy": divide(tp + tn, len(labels)),
        "damaged_precision": precision, "damaged_recall": recall, "damaged_f1": f1,
        "specificity": specificity, "balanced_accuracy": (recall + specificity) / 2,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def subgroup_metrics(labels, probabilities, threshold, subgroups):
    predictions = probabilities >= threshold
    result = {}
    for name in ("ordinary_damaged", "open_box"):
        mask = (labels == 1) & (subgroups == name)
        support = int(mask.sum())
        detected = int(predictions[mask].sum())
        result[name] = {"support": support, "detected": detected,
                        "false_negatives": support - detected,
                        "recall": float(detected / support) if support else None}
    return result


def calibrate(labels, probabilities):
    # A prediction changes only when the threshold crosses an observed score.
    # Including 0.5 also guarantees exact reproduction of the default operating point.
    thresholds = sorted(set(probabilities.tolist() + [0.5]))
    sweep = [calculate_metrics(labels, probabilities, threshold) for threshold in thresholds]
    eligible = [row for row in sweep if row["damaged_recall"] >= RECALL_TARGET]
    if not eligible:
        raise RuntimeError("No validation threshold satisfies the predefined recall target")
    selected = max(eligible, key=lambda row: (
        row["specificity"], row["damaged_precision"], row["damaged_f1"],
        -abs(row["threshold"] - 0.5),
    ))
    return sweep, selected


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [key for key in rows[0] if key != "confusion_matrix"] + ["confusion_matrix"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            output["confusion_matrix"] = json.dumps(output["confusion_matrix"])
            writer.writerow(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase9a", type=Path, default=Path("models/phase9a_custom_cnn/validation_predictions.csv"))
    parser.add_argument("--phase9b", type=Path, default=Path("models/phase9b_custom_cnn/validation_predictions.csv"))
    parser.add_argument("--phase9a-checkpoint", type=Path, default=Path("models/phase9a_custom_cnn/best_model.pt"))
    parser.add_argument("--phase9b-checkpoint", type=Path, default=Path("models/phase9b_custom_cnn/best_model.pt"))
    parser.add_argument("--output", type=Path, default=Path("models/phase9c_artifacts"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen selection artifacts: {args.output}")
    args.output.mkdir(parents=True)

    rows_a = read_csv(args.phase9a)
    rows_b = read_csv(args.phase9b)
    by_id_a = {row["image_id"]: row for row in rows_a}
    by_id_b = {row["image_id"]: row for row in rows_b}
    if len(rows_a) != 592 or len(rows_b) != 592 or set(by_id_a) != set(by_id_b):
        raise RuntimeError("Phase 9A and 9B validation prediction inventories differ")

    # Phase 9B saved the frozen subgroup for every validation sample. Reuse it
    # for both candidates so subgroup definitions cannot change by candidate.
    ordered_ids = sorted(by_id_a)
    labels = np.asarray([int(by_id_a[i]["true_label"]) for i in ordered_ids], dtype=np.uint8)
    labels_b = np.asarray([int(by_id_b[i]["true_label"]) for i in ordered_ids], dtype=np.uint8)
    if not np.array_equal(labels, labels_b):
        raise RuntimeError("Validation labels differ between candidate artifacts")
    subgroups = np.asarray([by_id_b[i]["subgroup"] for i in ordered_ids])
    probabilities = {
        "phase9a": np.asarray([float(by_id_a[i]["damaged_probability"]) for i in ordered_ids]),
        "phase9b": np.asarray([float(by_id_b[i]["damaged_probability"]) for i in ordered_ids]),
    }

    candidates = {}
    comparison_rows = []
    for name in ("phase9a", "phase9b"):
        scores = probabilities[name]
        sweep, selected = calibrate(labels, scores)
        default = calculate_metrics(labels, scores, 0.5)
        ranking = {"roc_auc": float(roc_auc_score(labels, scores)),
                   "pr_auc": float(average_precision_score(labels, scores))}
        default_subgroups = subgroup_metrics(labels, scores, 0.5, subgroups)
        selected_subgroups = subgroup_metrics(labels, scores, selected["threshold"], subgroups)
        candidates[name] = {"default": default, "calibrated": selected,
                            "ranking_metrics": ranking,
                            "default_subgroups": default_subgroups,
                            "calibrated_subgroups": selected_subgroups}
        write_rows(args.output / f"{name}_threshold_calibration.csv", sweep)
        for operating_point, metrics, subgroup in (("default", default, default_subgroups),
                                                    ("calibrated", selected, selected_subgroups)):
            comparison_rows.append({"candidate": name, "operating_point": operating_point,
                **{key: value for key, value in metrics.items() if key != "confusion_matrix"},
                **ranking,
                "confusion_matrix": json.dumps(metrics["confusion_matrix"]),
                "ordinary_damaged_recall": subgroup["ordinary_damaged"]["recall"],
                "open_box_recall": subgroup["open_box"]["recall"]})

        # Two complementary plots: metrics across threshold and the direct
        # recall/specificity trade-off independent of threshold spacing.
        threshold_values = [row["threshold"] for row in sweep]
        fig, axis = plt.subplots(figsize=(7, 4))
        axis.plot(threshold_values, [row["damaged_recall"] for row in sweep], label="Damaged recall")
        axis.plot(threshold_values, [row["specificity"] for row in sweep], label="Specificity")
        axis.axhline(RECALL_TARGET, color="green", linestyle=":", label="Recall target 90%")
        axis.axvline(0.5, color="gray", linestyle="--", label="Default 0.5")
        axis.axvline(selected["threshold"], color="red", linestyle="--", label=f"Selected {selected['threshold']:.4f}")
        axis.set(title=f"{name.upper()} validation threshold calibration", xlabel="Probability threshold", ylabel="Rate")
        axis.grid(alpha=.25); axis.legend(); fig.tight_layout(); fig.savefig(args.output / f"{name}_threshold_metrics.png", dpi=160); plt.close(fig)
        fig, axis = plt.subplots(figsize=(6, 5))
        axis.plot([row["specificity"] for row in sweep], [row["damaged_recall"] for row in sweep])
        axis.scatter([selected["specificity"]], [selected["damaged_recall"]], color="red", label="Selected")
        axis.axhline(RECALL_TARGET, color="green", linestyle=":", label="Recall target 90%")
        axis.set(title=f"{name.upper()} recall-specificity trade-off", xlabel="Specificity", ylabel="Damaged recall")
        axis.grid(alpha=.25); axis.legend(); fig.tight_layout(); fig.savefig(args.output / f"{name}_recall_specificity.png", dpi=160); plt.close(fig)

    with (args.output / "scratch_cnn_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0]))
        writer.writeheader(); writer.writerows(comparison_rows)

    # Both calibrated candidates satisfy recall. The predeclared winner rule is
    # specificity, precision, F1, ROC-AUC, then simpler no-augmentation pipeline.
    winner_name = max(("phase9a", "phase9b"), key=lambda name: (
        candidates[name]["calibrated"]["damaged_recall"] >= RECALL_TARGET,
        candidates[name]["calibrated"]["specificity"],
        candidates[name]["calibrated"]["damaged_precision"],
        candidates[name]["calibrated"]["damaged_f1"],
        candidates[name]["ranking_metrics"]["roc_auc"],
        name == "phase9a",
    ))
    winner = candidates[winner_name]
    checkpoint = args.phase9a_checkpoint if winner_name == "phase9a" else args.phase9b_checkpoint
    artifact = {
        "phase": "9C", "dataset_version": "parcel_binary_v2", "test_accessed": False,
        "recall_target": RECALL_TARGET,
        "threshold_rule": "recall >= 90%; maximize specificity, precision, F1; threshold closest to 0.5",
        "model_selection_rule": "eligible recall; maximize specificity, precision, F1, ROC-AUC; then no augmentation",
        "candidates": candidates, "selected_candidate": winner_name,
        "selected_checkpoint": checkpoint.as_posix(), "selected_checkpoint_sha256": sha256(checkpoint),
        "selected_architecture": "ParcelDamageCNN (93,825 parameters)",
        "selected_preprocessing": "Resize 224x224 RGB, ToTensor, ImageNet normalization",
        "selected_augmentation_policy": "none" if winner_name == "phase9a" else "Phase 9B mild training-only augmentation",
        "selected_threshold": winner["calibrated"]["threshold"],
        "selected_validation_metrics": winner["calibrated"],
        "selected_ranking_metrics": winner["ranking_metrics"],
        "selected_subgroup_metrics": winner["calibrated_subgroups"],
        "frozen_before_test": True, "timestamp_utc": datetime.now(timezone.utc).isoformat(), "version": "phase9c-v1",
    }
    (args.output / "selected_scratch_cnn.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    reproducibility = {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
        "scikit_learn": sklearn.__version__, "validation_samples": len(labels),
        "phase9a_predictions_sha256": sha256(args.phase9a), "phase9b_predictions_sha256": sha256(args.phase9b),
        "images_loaded": False, "checkpoints_loaded": False, "test_accessed": False, "models_retrained": False}
    (args.output / "reproducibility_metadata.json").write_text(json.dumps(reproducibility, indent=2), encoding="utf-8")
    print(json.dumps({"selected_candidate": winner_name, "selected_threshold": artifact["selected_threshold"],
                      "selected_metrics": artifact["selected_validation_metrics"],
                      "phase9a_threshold": candidates["phase9a"]["calibrated"]["threshold"],
                      "phase9b_threshold": candidates["phase9b"]["calibrated"]["threshold"],
                      "test_accessed": False}, indent=2))


if __name__ == "__main__":
    main()
