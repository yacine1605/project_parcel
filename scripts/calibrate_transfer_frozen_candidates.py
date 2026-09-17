"""Phase 10B/10C: compare and calibrate frozen transfer-learning models.

# ============================================================
# What are we doing in this script?
# ============================================================
#
# Experiment A produced three frozen-backbone models and saved one damaged
# probability for every validation image. This script uses those saved
# probabilities to choose an operational threshold for each model.
#
# The shared requirement is damaged recall >= 90%. Among thresholds that meet
# it, we maximize specificity, then precision, then F1, then choose the threshold
# closest to the familiar default 0.5.
#
# We are NOT training, fine-tuning, loading images, loading model checkpoints,
# or opening the test split. Model parameters cannot change in this phase.
#
# Threshold example:
#   probability = 0.72
#   threshold 0.50: 0.72 >= 0.50 -> damaged
#   threshold 0.80: 0.72 <  0.80 -> intact
#
# Lower thresholds create more damaged predictions: recall usually rises while
# specificity falls. Higher thresholds usually do the opposite. Calibration is
# therefore an operational decision, not another round of neural-network training.
"""

from __future__ import annotations

import argparse, csv, hashlib, json, platform, sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score


RECALL_TARGET = 0.90
ARCHITECTURES = ("resnet50", "efficientnet_b0", "mobilenet_v3_large")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def calculate_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, object]:
    """Convert probabilities to predictions and calculate operational metrics."""
    predictions = probabilities >= threshold
    tn = int(((labels == 0) & ~predictions).sum())
    fp = int(((labels == 0) & predictions).sum())
    fn = int(((labels == 1) & ~predictions).sum())
    tp = int(((labels == 1) & predictions).sum())
    safe_divide = lambda numerator, denominator: float(numerator / denominator) if denominator else 0.0
    recall = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    precision = safe_divide(tp, tp + fp)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    return {
        "threshold": float(threshold), "accuracy": safe_divide(tp + tn, len(labels)),
        # Recall: out of truly damaged parcels, how many were detected?
        "damaged_recall": recall,
        # Specificity: out of truly intact parcels, how many remained intact?
        "specificity": specificity,
        # Precision: out of parcels predicted damaged, how many were damaged?
        "damaged_precision": precision,
        # F1 balances damaged precision and recall.
        "damaged_f1": f1,
        "balanced_accuracy": (recall + specificity) / 2,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def calculate_subgroups(labels, probabilities, threshold, subgroup_names):
    predictions = probabilities >= threshold
    output = {}
    for subgroup in ("ordinary_damaged", "open_box"):
        mask = (labels == 1) & (subgroup_names == subgroup)
        support = int(mask.sum())
        detected = int(predictions[mask].sum())
        output[subgroup] = {"support": support, "detected": detected,
                            "false_negatives": support - detected,
                            "recall": float(detected / support) if support else None}
    return output


def calibrate(labels, probabilities):
    # A prediction changes only when a threshold crosses an observed probability.
    # Sweeping every unique score is exact and avoids an arbitrary coarse grid.
    thresholds = sorted(set(probabilities.tolist() + [0.5]))
    rows = [calculate_metrics(labels, probabilities, threshold) for threshold in thresholds]
    eligible = [row for row in rows if row["damaged_recall"] >= RECALL_TARGET]
    if not eligible:
        raise RuntimeError("No validation threshold satisfies damaged recall >= 90%")
    selected = max(eligible, key=lambda row: (
        row["specificity"], row["damaged_precision"], row["damaged_f1"],
        -abs(row["threshold"] - 0.5),
    ))
    return rows, selected


def write_sweep(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            output = dict(row); output["confusion_matrix"] = json.dumps(output["confusion_matrix"])
            writer.writerow(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-root", type=Path, default=Path("models/transfer_learning"))
    parser.add_argument("--output", type=Path, default=Path("models/transfer_learning_calibration"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite calibration experiment: {args.output}")
    args.output.mkdir(parents=True)

    prediction_rows = {}
    metadata = {}
    for architecture in ARCHITECTURES:
        directory = args.models_root / f"{architecture}_frozen"
        prediction_rows[architecture] = read_csv(directory / "validation_predictions.csv")
        metadata[architecture] = json.loads((directory / "experiment_metadata.json").read_text(encoding="utf-8"))

    # Confirm that all candidates scored exactly the same 592 validation samples
    # with identical labels and frozen subgroup definitions.
    reference_ids = {row["image_id"] for row in prediction_rows[ARCHITECTURES[0]]}
    if len(reference_ids) != 592:
        raise RuntimeError("Expected 592 unique validation samples")
    by_architecture = {name: {row["image_id"]: row for row in rows} for name, rows in prediction_rows.items()}
    if any(set(rows) != reference_ids for rows in by_architecture.values()):
        raise RuntimeError("Candidate validation inventories differ")
    ordered_ids = sorted(reference_ids)
    reference = by_architecture[ARCHITECTURES[0]]
    labels = np.asarray([int(reference[i]["true_label"]) for i in ordered_ids], dtype=np.uint8)
    subgroups = np.asarray([reference[i]["subgroup"] for i in ordered_ids])
    for architecture in ARCHITECTURES[1:]:
        candidate = by_architecture[architecture]
        if not np.array_equal(labels, np.asarray([int(candidate[i]["true_label"]) for i in ordered_ids])):
            raise RuntimeError("Candidate validation labels differ")
        if not np.array_equal(subgroups, np.asarray([candidate[i]["subgroup"] for i in ordered_ids])):
            raise RuntimeError("Candidate subgroup definitions differ")

    results = {}
    comparison_rows = []
    for architecture in ARCHITECTURES:
        candidate = by_architecture[architecture]
        probabilities = np.asarray([float(candidate[i]["damaged_probability"]) for i in ordered_ids])
        sweep, calibrated = calibrate(labels, probabilities)
        default = calculate_metrics(labels, probabilities, 0.5)

        # ROC-AUC measures damaged-vs-intact ranking across thresholds. PR-AUC
        # focuses on the positive/damaged class. Neither changes when only the
        # operating threshold changes.
        ranking = {"roc_auc": float(roc_auc_score(labels, probabilities)),
                   "pr_auc": float(average_precision_score(labels, probabilities))}
        default_subgroups = calculate_subgroups(labels, probabilities, 0.5, subgroups)
        calibrated_subgroups = calculate_subgroups(labels, probabilities, calibrated["threshold"], subgroups)
        deployment = {"checkpoint_size_mib": metadata[architecture]["checkpoint_size_mib"],
                      "latency_ms_per_image": metadata[architecture]["validation_inference_ms_per_image"],
                      "total_parameters": metadata[architecture]["total_parameters"],
                      "trainable_parameters": metadata[architecture]["trainable_parameters"]}
        results[architecture] = {"default": default, "calibrated": calibrated,
                                 "ranking_metrics": ranking, "default_subgroups": default_subgroups,
                                 "calibrated_subgroups": calibrated_subgroups, "deployment": deployment}
        write_sweep(args.output / f"{architecture}_threshold_calibration.csv", sweep)

        for point, point_metrics, point_subgroups in (("default", default, default_subgroups),
                                                       ("calibrated", calibrated, calibrated_subgroups)):
            comparison_rows.append({"architecture": architecture, "operating_point": point,
                **{key: value for key, value in point_metrics.items() if key != "confusion_matrix"},
                **ranking, **deployment,
                "ordinary_damaged_recall": point_subgroups["ordinary_damaged"]["recall"],
                "open_box_recall": point_subgroups["open_box"]["recall"],
                "confusion_matrix": json.dumps(point_metrics["confusion_matrix"])})

        thresholds = [row["threshold"] for row in sweep]
        fig, axis = plt.subplots(figsize=(7, 4))
        axis.plot(thresholds, [row["damaged_recall"] for row in sweep], label="Damaged recall")
        axis.plot(thresholds, [row["specificity"] for row in sweep], label="Specificity")
        axis.axhline(RECALL_TARGET, color="green", linestyle=":", label="Recall target 90%")
        axis.axvline(0.5, color="gray", linestyle="--", label="Default 0.5")
        axis.axvline(calibrated["threshold"], color="red", linestyle="--", label=f"Selected {calibrated['threshold']:.4f}")
        axis.set(title=f"{architecture}: validation threshold calibration", xlabel="Damaged probability threshold", ylabel="Rate")
        axis.grid(alpha=.25); axis.legend(); fig.tight_layout(); fig.savefig(args.output / f"{architecture}_threshold_plot.png", dpi=160); plt.close(fig)

        fig, axis = plt.subplots(figsize=(6, 5))
        axis.plot([row["specificity"] for row in sweep], [row["damaged_recall"] for row in sweep])
        axis.scatter([calibrated["specificity"]], [calibrated["damaged_recall"]], color="red", label="Selected")
        axis.axhline(RECALL_TARGET, color="green", linestyle=":", label="Recall target 90%")
        axis.set(title=f"{architecture}: recall-specificity trade-off", xlabel="Specificity", ylabel="Damaged recall")
        axis.grid(alpha=.25); axis.legend(); fig.tight_layout(); fig.savefig(args.output / f"{architecture}_recall_specificity.png", dpi=160); plt.close(fig)

    with (args.output / "calibrated_candidate_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0]))
        writer.writeheader(); writer.writerows(comparison_rows)

    # Phase 10B/10C identifies the validation leader. The next roadmap phase 10D
    # will formally freeze a winner, so this artifact is a comparison, not a test
    # authorization. Candidate rule: recall eligibility, then specificity,
    # precision, F1, ROC-AUC, PR-AUC. Only an exact tie reaches size/latency.
    leader = max(ARCHITECTURES, key=lambda architecture: (
        results[architecture]["calibrated"]["damaged_recall"] >= RECALL_TARGET,
        results[architecture]["calibrated"]["specificity"],
        results[architecture]["calibrated"]["damaged_precision"],
        results[architecture]["calibrated"]["damaged_f1"],
        results[architecture]["ranking_metrics"]["roc_auc"],
        results[architecture]["ranking_metrics"]["pr_auc"],
        -results[architecture]["deployment"]["checkpoint_size_mib"],
        -results[architecture]["deployment"]["latency_ms_per_image"],
    ))
    artifact = {"phase": "10B/10C", "dataset": "parcel_binary_v2", "calibration_data": "validation only",
                "test_accessed": False, "recall_target": RECALL_TARGET,
                "threshold_rule": "recall >= 90%; maximize specificity; then precision; then F1; then threshold closest to 0.5",
                "candidate_comparison_rule": "recall eligible; specificity; precision; F1; ROC-AUC; PR-AUC; exact tie: smaller/faster",
                "results": results, "validation_leader": leader,
                "leader_not_yet_phase10d_frozen": True,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(), "version": "phase10bc-v1"}
    (args.output / "validation_calibration_results.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    reproducibility = {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
        "scikit_learn": sklearn.__version__, "validation_samples": len(labels), "images_loaded": False,
        "models_loaded": False, "models_trained": False, "test_accessed": False,
        "input_artifacts": {architecture: {
            "predictions_sha256": sha256(args.models_root / f"{architecture}_frozen" / "validation_predictions.csv"),
            "checkpoint_sha256": sha256(args.models_root / f"{architecture}_frozen" / "best_model.pt")}
            for architecture in ARCHITECTURES}}
    (args.output / "reproducibility_metadata.json").write_text(json.dumps(reproducibility, indent=2), encoding="utf-8")
    print(json.dumps({"validation_leader": leader,
                      "thresholds": {name: results[name]["calibrated"]["threshold"] for name in ARCHITECTURES},
                      "calibrated_metrics": {name: results[name]["calibrated"] for name in ARCHITECTURES},
                      "test_accessed": False}, indent=2))


if __name__ == "__main__":
    main()
