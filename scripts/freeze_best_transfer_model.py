"""Phase 10D: freeze the selected transfer-learning system before test.

# ============================================================
# What are we doing in this script?
# ============================================================
#
# Phase 10B/10C identified frozen MobileNetV3-Large as the validation leader.
# This script records exactly what the final transfer system is:
# architecture, pretrained weights, checkpoint hash, preprocessing, probability
# threshold, dataset version, validation metrics, and deployment measurements.
#
# A freeze is an experimental promise: after this manifest is created, none of
# those choices may change in response to future test results.
#
# We are NOT training, fine-tuning, recalibrating, loading images, computing new
# predictions, or opening the test split. We only verify existing metadata and
# create a reproducible description of the already-selected system.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_ARCHITECTURE = "mobilenet_v3_large"
EXPECTED_DATASET = "parcel_binary_v2"
EXPECTED_RECALL_TARGET = 0.90


def sha256(path: Path) -> str:
    """Return a file fingerprint so later audits can detect any modification."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, default=Path("models/transfer_learning/mobilenet_v3_large_frozen"))
    parser.add_argument("--calibration", type=Path, default=Path("models/transfer_learning_calibration/validation_calibration_results.json"))
    parser.add_argument("--strict-freeze-audit", type=Path, default=Path("models/transfer_learning/strict_freeze_audit.json"))
    parser.add_argument("--output", type=Path, default=Path("models/selected_transfer_model"))
    args = parser.parse_args()

    # Refusing overwrite protects the original pre-test freeze decision.
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen selection: {args.output}")

    checkpoint_path = args.model_root / "best_model.pt"
    model_metadata_path = args.model_root / "experiment_metadata.json"
    predictions_path = args.model_root / "validation_predictions.csv"
    for path in (checkpoint_path, model_metadata_path, predictions_path, args.calibration, args.strict_freeze_audit):
        if not path.is_file():
            raise FileNotFoundError(path)

    model_metadata = json.loads(model_metadata_path.read_text(encoding="utf-8"))
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    freeze_audit = json.loads(args.strict_freeze_audit.read_text(encoding="utf-8"))

    # Fail closed if any scientific assumption differs from the completed phases.
    checks = {
        "architecture_is_selected_mobile_net": model_metadata["architecture"] == EXPECTED_ARCHITECTURE,
        "dataset_is_parcel_binary_v2": model_metadata["dataset"] == EXPECTED_DATASET,
        "training_used_train_validation_only": model_metadata["splits_used"] == ["train", "valid"],
        "test_was_closed_during_training": model_metadata["test_accessed"] is False,
        "backbone_was_frozen": model_metadata["backbone_frozen"] is True,
        "strict_backbone_state_audit_passed": freeze_audit[EXPECTED_ARCHITECTURE]["strictly_frozen"] is True,
        "calibration_used_validation_only": calibration["calibration_data"] == "validation only",
        "test_was_closed_during_calibration": calibration["test_accessed"] is False,
        "recall_target_is_predefined_90_percent": calibration["recall_target"] == EXPECTED_RECALL_TARGET,
        "validation_leader_is_mobile_net": calibration["validation_leader"] == EXPECTED_ARCHITECTURE,
        "leader_was_not_previously_frozen": calibration["leader_not_yet_phase10d_frozen"] is True,
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Phase 10D freeze checks failed: {failed}")

    selected = calibration["results"][EXPECTED_ARCHITECTURE]
    selected_metrics = selected["calibrated"]
    selected_threshold = selected_metrics["threshold"]
    if selected_metrics["damaged_recall"] < EXPECTED_RECALL_TARGET:
        raise RuntimeError("Selected threshold does not satisfy the recall target")

    args.output.mkdir(parents=True)

    # The checkpoint already exists in the Experiment A directory. Referencing it
    # by path and SHA-256 avoids making a second copy that could later diverge.
    freeze_manifest = {
        "phase": "10D",
        "status": "frozen_before_test",
        "dataset_version": EXPECTED_DATASET,
        "architecture": EXPECTED_ARCHITECTURE,
        "architecture_display_name": "MobileNetV3-Large",
        "torchvision_constructor": "torchvision.models.mobilenet_v3_large",
        "pretrained_weights": model_metadata["pretrained_weights"],
        "training_strategy": "strictly frozen ImageNet backbone; only classifier.3 linear head trained",
        "checkpoint_path": checkpoint_path.as_posix(),
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_size_mib": model_metadata["checkpoint_size_mib"],
        "total_parameters": model_metadata["total_parameters"],
        "trainable_parameters_during_experiment_a": model_metadata["trainable_parameters"],
        "input": {
            "color": "RGB",
            "resize": "resize shorter side to 256 while preserving aspect ratio",
            "crop": "center crop 224x224",
            "tensor_shape": "[batch_size, 3, 224, 224]",
            "normalization_mean": [0.485, 0.456, 0.406],
            "normalization_std": [0.229, 0.224, 0.225],
            "random_augmentation_at_inference": False,
        },
        "model_output": "one raw damaged logit per image; shape [batch_size]",
        "probability_conversion": "torch.sigmoid(logit)",
        "prediction_rule": "damaged if probability >= frozen_threshold, otherwise intact",
        "frozen_threshold": selected_threshold,
        "threshold_calibration_dataset": "validation only",
        "damaged_recall_target": EXPECTED_RECALL_TARGET,
        "threshold_selection_rule": calibration["threshold_rule"],
        "validation_metrics_at_frozen_threshold": selected_metrics,
        "validation_ranking_metrics": selected["ranking_metrics"],
        "validation_subgroup_metrics": selected["calibrated_subgroups"],
        "validation_inference_ms_per_image_rtx5060": model_metadata["validation_inference_ms_per_image"],
        "source_artifacts": {
            "model_metadata_path": model_metadata_path.as_posix(),
            "model_metadata_sha256": sha256(model_metadata_path),
            "validation_predictions_path": predictions_path.as_posix(),
            "validation_predictions_sha256": sha256(predictions_path),
            "calibration_path": args.calibration.as_posix(),
            "calibration_sha256": sha256(args.calibration),
            "strict_freeze_audit_path": args.strict_freeze_audit.as_posix(),
            "strict_freeze_audit_sha256": sha256(args.strict_freeze_audit),
        },
        "integrity_checks": checks,
        "test_accessed": False,
        "model_retrained_in_phase10d": False,
        "threshold_recalibrated_in_phase10d": False,
        "must_not_change_after_test": True,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_version": "transfer-mobile-v1",
    }
    manifest_path = args.output / "selected_transfer_model.json"
    manifest_path.write_text(json.dumps(freeze_manifest, indent=2), encoding="utf-8")

    reproducibility = {
        "python": sys.version,
        "platform": platform.platform(),
        "files_created": [manifest_path.as_posix()],
        "files_modified": [],
        "models_loaded": False,
        "images_loaded": False,
        "predictions_computed": False,
        "test_accessed": False,
    }
    (args.output / "reproducibility_metadata.json").write_text(json.dumps(reproducibility, indent=2), encoding="utf-8")

    print("Phase 10D freeze complete.")
    print("Selected model: MobileNetV3-Large with strictly frozen ImageNet backbone")
    print(f"Frozen threshold: {selected_threshold:.12f}")
    print(f"Validation damaged recall: {selected_metrics['damaged_recall']:.2%}")
    print(f"Validation specificity: {selected_metrics['specificity']:.2%}")
    print(f"Checkpoint SHA-256: {freeze_manifest['checkpoint_sha256']}")
    print("The test split was not accessed.")
    print("The next roadmap phase may evaluate this exact frozen system once.")


if __name__ == "__main__":
    main()
