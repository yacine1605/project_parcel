"""Smoke-test the exact portable Colab bundle in an isolated temporary folder."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATH = PROJECT_ROOT / "colab" / "parcel_damage_demo_bundle.zip"
RESULT_PREFIX = "COLAB_SMOKE_RESULTS="


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_bundle(destination: Path) -> Path:
    if not BUNDLE_PATH.is_file():
        raise FileNotFoundError(f"Build the Colab bundle first: {BUNDLE_PATH}")
    with zipfile.ZipFile(BUNDLE_PATH) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError(f"Corrupt ZIP member: {bad_member}")
        archive.extractall(destination)
    bundle_root = destination / "parcel_damage_demo"
    if not bundle_root.is_dir():
        raise RuntimeError(f"Bundle root is missing: {bundle_root}")
    return bundle_root


def verify_manifest(bundle_root: Path) -> dict:
    manifest_path = bundle_root / "colab" / "asset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for record in manifest["models"]:
        model_path = bundle_root / record["file"]
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        observed = sha256(model_path)
        if observed != record["sha256"]:
            raise RuntimeError(f"Hash mismatch for {record['name']}: {observed}")
    return manifest


def run_isolated_inference(bundle_root: Path) -> list[dict]:
    child_code = """
import json
from pathlib import Path
from prototype.inspection_pipeline import inspect_parcel_image, load_prototype_models, open_supported_image

root = Path.cwd()
manifest = json.loads((root / "colab" / "asset_manifest.json").read_text(encoding="utf-8"))
models = load_prototype_models()
rows = []
for sample in manifest["samples"]:
    evidence, _ = inspect_parcel_image(open_supported_image(root / sample["file"]), models)
    rows.append({
        "file": sample["file"],
        "status": evidence["status"],
        "classifier_label": evidence["classifier_label"],
        "classifier_executed": evidence["classifier_executed"],
        "classifier_probability": evidence["classifier_probability"],
        "yolo_detection_count": evidence["yolo_detection_count"],
        "yolo_damage_types": evidence["yolo_damage_types"],
        "parcel_states": [parcel["parcel_state"] for parcel in evidence["parcels"]],
        "final_statuses": [parcel["final_status"] for parcel in evidence["parcels"]],
        "parcel_count": len(evidence["parcels"]),
        "decision": evidence["prototype_decision"],
    })
print("COLAB_SMOKE_RESULTS=" + json.dumps(rows))
"""
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = str(bundle_root)
    environment["YOLO_CONFIG_DIR"] = str(bundle_root / "Ultralytics")
    completed = subprocess.run(
        [sys.executable, "-c", child_code],
        cwd=bundle_root,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Isolated inference failed.\n"
            f"STDOUT:\n{completed.stdout}\n"
            f"STDERR:\n{completed.stderr}"
        )
    result_lines = [line for line in completed.stdout.splitlines() if line.startswith(RESULT_PREFIX)]
    if len(result_lines) != 1:
        raise RuntimeError(f"Could not find smoke-test results in output:\n{completed.stdout}")
    return json.loads(result_lines[0][len(RESULT_PREFIX):])


def check_expected_outputs(manifest: dict, observed_rows: list[dict]) -> None:
    observed_by_file = {row["file"]: row for row in observed_rows}
    for sample in manifest["samples"]:
        expected = sample["expected_demo_output"]
        observed = observed_by_file.get(sample["file"])
        if observed is None:
            raise AssertionError(f"No result for {sample['file']}")
        if observed["classifier_label"] != expected["classifier_label"]:
            raise AssertionError(f"Classifier mismatch for {sample['file']}: {observed}")
        if observed["classifier_executed"] != expected.get("classifier_executed", True):
            raise AssertionError(f"Classifier execution mismatch for {sample['file']}: {observed}")
        if observed["decision"] != expected["decision"]:
            raise AssertionError(f"Decision mismatch for {sample['file']}: {observed}")
        if observed["status"] != expected.get("status", "ok"):
            raise AssertionError(f"Pipeline status mismatch for {sample['file']}: {observed}")
        if observed["parcel_count"] < expected.get("minimum_parcels", 0):
            raise AssertionError(f"Too few parcels for {sample['file']}: {observed}")
        if observed["parcel_count"] > expected.get("maximum_parcels", float("inf")):
            raise AssertionError(f"Too many parcels for {sample['file']}: {observed}")
        count = observed["yolo_detection_count"]
        if count < expected.get("minimum_yolo_detections", 0):
            raise AssertionError(f"Too few YOLO detections for {sample['file']}: {observed}")
        if "maximum_yolo_detections" in expected and count > expected["maximum_yolo_detections"]:
            raise AssertionError(f"Too many YOLO detections for {sample['file']}: {observed}")
        required_class = expected.get("required_yolo_class")
        if required_class and required_class not in observed["yolo_damage_types"]:
            raise AssertionError(f"Missing YOLO class {required_class} for {sample['file']}: {observed}")
        required_state = expected.get("required_parcel_state")
        if required_state and required_state not in observed["parcel_states"]:
            raise AssertionError(f"Missing parcel state {required_state} for {sample['file']}: {observed}")
        required_final = expected.get("required_final_status")
        if required_final and required_final not in observed["final_statuses"]:
            raise AssertionError(f"Missing final status {required_final} for {sample['file']}: {observed}")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="parcel_colab_smoke_") as temporary:
        bundle_root = extract_bundle(Path(temporary))
        manifest = verify_manifest(bundle_root)
        observed_rows = run_isolated_inference(bundle_root)
        check_expected_outputs(manifest, observed_rows)

    print("PASS: portable bundle integrity, isolated imports, and four inference cases")
    for row in observed_rows:
        classifier = row["classifier_label"] if row["classifier_executed"] else "not-run"
        probability = row["classifier_probability"]
        types = ",".join(row["yolo_damage_types"]) or "none"
        print(
            f"  {Path(row['file']).name}: classifier={classifier} "
            f"p={'n/a' if not row['classifier_executed'] else f'{probability:.3f}'}, "
            f"yolo={row['yolo_detection_count']} [{types}], "
            f"decision={row['decision']}"
        )


if __name__ == "__main__":
    main()
