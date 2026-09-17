"""Build the presentation notebook and its portable Colab asset bundle.

The generated bundle contains the four-stage inference code, four checkpoints,
their contracts, and four development/validation examples. It deliberately excludes
training datasets, test images, experiment runs, databases, and local caches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from textwrap import dedent


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COLAB_DIR = PROJECT_ROOT / "colab"
BUNDLE_ROOT_NAME = "parcel_damage_demo"
BUNDLE_PATH = COLAB_DIR / "parcel_damage_demo_bundle.zip"
NOTEBOOK_PATH = COLAB_DIR / "Parcel_Damage_Detection_Demo.ipynb"
MANIFEST_PATH = COLAB_DIR / "asset_manifest.json"
SAMPLE_DIR = COLAB_DIR / "demo_samples"

EXPECTED_MODEL_HASHES = {
    "runs/PARCEL-DET-001_yolo11n_v3/weights/best.pt": (
        "5ffb658923312b26521fed8035cfa36131a4968d773e5a83042ec91588ae9c0f"
    ),
    "models/parcel_state_classifier/best.pt": (
        "1a89ab91dea29cd0f9db1c92d9131c285724bf53438b6d1c926bd4e93702cf7b"
    ),
    "models/transfer_learning/mobilenet_v3_large_frozen/best_model.pt": (
        "f255600ad26289e7a3cc3b277926ffd91c8384a013a4203808b12d4ea77629d1"
    ),
    "runs/EXP-002_yolo11n_ontology_corrected/weights/best.pt": (
        "aa52a6452f593f70759754b240d3a714c62af09f3cab7372aab40545ddce8e9f"
    ),
}

CORE_FILES = [
    "prototype/__init__.py",
    "prototype/inference_classifier.py",
    "prototype/inference_parcel_detector.py",
    "prototype/inference_parcel_state.py",
    "prototype/inference_yolo.py",
    "prototype/inspection_pipeline.py",
    "models/selected_transfer_model/selected_transfer_model.json",
    "models/parcel_state_classifier/best.pt",
    "models/transfer_learning/mobilenet_v3_large_frozen/best_model.pt",
    "runs/EXP-002_yolo11n_ontology_corrected/weights/best.pt",
    "runs/PARCEL-DET-001_yolo11n_v3/weights/best.pt",
    "requirements-colab.txt",
    "colab/ATTRIBUTION.md",
]

SAMPLES = [
    {
        "source": "datasets/processed/parcel_binary_v2/valid/intact/0207df0dfad84f04.jpg",
        "file": "colab/demo_samples/intact_conveyor.jpg",
        "title": "Domain-shift review case",
        "ground_truth": "intact",
        "source_split": "parcel_binary_v2 validation",
        "presentation_point": "Shows a conservative false-positive REVIEW on an out-of-domain scene",
        "source_dataset": "Damaged Box Detection",
        "license": "CC BY 4.0",
        "attribution_url": "https://universe.roboflow.com/project-33xgh/damaged-box-detection",
        "expected_demo_output": {
            "classifier_label": "intact",
            "classifier_executed": False,
            "minimum_yolo_detections": 0,
            "maximum_yolo_detections": 0,
            "decision": "REVIEW",
            "status": "ok",
            "minimum_parcels": 1,
            "required_parcel_state": "open_box",
        },
    },
    {
        "source": (
            "datasets/processed/parcel_damage_v3/valid/images/"
            "-1-_jpeg_jpg.rf.abf23022b21bc1cc74b20b7dd1521a01.jpg"
        ),
        "file": "colab/demo_samples/hole.jpg",
        "title": "First-stage detector miss",
        "ground_truth": "damaged: hole",
        "source_split": "parcel_damage_v3 validation",
        "presentation_point": "Shows that failed parcel localization is routed to REVIEW",
        "source_dataset": "My First Project",
        "license": "CC BY 4.0",
        "attribution_url": "https://universe.roboflow.com/damaged-package-detection/my-first-project-5g94h",
        "expected_demo_output": {
            "classifier_label": "intact",
            "classifier_executed": False,
            "minimum_yolo_detections": 0,
            "maximum_yolo_detections": 0,
            "decision": "REVIEW",
            "status": "no_parcel_detected",
            "maximum_parcels": 0,
        },
    },
    {
        "source": "datasets/processed/parcel_binary_v2/valid/damaged/0958e32e2f57e516.jpg",
        "file": "colab/demo_samples/classifier_miss_caught_by_yolo.jpg",
        "title": "Closed damaged parcel",
        "ground_truth": "damaged",
        "source_split": "parcel_binary_v2 validation",
        "presentation_point": "Shows closed-parcel classification and localized damage evidence",
        "source_dataset": "Damaged Box Detection",
        "license": "CC BY 4.0",
        "attribution_url": "https://universe.roboflow.com/project-33xgh/damaged-box-detection",
        "expected_demo_output": {
            "classifier_label": "damaged",
            "classifier_executed": True,
            "minimum_yolo_detections": 1,
            "required_yolo_class": "hole",
            "decision": "REVIEW",
            "status": "ok",
            "minimum_parcels": 1,
            "required_final_status": "closed_damaged",
        },
    },
    {
        "source": "datasets/processed/parcel_binary_v2/valid/damaged/a505433274d700c2.jpg",
        "file": "colab/demo_samples/known_open_box_failure.jpg",
        "title": "Open-box gate",
        "ground_truth": "damaged: open box",
        "source_split": "parcel_binary_v2 validation",
        "presentation_point": "Shows state classification stopping unnecessary downstream models",
        "source_dataset": "Damaged Box Detection",
        "license": "CC BY 4.0",
        "attribution_url": "https://universe.roboflow.com/project-33xgh/damaged-box-detection",
        "expected_demo_output": {
            "classifier_label": "intact",
            "classifier_executed": False,
            "minimum_yolo_detections": 0,
            "maximum_yolo_detections": 0,
            "decision": "REVIEW",
            "status": "ok",
            "minimum_parcels": 1,
            "required_parcel_state": "open_box",
        },
    },
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_file(relative_path: str) -> Path:
    path = PROJECT_ROOT / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Required Colab asset is missing: {path}")
    return path


def copy_demo_samples() -> None:
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    for sample in SAMPLES:
        source = require_file(sample["source"])
        destination = PROJECT_ROOT / sample["file"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def build_manifest() -> dict:
    for relative_path, expected_hash in EXPECTED_MODEL_HASHES.items():
        observed_hash = sha256(require_file(relative_path))
        if observed_hash != expected_hash:
            raise RuntimeError(
                f"Frozen model hash mismatch for {relative_path}: "
                f"expected {expected_hash}, observed {observed_hash}"
            )

    manifest = {
        "bundle_version": "1.0.0",
        "purpose": "Colab presentation inference only; no training or test-set access",
        "models": [
            {
                "role": "parcel localizer",
                "name": "YOLO11n",
                "version": "PARCEL-DET-001",
                "file": "runs/PARCEL-DET-001_yolo11n_v3/weights/best.pt",
                "sha256": EXPECTED_MODEL_HASHES[
                    "runs/PARCEL-DET-001_yolo11n_v3/weights/best.pt"
                ],
                "classes": ["Boxes"],
            },
            {
                "role": "parcel-state classifier",
                "name": "MobileNetV3-Small",
                "version": "parcel-state-v1",
                "file": "models/parcel_state_classifier/best.pt",
                "sha256": EXPECTED_MODEL_HASHES[
                    "models/parcel_state_classifier/best.pt"
                ],
                "classes": ["closed_box", "open_box"],
            },
            {
                "role": "image classifier",
                "name": "MobileNetV3-Large",
                "version": "transfer-mobile-v1",
                "file": "models/transfer_learning/mobilenet_v3_large_frozen/best_model.pt",
                "sha256": EXPECTED_MODEL_HASHES[
                    "models/transfer_learning/mobilenet_v3_large_frozen/best_model.pt"
                ],
            },
            {
                "role": "damage detector",
                "name": "YOLO11n",
                "version": "EXP-002",
                "file": "runs/EXP-002_yolo11n_ontology_corrected/weights/best.pt",
                "sha256": EXPECTED_MODEL_HASHES[
                    "runs/EXP-002_yolo11n_ontology_corrected/weights/best.pt"
                ],
                "classes": ["minor_damage", "compressed", "hole", "wet"],
            },
        ],
        "samples": [
            {key: value for key, value in sample.items() if key != "source"}
            for sample in SAMPLES
        ],
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(source).strip().splitlines(keepends=True),
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip().splitlines(keepends=True),
    }


def build_notebook() -> None:
    cells = [
        markdown_cell(
            """
            # AI-Powered Parcel Damage Inspection

            **Presentation demo — four-stage, human-in-the-loop computer vision**

            This notebook runs the project's current parcel-level inference pipeline:

            ```text
            phone image -> parcel detector -> parcel crop -> open/closed classifier
                open -> REVIEW
                closed -> damaged/intact classifier -> optional damage localization
            ```

            It performs inference only: no training, threshold tuning, label changes, or final-test access.
            A GPU is recommended for a smooth presentation, but CPU fallback is supported.
            """
        ),
        markdown_cell(
            """
            ## 1. Runtime check and dependency

            In Colab, choose **Runtime → Change runtime type → T4 GPU** when one is available.
            Colab GPU availability varies, so the code reports the actual device and safely falls back to CPU.
            """
        ),
        code_cell(
            """
            import subprocess
            import sys

            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.129"
            ])
            print("Dependency installed.")
            """
        ),
        code_cell(
            """
            import platform
            import torch
            import torchvision

            print(f"Python:      {platform.python_version()}")
            print(f"PyTorch:     {torch.__version__}")
            print(f"torchvision: {torchvision.__version__}")
            print(f"CUDA ready:  {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"GPU:         {torch.cuda.get_device_name(0)}")
            else:
                print("GPU is unavailable; the demo will run on CPU.")
            """
        ),
        markdown_cell(
            """
            ## 2. Get the slim demo project

            The default mode clones the curated GitHub repository. If the repository is private or has not
            been pushed yet, choose `upload_bundle` and upload `parcel_damage_demo_bundle.zip` when prompted.
            The bundle is about 40 MB and contains only the inference code, frozen weights, and development
            examples—not the 1+ GB training datasets.
            """
        ),
        code_cell(
            """
            #@title Project source
            SOURCE = "github" #@param ["github", "upload_bundle"]
            REPOSITORY_URL = "https://github.com/yacine1605/project_parcel.git" #@param {type:"string"}
            REPOSITORY_BRANCH = "main" #@param {type:"string"}

            import shutil
            import subprocess
            import zipfile
            from pathlib import Path

            WORKSPACE = Path("/content") if Path("/content").is_dir() else Path.cwd()

            def safe_extract(archive_path: Path, destination: Path) -> None:
                destination = destination.resolve()
                with zipfile.ZipFile(archive_path) as archive:
                    for member in archive.infolist():
                        candidate = (destination / member.filename).resolve()
                        if destination != candidate and destination not in candidate.parents:
                            raise ValueError(f"Unsafe archive path: {member.filename}")
                    archive.extractall(destination)

            if SOURCE == "github":
                PROJECT_ROOT = WORKSPACE / "project_parcel"
                if not PROJECT_ROOT.is_dir():
                    subprocess.check_call([
                        "git", "clone", "--depth", "1", "--branch", REPOSITORY_BRANCH,
                        REPOSITORY_URL, str(PROJECT_ROOT),
                    ])
                else:
                    print(f"Reusing existing checkout: {PROJECT_ROOT}")
            else:
                try:
                    from google.colab import files
                except ImportError as error:
                    raise RuntimeError("Bundle upload mode must run in Google Colab.") from error
                uploaded = files.upload()
                zip_names = [name for name in uploaded if name.lower().endswith(".zip")]
                if len(zip_names) != 1:
                    raise ValueError("Upload exactly one parcel_damage_demo_bundle.zip file.")
                uploaded_bundle = WORKSPACE / zip_names[0]
                uploaded_bundle.write_bytes(uploaded[zip_names[0]])
                safe_extract(uploaded_bundle, WORKSPACE)
                PROJECT_ROOT = WORKSPACE / "parcel_damage_demo"

            if not PROJECT_ROOT.is_dir():
                raise FileNotFoundError(f"Project root was not created: {PROJECT_ROOT}")
            print(f"Project root: {PROJECT_ROOT}")
            """
        ),
        markdown_cell(
            """
            ## 3. Verify model artifacts

            SHA-256 checks protect the presentation from accidentally loading a different checkpoint.
            All four checkpoint hashes are verified before any model is loaded.
            """
        ),
        code_cell(
            """
            import hashlib
            import json

            def file_sha256(path: Path) -> str:
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(block)
                return digest.hexdigest()

            manifest_path = PROJECT_ROOT / "colab" / "asset_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for model_record in manifest["models"]:
                model_path = PROJECT_ROOT / model_record["file"]
                if not model_path.is_file():
                    raise FileNotFoundError(f"Missing model: {model_path}")
                observed = file_sha256(model_path)
                if observed != model_record["sha256"]:
                    raise RuntimeError(f"Hash mismatch: {model_record['name']}")
                print(f"PASS  {model_record['role']}: {model_record['name']} ({model_record['version']})")
            """
        ),
        markdown_cell(
            """
            ## 4. Load the four inference components

            The generic detector finds parcels. A crop classifier identifies open boxes. Closed parcels then
            receive damaged/intact classification, and damage YOLO localizes defects only when requested.
            """
        ),
        code_cell(
            """
            import os
            import sys

            os.environ["YOLO_CONFIG_DIR"] = str(PROJECT_ROOT / "Ultralytics")
            if str(PROJECT_ROOT) not in sys.path:
                sys.path.insert(0, str(PROJECT_ROOT))

            from prototype.inspection_pipeline import (
                inspect_parcel_image,
                load_prototype_models,
                open_supported_image,
            )

            loaded_models = load_prototype_models()
            print("All four inference components loaded successfully.")
            """
        ),
        markdown_cell(
            """
            ## 5. Run the prepared presentation examples

            These are labeled **validation/development examples**, not fresh evidence of model quality. They
            demonstrate damage localization, open-box gating, a first-stage detector miss, and a state-classifier
            domain-shift error. Failures are retained because understanding them is part of the project.
            """
        ),
        code_cell(
            """
            import pandas as pd
            import matplotlib.pyplot as plt
            from IPython.display import display

            def inspect_and_display(image_path: Path, title: str | None = None):
                image = open_supported_image(image_path)
                evidence, annotated = inspect_parcel_image(image, loaded_models)

                summary = {
                    "image": title or image_path.name,
                    "status": evidence["status"],
                    "parcels": len(evidence["parcels"]),
                    "parcel_states": ", ".join(p["parcel_state"] for p in evidence["parcels"]) or "none",
                    "classifier": evidence["classifier_label"] if evidence["classifier_executed"] else "not run",
                    "damaged_probability": (
                        evidence["classifier_probability"] if evidence["classifier_executed"] else None
                    ),
                    "yolo_detections": evidence["yolo_detection_count"],
                    "damage_types": ", ".join(evidence["yolo_damage_types"]) or "none",
                    "decision": evidence["prototype_decision"],
                    "total_ms": evidence["timing_ms"]["total_pipeline"],
                }
                return evidence, annotated, summary

            prepared_results = []
            figure, axes = plt.subplots(2, 2, figsize=(13, 12))
            for axis, sample in zip(axes.flat, manifest["samples"]):
                evidence, annotated, summary = inspect_and_display(
                    PROJECT_ROOT / sample["file"], sample["title"]
                )
                prepared_results.append(summary)
                axis.imshow(annotated)
                axis.set_title(
                    f"{sample['title']} — {evidence['prototype_decision']}\\n"
                    f"ground truth: {sample['ground_truth']}"
                )
                axis.axis("off")
            plt.tight_layout()
            plt.show()

            prepared_frame = pd.DataFrame(prepared_results)
            prepared_frame["damaged_probability"] = prepared_frame["damaged_probability"].map(
                lambda value: "not run" if pd.isna(value) else f"{value:.1%}"
            )
            prepared_frame["total_ms"] = prepared_frame["total_ms"].map(lambda value: f"{value:.1f}")
            display(prepared_frame)
            """
        ),
        markdown_cell(
            """
            ### Demo-image attribution

            The prepared examples are derivative validation images from the following Roboflow Universe datasets,
            both provided under **CC BY 4.0**:

            - [Damaged Box Detection](https://universe.roboflow.com/project-33xgh/damaged-box-detection)
            - [My First Project / damaged-package-detection](https://universe.roboflow.com/damaged-package-detection/my-first-project-5g94h)

            See `colab/ATTRIBUTION.md` in the project for details.
            """
        ),
        markdown_cell(
            """
            ## 6. Inspect your own parcel image

            Run the next cell, upload one JPG/PNG/WebP image, and the notebook will show the annotated evidence.
            Images stay in the temporary Colab runtime unless you explicitly download the result.
            """
        ),
        code_cell(
            """
            from google.colab import files

            uploaded_images = files.upload()
            if len(uploaded_images) != 1:
                raise ValueError("Upload exactly one parcel image.")

            uploaded_name, uploaded_bytes = next(iter(uploaded_images.items()))
            uploaded_path = WORKSPACE / Path(uploaded_name).name
            uploaded_path.write_bytes(uploaded_bytes)

            custom_evidence, custom_annotated, custom_summary = inspect_and_display(uploaded_path)
            display(pd.DataFrame([custom_summary]))
            display(custom_annotated)
            """
        ),
        markdown_cell(
            """
            ## 7. Download the evidence (optional)

            The JSON keeps every model output and timing value. The annotated image provides visual evidence
            for an operator. The notebook does not claim a severity score or automatically reject a parcel.
            """
        ),
        code_cell(
            """
            from google.colab import files

            evidence_path = WORKSPACE / "parcel_inspection_evidence.json"
            annotated_path = WORKSPACE / "parcel_inspection_annotated.png"
            evidence_path.write_text(json.dumps(custom_evidence, indent=2), encoding="utf-8")
            custom_annotated.save(annotated_path)

            files.download(str(evidence_path))
            files.download(str(annotated_path))
            """
        ),
        markdown_cell(
            """
            ## Results to present honestly

            - **MobileNetV3-Large final test (592 images):** 89.29% damaged recall, 89.06% precision,
              89.17% F1, 93.35% ROC-AUC, and 78.50% specificity.
            - **YOLO11n EXP-002 validation (444 images / 681 instances):** 0.584 precision, 0.561 recall,
              0.588 mAP50, and 0.296 mAP50–95.
            - **Parcel-state classifier validation (66 crops):** macro-F1 0.8019; this small result is preliminary.
            - Each metric belongs to its component dataset. The combined four-stage policy has not been validated
              as a production system.
            - Open parcels, missing localization, model disagreement, and detected damage go to **REVIEW**. The
              prototype never issues **REJECT** or invents physical severity.
            - The project is a research prototype, not a validated warehouse safety system.

            The strongest presentation story is not “the model is perfect.” It is that the system keeps raw
            evidence, exposes known failure modes, and uses a conservative human-in-the-loop decision.
            """
        ),
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "name": NOTEBOOK_PATH.name,
                "provenance": [],
            },
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def bundle_file_list() -> list[str]:
    return CORE_FILES + ["colab/asset_manifest.json"] + [sample["file"] for sample in SAMPLES]


def validate_notebook() -> None:
    import ast

    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    if notebook.get("nbformat") != 4:
        raise RuntimeError("Generated notebook is not nbformat 4.")
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") == "code":
            try:
                ast.parse("".join(cell.get("source", [])))
            except SyntaxError as error:
                raise RuntimeError(f"Generated notebook code cell {index} is invalid: {error}") from error


def build_bundle() -> None:
    with zipfile.ZipFile(BUNDLE_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for relative_path in bundle_file_list():
            source = require_file(relative_path)
            archive_name = f"{BUNDLE_ROOT_NAME}/{relative_path}"
            info = zipfile.ZipInfo.from_file(source, arcname=archive_name)
            # Fixed timestamp and permissions make repeated builds deterministic.
            info.date_time = (2026, 8, 30, 0, 0, 0)
            info.external_attr = 0o100644 << 16
            with source.open("rb") as handle:
                archive.writestr(info, handle.read(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)


def validate_bundle() -> None:
    expected_names = {f"{BUNDLE_ROOT_NAME}/{path}" for path in bundle_file_list()}
    with zipfile.ZipFile(BUNDLE_PATH) as archive:
        observed_names = set(archive.namelist())
        if observed_names != expected_names:
            raise RuntimeError(
                f"Unexpected bundle contents. Missing={expected_names - observed_names}; "
                f"extra={observed_names - expected_names}"
            )
        bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError(f"Corrupt ZIP member: {bad_member}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-bundle", action="store_true", help="Generate notebook/assets but skip the ZIP")
    args = parser.parse_args()

    copy_demo_samples()
    build_manifest()
    build_notebook()
    validate_notebook()
    if not args.no_bundle:
        build_bundle()
        validate_bundle()

    print(f"Notebook: {NOTEBOOK_PATH}")
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Samples:  {len(SAMPLES)}")
    if not args.no_bundle:
        print(f"Bundle:   {BUNDLE_PATH} ({BUNDLE_PATH.stat().st_size / 1024 / 1024:.2f} MiB)")
        print(f"SHA-256: {sha256(BUNDLE_PATH)}")


if __name__ == "__main__":
    main()
