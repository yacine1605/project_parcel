"""Small, readable helpers shared by the Phase 18 Streamlit dashboard.

This module does not contain model or SQL logic. It validates uploaded images,
calls the existing Phase 16 pipeline, and formats Phase 17 query results for the
user interface.
"""

from __future__ import annotations

import io
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from prototype.database import (
    DEFAULT_DATABASE_PATH,
    get_basic_analytics,
    get_daily_counts,
    get_inspection_by_id,
    get_recent_inspections,
    get_review_inspections,
)
from prototype.inspection_pipeline import inspect_parcel_image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_IMAGE_ROOT = PROJECT_ROOT / "data" / "dashboard_images"
SUPPORTED_UPLOAD_SUFFIXES = {".jpg", ".jpeg", ".png"}


def decode_uploaded_image(file_bytes: bytes, filename: str) -> Image.Image:
    """Validate a browser upload and return a detached RGB image."""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_UPLOAD_SUFFIXES:
        raise ValueError("Please upload a JPG, JPEG, or PNG image.")
    if not file_bytes:
        raise ValueError("The uploaded image is empty.")
    try:
        with Image.open(io.BytesIO(file_bytes)) as opened:
            # Streamlit uploads live in memory. copy() detaches the image from
            # that temporary byte stream, and RGB matches both frozen models.
            return opened.convert("RGB").copy()
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError("The uploaded file is not a readable image.") from error


def run_uploaded_inspection(
    file_bytes: bytes,
    filename: str,
    package_id: str,
    loaded_models: dict[str, Any],
    output_root: Path = DASHBOARD_IMAGE_ROOT,
) -> dict[str, Any]:
    """Save one unique upload and run the unchanged combined pipeline."""
    image = decode_uploaded_image(file_bytes, filename)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    unique_name = f"{timestamp}_{uuid.uuid4().hex}.png"
    original_path = Path(output_root) / "original" / unique_name
    annotated_path = Path(output_root) / "annotated" / unique_name
    # Run inference before writing files so failed requests do not leave orphan
    # uploads behind. Unique names prevent successful uploads from colliding.
    evidence, annotated_image = inspect_parcel_image(image, loaded_models)
    original_path.parent.mkdir(parents=True, exist_ok=True)
    annotated_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(original_path, format="PNG")
    annotated_image.save(annotated_path, format="PNG")
    return {
        "evidence": evidence,
        "package_id": package_id.strip(),
        "original_path": str(original_path),
        "annotated_path": str(annotated_path),
    }


def resolve_stored_image_path(stored_path: str | None) -> Path | None:
    """Resolve an absolute or project-relative database path if it exists."""
    if not stored_path:
        return None
    candidate = Path(stored_path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate if candidate.is_file() else None


def get_review_queue(database_path: Path = DEFAULT_DATABASE_PATH) -> list[dict[str, Any]]:
    """Add readable detection-class summaries to existing REVIEW rows."""
    queue: list[dict[str, Any]] = []
    for row in get_review_inspections(database_path=database_path):
        detail = get_inspection_by_id(row["id"], database_path=database_path) or {}
        classes = [item["class_name"] for item in detail.get("detections", [])]
        queue.append(
            {
                "ID": row["id"],
                "Timestamp (UTC)": row["timestamp_utc"],
                "Package ID": row["package_id"],
                "YOLO classes": ", ".join(classes) if classes else "None",
                "Damaged probability": (
                    row["classifier_probability"] if row.get("classifier_executed", 1) else None
                ),
            }
        )
    return queue


def get_dashboard_snapshot(database_path: Path = DEFAULT_DATABASE_PATH) -> dict[str, Any]:
    """Reuse Phase 17 queries to gather dashboard analytics."""
    return {
        "analytics": get_basic_analytics(database_path=database_path),
        "daily_counts": get_daily_counts(database_path=database_path),
    }


def get_history_rows(limit: int = 100, database_path: Path = DEFAULT_DATABASE_PATH) -> list[dict[str, Any]]:
    """Return only operationally useful columns for the history table."""
    rows = get_recent_inspections(limit=limit, database_path=database_path)
    return [
        {
            "ID": row["id"],
            "Timestamp (UTC)": row["timestamp_utc"],
            "Package ID": row["package_id"],
            "Classifier": row["classifier_label"] if row.get("classifier_executed", 1) else "not run",
            "Damaged probability": row["classifier_probability"] if row.get("classifier_executed", 1) else None,
            "YOLO detections": row["yolo_detection_count"],
            "Decision": row["prototype_decision"],
            "Total latency (ms)": row["total_latency_ms"],
        }
        for row in rows
    ]
