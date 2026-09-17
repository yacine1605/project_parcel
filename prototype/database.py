"""
# ============================================================
# What is stored in this database?
# ============================================================
#
# SQLite stores structured inspection RECORDS. It does not store model weights,
# training datasets, or full image binary data. Image paths are stored instead,
# which keeps the database small and makes image files easier to manage.
#
# Relationship:
#     one inspections row  --->  zero or many detections rows
#
# The inspections.id primary key is SQLite's unique internal row identifier.
# package_id is different: it represents a warehouse parcel/barcode identifier.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE_PATH = PROJECT_ROOT / "data" / "parcel_inspections.db"
VALID_CLASSIFIER_LABELS = {"damaged", "intact"}
VALID_DECISIONS = {"ACCEPT", "REVIEW"}


def connect_database(database_path: Path = DEFAULT_DATABASE_PATH) -> sqlite3.Connection:
    """Open SQLite and enable foreign-key relationship checks."""
    database_path = Path(database_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)

    # A connection represents an open session with the SQLite file.
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    # SQLite requires foreign-key enforcement to be enabled per connection.
    # This prevents a detection from referencing a nonexistent inspection.
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database(database_path: Path = DEFAULT_DATABASE_PATH) -> None:
    """Create the two tables if they do not already exist."""
    connection = connect_database(database_path)
    try:
        # A cursor executes SQL commands and reads their returned rows.
        cursor = connection.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS inspections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                package_id TEXT NOT NULL,
                timestamp_utc TEXT NOT NULL,
                image_path TEXT NOT NULL,
                annotated_image_path TEXT,
                classifier_label TEXT NOT NULL CHECK (classifier_label IN ('damaged', 'intact')),
                classifier_probability REAL NOT NULL CHECK (classifier_probability BETWEEN 0.0 AND 1.0),
                classifier_threshold REAL NOT NULL CHECK (classifier_threshold BETWEEN 0.0 AND 1.0),
                yolo_detection_count INTEGER NOT NULL CHECK (yolo_detection_count >= 0),
                prototype_decision TEXT NOT NULL CHECK (prototype_decision IN ('ACCEPT', 'REVIEW')),
                classifier_latency_ms REAL NOT NULL CHECK (classifier_latency_ms >= 0.0),
                yolo_latency_ms REAL NOT NULL CHECK (yolo_latency_ms >= 0.0),
                total_latency_ms REAL NOT NULL CHECK (total_latency_ms >= 0.0),
                classifier_model TEXT NOT NULL,
                classifier_version TEXT NOT NULL,
                yolo_model TEXT NOT NULL,
                yolo_version TEXT NOT NULL,
                classifier_executed INTEGER NOT NULL DEFAULT 1
                    CHECK (classifier_executed IN (0, 1)),
                damage_detector_executed INTEGER NOT NULL DEFAULT 1
                    CHECK (damage_detector_executed IN (0, 1)),
                evidence_json TEXT,
                created_at_utc TEXT NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                inspection_id INTEGER NOT NULL,
                class_name TEXT NOT NULL,
                confidence REAL NOT NULL CHECK (confidence BETWEEN 0.0 AND 1.0),
                x1 REAL NOT NULL,
                y1 REAL NOT NULL,
                x2 REAL NOT NULL,
                y2 REAL NOT NULL,
                FOREIGN KEY (inspection_id)
                    REFERENCES inspections(id)
                    ON DELETE CASCADE
            )
            """
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_inspections_timestamp ON inspections(timestamp_utc DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_inspections_decision ON inspections(prototype_decision)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_detections_inspection ON detections(inspection_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_detections_class ON detections(class_name)")

        # These additive migrations keep databases created by earlier versions
        # readable. Old rows lack explicit execution metadata, so DEFAULT 1 is
        # the backward-compatible assumption; new rows store exact flags.
        inspection_columns = {
            row["name"] for row in cursor.execute("PRAGMA table_info(inspections)").fetchall()
        }
        migrations = {
            "classifier_executed": (
                "ALTER TABLE inspections ADD COLUMN classifier_executed "
                "INTEGER NOT NULL DEFAULT 1 CHECK (classifier_executed IN (0, 1))"
            ),
            "damage_detector_executed": (
                "ALTER TABLE inspections ADD COLUMN damage_detector_executed "
                "INTEGER NOT NULL DEFAULT 1 CHECK (damage_detector_executed IN (0, 1))"
            ),
            "evidence_json": "ALTER TABLE inspections ADD COLUMN evidence_json TEXT",
        }
        for column_name, statement in migrations.items():
            if column_name not in inspection_columns:
                cursor.execute(statement)

        # CREATE TABLE and CREATE INDEX change the database schema. commit()
        # permanently saves those changes.
        connection.commit()
    finally:
        # Closing releases the file handle even if a SQL error occurs.
        connection.close()


def generate_prototype_package_id() -> str:
    """Create a temporary prototype ID when no barcode ID is available."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    # This is only a prototype identifier—not a real logistics tracking number.
    return f"PROTO-{timestamp}-{uuid.uuid4().hex[:8].upper()}"


def validate_evidence(evidence: dict[str, Any]) -> None:
    """Reject incomplete or inconsistent records before starting an INSERT."""
    required = {
        "timestamp_utc", "classifier_label", "classifier_probability",
        "classifier_threshold", "yolo_detection_count", "prototype_decision",
        "classifier", "yolo", "timing_ms",
    }
    missing = sorted(required - set(evidence))
    if missing:
        raise ValueError(f"Inspection evidence is missing fields: {missing}")
    if evidence["classifier_label"] not in VALID_CLASSIFIER_LABELS:
        raise ValueError(f"Invalid classifier label: {evidence['classifier_label']}")
    if evidence["prototype_decision"] not in VALID_DECISIONS:
        raise ValueError(f"Invalid prototype decision: {evidence['prototype_decision']}")
    detections = evidence["yolo"].get("detections", [])
    if evidence["yolo_detection_count"] != len(detections):
        raise ValueError("YOLO detection count does not match the detection list.")
    for detection in detections:
        box = detection.get("bounding_box_xyxy", [])
        if len(box) != 4 or box[2] < box[0] or box[3] < box[1]:
            raise ValueError(f"Invalid detection box: {box}")
        confidence = float(detection["confidence"])
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"Invalid detection confidence: {confidence}")


def save_inspection(
    evidence: dict[str, Any],
    image_path: str | Path,
    annotated_image_path: str | Path | None,
    package_id: str | None = None,
    database_path: Path = DEFAULT_DATABASE_PATH,
) -> int:
    """Save one inspection and all of its detections in one transaction."""
    validate_evidence(evidence)
    package_id = package_id.strip() if package_id else generate_prototype_package_id()
    if not package_id:
        raise ValueError("package_id cannot be empty.")

    initialize_database(database_path)
    connection = connect_database(database_path)
    try:
        cursor = connection.cursor()
        # Question-mark placeholders pass values separately from SQL. This is
        # safer than building SQL strings from package IDs or paths.
        cursor.execute(
            """
            INSERT INTO inspections (
                package_id, timestamp_utc, image_path, annotated_image_path,
                classifier_label, classifier_probability, classifier_threshold,
                yolo_detection_count, prototype_decision,
                classifier_latency_ms, yolo_latency_ms, total_latency_ms,
                classifier_model, classifier_version, yolo_model, yolo_version,
                classifier_executed, damage_detector_executed, evidence_json,
                created_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                package_id,
                evidence["timestamp_utc"],
                str(image_path),
                str(annotated_image_path) if annotated_image_path else None,
                evidence["classifier_label"],
                float(evidence["classifier_probability"]),
                float(evidence["classifier_threshold"]),
                int(evidence["yolo_detection_count"]),
                evidence["prototype_decision"],
                float(evidence["timing_ms"]["classifier_forward"]),
                float(evidence["timing_ms"]["yolo_inference"]),
                float(evidence["timing_ms"]["total_pipeline"]),
                "MobileNetV3-Large",
                "transfer-mobile-v1",
                "YOLO11n",
                "EXP-002",
                int(bool(evidence.get("classifier_executed", True))),
                int(bool(evidence.get("damage_detector_executed", True))),
                json.dumps(evidence, separators=(",", ":")),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        inspection_id = int(cursor.lastrowid)

        for detection in evidence["yolo"]["detections"]:
            x1, y1, x2, y2 = detection["bounding_box_xyxy"]
            cursor.execute(
                """
                INSERT INTO detections (
                    inspection_id, class_name, confidence, x1, y1, x2, y2
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    inspection_id,
                    detection["class_name"],
                    float(detection["confidence"]),
                    float(x1), float(y1), float(x2), float(y2),
                ),
            )

        # Both the inspection and detections are committed together. If an error
        # occurs first, rollback() below prevents a partially saved inspection.
        connection.commit()
        return inspection_id
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def rows_to_dicts(rows: list[sqlite3.Row]) -> list[dict[str, Any]]:
    """Convert sqlite3.Row objects into ordinary dictionaries for UI code."""
    return [dict(row) for row in rows]


def get_recent_inspections(limit: int = 10, database_path: Path = DEFAULT_DATABASE_PATH) -> list[dict]:
    """Return newest inspections first."""
    if limit <= 0:
        raise ValueError("limit must be positive.")
    connection = connect_database(database_path)
    try:
        rows = connection.execute(
            "SELECT * FROM inspections ORDER BY timestamp_utc DESC, id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return rows_to_dicts(rows)
    finally:
        connection.close()


def get_inspection_by_id(inspection_id: int, database_path: Path = DEFAULT_DATABASE_PATH) -> dict | None:
    """Return one inspection and its zero-or-many linked detections."""
    connection = connect_database(database_path)
    try:
        inspection = connection.execute("SELECT * FROM inspections WHERE id = ?", (inspection_id,)).fetchone()
        if inspection is None:
            return None
        detections = connection.execute(
            "SELECT * FROM detections WHERE inspection_id = ? ORDER BY id", (inspection_id,)
        ).fetchall()
        result = dict(inspection)
        result["detections"] = rows_to_dicts(detections)
        if result.get("evidence_json"):
            result["evidence"] = json.loads(result["evidence_json"])
        return result
    finally:
        connection.close()


def get_review_inspections(limit: int = 100, database_path: Path = DEFAULT_DATABASE_PATH) -> list[dict]:
    """Return inspections currently routed to human REVIEW."""
    connection = connect_database(database_path)
    try:
        rows = connection.execute(
            "SELECT * FROM inspections WHERE prototype_decision = 'REVIEW' ORDER BY timestamp_utc DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return rows_to_dicts(rows)
    finally:
        connection.close()


def get_inspections_by_damage_class(class_name: str, database_path: Path = DEFAULT_DATABASE_PATH) -> list[dict]:
    """Find unique inspections containing a requested YOLO damage class."""
    connection = connect_database(database_path)
    try:
        rows = connection.execute(
            """
            SELECT DISTINCT i.*
            FROM inspections AS i
            JOIN detections AS d ON d.inspection_id = i.id
            WHERE d.class_name = ?
            ORDER BY i.timestamp_utc DESC
            """,
            (class_name,),
        ).fetchall()
        return rows_to_dicts(rows)
    finally:
        connection.close()


def get_daily_counts(database_path: Path = DEFAULT_DATABASE_PATH) -> list[dict]:
    """Count saved inspections per UTC calendar date."""
    connection = connect_database(database_path)
    try:
        rows = connection.execute(
            """
            SELECT substr(timestamp_utc, 1, 10) AS inspection_date, COUNT(*) AS inspection_count
            FROM inspections
            GROUP BY substr(timestamp_utc, 1, 10)
            ORDER BY inspection_date DESC
            """
        ).fetchall()
        return rows_to_dicts(rows)
    finally:
        connection.close()


def get_basic_analytics(database_path: Path = DEFAULT_DATABASE_PATH) -> dict[str, Any]:
    """Return simple counts/averages that the future dashboard can display."""
    connection = connect_database(database_path)
    try:
        summary = connection.execute(
            """
            SELECT
                COUNT(*) AS total_inspections,
                SUM(CASE WHEN prototype_decision = 'ACCEPT' THEN 1 ELSE 0 END) AS accept_count,
                SUM(CASE WHEN prototype_decision = 'REVIEW' THEN 1 ELSE 0 END) AS review_count,
                AVG(CASE WHEN classifier_executed = 1 THEN classifier_probability END)
                    AS average_classifier_probability,
                AVG(total_latency_ms) AS average_total_latency_ms
            FROM inspections
            """
        ).fetchone()
        damage_rows = connection.execute(
            "SELECT class_name, COUNT(*) AS detection_count FROM detections GROUP BY class_name ORDER BY class_name"
        ).fetchall()
        result = dict(summary)
        result["damage_class_counts"] = {row["class_name"]: row["detection_count"] for row in damage_rows}
        return result
    finally:
        connection.close()
