"""Test the Phase 17 SQLite functions without loading or changing any model."""

from __future__ import annotations

import copy
import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# When this file is executed directly, Python initially searches `scripts/`.
# Add the project root so the reusable `prototype` package can be imported.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prototype.database import (
    connect_database,
    get_basic_analytics,
    get_daily_counts,
    get_inspection_by_id,
    get_inspections_by_damage_class,
    get_recent_inspections,
    get_review_inspections,
    initialize_database,
    save_inspection,
)


EXAMPLE_EVIDENCE = {
    "timestamp_utc": "2026-09-17T12:00:00+00:00",
    "classifier_label": "damaged",
    "classifier_probability": 0.96,
    "classifier_threshold": 0.434584259987,
    "classifier_executed": True,
    "damage_detector_executed": True,
    "yolo_detection_count": 1,
    "prototype_decision": "REVIEW",
    "classifier": {"label": "damaged", "damaged_probability": 0.96},
    "yolo": {
        "detections": [{
            "class_name": "hole",
            "confidence": 0.81,
            "bounding_box_xyxy": [10.0, 20.0, 50.0, 70.0],
        }]
    },
    "parcels": [{
        "parcel_index": 0,
        "parcel_state": "closed_box",
        "final_status": "closed_damaged",
    }],
    "timing_ms": {
        "classifier_forward": 2.0,
        "yolo_inference": 3.0,
        "total_pipeline": 8.0,
    },
}


def main() -> None:
    evidence = copy.deepcopy(EXAMPLE_EVIDENCE)

    # TemporaryDirectory creates a disposable test location. The real database
    # is not affected by these structural tests.
    with tempfile.TemporaryDirectory(prefix="parcel_db_test_") as temporary_directory:
        database_path = Path(temporary_directory) / "test_inspections.db"
        initialize_database(database_path)

        # Save several records using real Phase 16 evidence. We change only the
        # operational package IDs, never model outputs or probabilities.
        saved_ids = []
        for index in range(3):
            saved_ids.append(save_inspection(
                evidence=copy.deepcopy(evidence),
                image_path="development_example.jpg",
                annotated_image_path="development_example_annotated.jpg",
                package_id=f"DB-TEST-{index + 1:03d}",
                database_path=database_path,
            ))

        assert saved_ids == [1, 2, 3]
        first = get_inspection_by_id(1, database_path)
        assert first is not None
        assert first["package_id"] == "DB-TEST-001"
        assert first["classifier_executed"] == 1
        assert first["damage_detector_executed"] == 1
        assert first["evidence"]["parcels"][0]["final_status"] == "closed_damaged"
        assert len(first["detections"]) == evidence["yolo_detection_count"] == 1
        assert first["detections"][0]["class_name"] == "hole"
        assert len(get_recent_inspections(10, database_path)) == 3
        assert len(get_review_inspections(database_path=database_path)) == 3
        assert len(get_inspections_by_damage_class("hole", database_path)) == 3
        assert get_daily_counts(database_path)[0]["inspection_count"] == 3

        analytics = get_basic_analytics(database_path)
        assert analytics["total_inspections"] == 3
        assert analytics["review_count"] == 3
        assert analytics["accept_count"] == 0
        assert analytics["damage_class_counts"] == {"hole": 3}

        # A state-gated inspection must remain REVIEW without pretending that
        # either downstream damage model produced an intact result.
        skipped = copy.deepcopy(evidence)
        skipped.update({
            "classifier_label": "intact",
            "classifier_probability": 0.0,
            "classifier_executed": False,
            "damage_detector_executed": False,
            "yolo_detection_count": 0,
            "parcels": [{
                "parcel_index": 0,
                "parcel_state": "open_box",
                "final_status": "open_box",
            }],
        })
        skipped["yolo"] = {"detections": []}
        skipped_id = save_inspection(
            evidence=skipped,
            image_path="open_parcel.jpg",
            annotated_image_path="open_parcel_annotated.jpg",
            package_id="DB-TEST-OPEN",
            database_path=database_path,
        )
        skipped_row = get_inspection_by_id(skipped_id, database_path)
        assert skipped_row["classifier_executed"] == 0
        assert skipped_row["damage_detector_executed"] == 0
        assert skipped_row["evidence"]["parcels"][0]["parcel_state"] == "open_box"
        analytics = get_basic_analytics(database_path)
        assert analytics["total_inspections"] == 4
        assert analytics["average_classifier_probability"] == 0.96

        # Prove the foreign key works: inspection 999999 does not exist, so the
        # database must reject a detection that tries to reference it.
        connection = connect_database(database_path)
        try:
            try:
                connection.execute(
                    """
                    INSERT INTO detections
                        (inspection_id, class_name, confidence, x1, y1, x2, y2)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (999999, "hole", 0.5, 0.0, 0.0, 10.0, 10.0),
                )
                connection.commit()
                raise AssertionError("Foreign key incorrectly accepted an orphan detection.")
            except sqlite3.IntegrityError:
                connection.rollback()
        finally:
            connection.close()

    print("Database tests passed.")
    print("Inserted 3 detected-damage inspections and 1 state-gated inspection.")
    print("Recent, REVIEW, damage-class, daily-count, and analytics queries passed.")
    print("Foreign-key enforcement correctly rejected an orphan detection.")
    print("No model was loaded or retrained by this test.")


if __name__ == "__main__":
    main()
