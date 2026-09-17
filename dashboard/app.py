"""
# ============================================================
# What are we doing in this script?
# ============================================================
#
# Streamlit turns ordinary Python into an interactive web interface. This app
# lets a warehouse user upload an image, detect every parcel and its state, then
# conditionally run frozen MobileNet and damage YOLO on each closed parcel crop,
# and view history and simple analytics.
#
# This file contains no training, backpropagation, threshold tuning, or new SQL.
# It reuses the Phase 16 inference pipeline and Phase 17 database functions.
#
# Architecture:
#   UI -> generic parcel YOLO -> crop -> state model -> closed only -> MobileNet -> damage YOLO
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import torch


# ``streamlit run dashboard/app.py`` starts from the project folder, but Python
# first places dashboard/ on its import path. Adding the project root lets us
# import the existing prototype package without copying any inference code.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.dashboard_helpers import (  # noqa: E402
    get_dashboard_snapshot,
    get_history_rows,
    get_review_queue,
    resolve_stored_image_path,
    run_uploaded_inspection,
)
from prototype.database import (  # noqa: E402
    DEFAULT_DATABASE_PATH,
    get_inspection_by_id,
    initialize_database,
)
from prototype.inspection_pipeline import (  # noqa: E402
    load_prototype_models,
    save_completed_inspection,
)


# st.set_page_config controls the browser title and uses the available screen
# width. It must be the first Streamlit command in the script.
st.set_page_config(page_title="Parcel Inspection", page_icon="📦", layout="wide")


@st.cache_resource(show_spinner="Loading parcel detector, state model, MobileNet, and damage detector...")
def get_frozen_models():
    """Load heavy model objects once and reuse them across Streamlit reruns.

    Streamlit normally executes this script from top to bottom whenever a user
    interacts with a widget. cache_resource prevents checkpoint reloads on every
    click. It does not change model weights or predictions.
    """
    return load_prototype_models()


def initialize_session_state() -> None:
    """Create values that should survive Streamlit's top-to-bottom reruns."""
    # session_state is short-term memory for this browser session. Without it,
    # the latest result and its saved status would disappear after a button click.
    defaults = {
        "latest_inspection": None,
        "latest_saved": False,
        "latest_saved_id": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_model_information() -> None:
    """Show frozen-model traceability without overwhelming the main screen."""
    with st.expander("Models used by this prototype"):
        st.markdown(
            """
**Classifier**

- MobileNetV3-Large with ImageNet V2 pretrained features
- Strictly frozen backbone; model version `transfer-mobile-v1`
- Frozen damaged threshold: `0.434584259987`

**Detector**

- YOLO11n, experiment `EXP-002`
- Known classes: `minor_damage`, `compressed`, `hole`, `wet`

**Generic parcel detector**

- YOLO11n `PARCEL-DET-001`, class: `Boxes`
- First-stage localization for every phone-image parcel

**Parcel-state classifier**

- Separate crop-level MobileNetV3 classifier at `models/parcel_state_classifier/best.pt`
- Required outputs: `closed_box`, `open_box`

Open parcels stop after crop-level state classification. Every closed parcel crop is classified;
damage YOLO runs only when MobileNet reports damaged.
"""
        )


def show_latest_result() -> None:
    """Render the current result and provide one explicit database-save action."""
    latest = st.session_state.latest_inspection
    if latest is None:
        return
    evidence = latest["evidence"]

    st.divider()
    left, right = st.columns(2)
    with left:
        st.subheader("Original image")
        st.image(latest["original_path"], width="stretch")
    with right:
        st.subheader("Annotated image")
        st.image(latest["annotated_path"], width="stretch")

    st.subheader("Per-parcel results")
    if evidence["status"] == "no_parcel_detected":
        st.warning("No parcel detected. MobileNet and damage YOLO were not run.")
    for parcel in evidence["parcels"]:
        with st.container(border=True):
            st.write(
                f"**Parcel {parcel['parcel_index'] + 1}: {parcel['final_status']}** — "
                f"{parcel['parcel_state']} "
                + (f"({parcel['parcel_state_confidence']:.2%})" if parcel['parcel_state_confidence'] is not None else "")
            )
            if parcel["damage_status"] is not None:
                st.write(f"MobileNet: {parcel['damage_status']} ({parcel['damage_confidence']:.2%} damaged)")
            for damage in parcel["damages"]:
                st.write(f"{damage['class']} — {damage['confidence']:.2%}")
            if parcel["warning"]:
                st.warning(parcel["warning"])

    decision = evidence["prototype_decision"]
    st.subheader("Final prototype decision")
    if decision == "ACCEPT":
        st.success("ACCEPT")
    else:
        st.warning("REVIEW")
    st.write(evidence["prototype_decision_reason"])
    st.caption("Any model disagreement is routed to REVIEW; the prototype does not issue REJECT decisions.")

    st.subheader("Prototype timing")
    timing_columns = st.columns(4)
    timing_columns[0].metric("Parcel detection", f"{evidence['timing_ms']['parcel_detection']:.1f} ms")
    timing_columns[1].metric("Classifier forward", f"{evidence['timing_ms']['classifier_forward']:.1f} ms")
    timing_columns[2].metric("Damage detection", f"{evidence['timing_ms']['yolo_inference']:.1f} ms")
    timing_columns[3].metric("Total pipeline", f"{evidence['timing_ms']['total_pipeline']:.1f} ms")
    st.caption("Latency varies with CUDA warm-up, GPU state, image size, and machine load; this is not a certified benchmark.")

    # The Save button is separate from Inspect. session_state remembers whether
    # this exact result was saved, preventing duplicate rows during reruns.
    if st.session_state.latest_saved:
        st.success(f"Inspection saved successfully (database row {st.session_state.latest_saved_id}).")
    elif st.button("Save Inspection", type="primary"):
        try:
            inspection_id = save_completed_inspection(
                evidence=evidence,
                image_path=latest["original_path"],
                annotated_image_path=latest["annotated_path"],
                package_id=latest["package_id"] or None,
                database_path=DEFAULT_DATABASE_PATH,
            )
            st.session_state.latest_saved = True
            st.session_state.latest_saved_id = inspection_id
            st.success(f"Inspection saved successfully (database row {inspection_id}).")
        except (ValueError, sqlite3.DatabaseError) as error:
            st.error(f"The inspection could not be saved: {error}")


def render_inspection_page() -> None:
    """Upload, inspect, display, and optionally save one parcel image."""
    st.title("📦 Parcel Inspection")
    st.write("Upload an image to localize the parcel, then classify and inspect its crop for damage.")

    package_id = st.text_input(
        "Package ID (optional)",
        help="A production warehouse would normally supply a barcode ID. If blank, SQLite creates a prototype ID.",
    )
    # st.file_uploader keeps uploaded bytes in memory and restricts the browser's
    # file picker to the formats supported by this educational prototype.
    uploaded_file = st.file_uploader("Parcel image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded parcel", width=500)

    # st.button is an explicit action: merely uploading does not run either model.
    if st.button("Inspect Parcel", type="primary"):
        # Once a new inspection is requested, an older result must not remain
        # visible or saveable if validation or inference fails.
        st.session_state.latest_inspection = None
        st.session_state.latest_saved = False
        st.session_state.latest_saved_id = None
        if uploaded_file is None:
            st.error("Please upload a JPG, JPEG, or PNG parcel image first.")
        else:
            try:
                models = get_frozen_models()
                with st.spinner("Localizing parcel, then running condition inspection..."):
                    result = run_uploaded_inspection(
                        file_bytes=uploaded_file.getvalue(),
                        filename=uploaded_file.name,
                        package_id=package_id,
                        loaded_models=models,
                    )
                st.session_state.latest_inspection = result
            except (FileNotFoundError, ValueError) as error:
                st.error(str(error))
            except Exception as error:
                # Normal users receive a concise message rather than a giant
                # stack trace. Developers can still inspect the terminal log.
                st.error(f"Inference failed: {type(error).__name__}: {error}")

    show_latest_result()
    render_model_information()


def render_history_page() -> None:
    """Show recent rows and a linked inspection/detection detail view."""
    st.title("Inspection History")
    try:
        rows = get_history_rows()
        if not rows:
            st.info("No inspections have been saved yet.")
            return
        frame = pd.DataFrame(rows)
        frame["Damaged probability"] = frame["Damaged probability"].map(
            lambda value: "Not run" if pd.isna(value) else f"{value:.2%}"
        )
        frame["Total latency (ms)"] = frame["Total latency (ms)"].map(lambda value: f"{value:.1f}")
        # st.dataframe provides a scrollable, sortable table without custom HTML.
        st.dataframe(frame, hide_index=True, width="stretch")

        selected_id = st.selectbox("Select an inspection for details", [row["ID"] for row in rows])
        detail = get_inspection_by_id(int(selected_id))
        if detail is None:
            st.warning("That inspection no longer exists.")
            return
        st.subheader(f"Inspection {detail['id']} — {detail['package_id']}")
        st.write(f"Timestamp: {detail['timestamp_utc']}")
        if detail.get("classifier_executed", 1):
            st.write(f"Classifier: **{detail['classifier_label'].upper()}** ({detail['classifier_probability']:.2%})")
        else:
            st.write("Classifier: **NOT RUN** (the parcel-state gate stopped downstream inspection)")
        st.write(f"Decision: **{detail['prototype_decision']}**")
        st.write(f"Total latency: {detail['total_latency_ms']:.1f} ms")

        image_columns = st.columns(2)
        original = resolve_stored_image_path(detail["image_path"])
        annotated = resolve_stored_image_path(detail["annotated_image_path"])
        with image_columns[0]:
            st.caption("Original")
            st.image(original, width="stretch") if original else st.warning("Stored original image is missing.")
        with image_columns[1]:
            st.caption("Annotated")
            st.image(annotated, width="stretch") if annotated else st.warning("Stored annotated image is missing.")

        if detail["detections"]:
            detection_frame = pd.DataFrame(detail["detections"])[
                ["class_name", "confidence", "x1", "y1", "x2", "y2"]
            ]
            st.dataframe(detection_frame, hide_index=True, width="stretch")
        elif detail.get("damage_detector_executed", 1):
            st.info("No linked YOLO detections were stored for this inspection.")
        else:
            st.info("Damage YOLO was not run for this inspection.")
    except sqlite3.DatabaseError as error:
        st.error(f"Inspection history could not be loaded: {error}")


def render_review_page() -> None:
    """Show the conservative manual-review queue."""
    st.title("Review Queue")
    st.write("These parcels require a person to inspect the available evidence.")
    try:
        rows = get_review_queue()
        if not rows:
            st.success("The REVIEW queue is currently empty.")
            return
        frame = pd.DataFrame(rows)
        frame["Damaged probability"] = frame["Damaged probability"].map(
            lambda value: "Not run" if pd.isna(value) else f"{value:.2%}"
        )
        st.dataframe(frame, hide_index=True, width="stretch")
    except sqlite3.DatabaseError as error:
        st.error(f"The REVIEW queue could not be loaded: {error}")


def render_analytics_page() -> None:
    """Display basic counts derived from real SQLite rows only."""
    st.title("Inspection Analytics")
    try:
        snapshot = get_dashboard_snapshot()
        analytics = snapshot["analytics"]
        cards = st.columns(4)
        cards[0].metric("Total inspections", int(analytics["total_inspections"] or 0))
        cards[1].metric("ACCEPT", int(analytics["accept_count"] or 0))
        cards[2].metric("REVIEW", int(analytics["review_count"] or 0))
        average_latency = analytics["average_total_latency_ms"]
        cards[3].metric("Average pipeline latency", "No data" if average_latency is None else f"{average_latency:.1f} ms")

        st.subheader("YOLO damage-class distribution")
        class_counts = analytics["damage_class_counts"]
        if class_counts:
            class_frame = pd.DataFrame(
                {"Damage class": list(class_counts), "Detection count": list(class_counts.values())}
            ).set_index("Damage class")
            st.bar_chart(class_frame)
        else:
            st.info("No YOLO detections have been saved yet.")

        st.subheader("Daily inspection count (UTC)")
        daily_rows = snapshot["daily_counts"]
        if daily_rows:
            daily_frame = pd.DataFrame(daily_rows).set_index("inspection_date")
            st.bar_chart(daily_frame)
            if len(daily_rows) < 2:
                st.caption("Only one day is available, so this is a count—not a meaningful trend yet.")
        else:
            st.info("No daily inspection data is available yet.")
        render_model_information()
    except sqlite3.DatabaseError as error:
        st.error(f"Analytics could not be loaded: {error}")


def main() -> None:
    """Initialize storage and route the user to one simple dashboard page."""
    initialize_session_state()
    try:
        initialize_database()
    except sqlite3.DatabaseError as error:
        st.error(f"The SQLite database could not be initialized: {error}")
        return

    st.sidebar.title("Warehouse QC")
    # A sidebar radio is a beginner-friendly alternative to a complex router.
    selected_page = st.sidebar.radio(
        "Page",
        ["Parcel Inspection", "Inspection History", "Review Queue", "Analytics"],
    )
    selected_device = "CUDA GPU" if torch.cuda.is_available() else "CPU"
    st.sidebar.caption(f"Inference device available: {selected_device}")

    pages = {
        "Parcel Inspection": render_inspection_page,
        "Inspection History": render_history_page,
        "Review Queue": render_review_page,
        "Analytics": render_analytics_page,
    }
    pages[selected_page]()


if __name__ == "__main__":
    main()
