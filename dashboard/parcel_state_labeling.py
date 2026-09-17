"""Streamlit UI for human review of open/closed parcel crops."""

from pathlib import Path
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prototype.parcel_state_review import (  # noqa: E402
    EXPORT_ROOT,
    MANIFEST_PATH,
    assign_candidate,
    load_manifest,
)

st.set_page_config(page_title="Parcel-state review", page_icon=":material/package_2:", layout="wide")
st.title("Parcel-state image review")
st.caption("Judge each crop yourself. Your choice is saved immediately and copied into the matching classification folder.")

st.session_state.setdefault("review_index", 0)
st.session_state.setdefault("last_saved", "")

if not MANIFEST_PATH.is_file():
    st.error("Review candidates have not been prepared.")
    st.code("python scripts/prepare_parcel_state_classification_review.py")
    st.stop()

rows = load_manifest()
view = st.segmented_control("Show", ["Pending", "All"], default="Pending", key="review_view")
visible = [row for row in rows if view == "All" or not row["decision"]]
reviewed = sum(bool(row["decision"]) for row in rows)

with st.container(horizontal=True):
    st.metric("Reviewed", reviewed)
    st.metric("Remaining", len(rows) - reviewed)
    st.metric("Total", len(rows))
st.progress(reviewed / len(rows) if rows else 1.0)

if st.session_state.last_saved:
    st.success(st.session_state.last_saved)
    st.session_state.last_saved = ""

if not visible:
    st.success(f"All candidates are reviewed. Classification folders are in {EXPORT_ROOT}")
    st.stop()

st.session_state.review_index = min(st.session_state.review_index, len(visible) - 1)
row = visible[st.session_state.review_index]
crop_path = MANIFEST_PATH.parent / row["crop_path"]

with st.container(border=True):
    st.image(str(crop_path), width="stretch")
    st.write(f"**Candidate:** `{row['candidate_id']}`")
    st.caption(
        f"Split: {row['split']} · source annotation: {row['source_label']} · "
        f"suggested only: {row['suggested_label']} · saved decision: {row['decision'] or 'pending'}"
    )


def save(decision: str) -> None:
    assign_candidate(row["candidate_id"], decision)
    st.session_state.last_saved = f"Saved {row['candidate_id']} as {decision}."
    if view == "All" and st.session_state.review_index < len(visible) - 1:
        st.session_state.review_index += 1


with st.container(horizontal=True, horizontal_alignment="distribute"):
    st.button("Closed box", icon=":material/inventory_2:", type="primary", on_click=save, args=("closed_box",))
    st.button("Open box", icon=":material/package_2:", on_click=save, args=("open_box",))
    st.button("Exclude", icon=":material/block:", on_click=save, args=("exclude",))

with st.container(horizontal=True):
    if st.button("Previous", icon=":material/arrow_back:", disabled=st.session_state.review_index == 0):
        st.session_state.review_index -= 1
        st.rerun()
    if st.button("Skip", icon=":material/arrow_forward:", disabled=st.session_state.review_index >= len(visible) - 1):
        st.session_state.review_index += 1
        st.rerun()
