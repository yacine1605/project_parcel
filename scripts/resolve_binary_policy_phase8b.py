"""Resolve Phase 8B warehouse-QC binary policy and second-review decisions.

The review sheets intentionally omit Phase 8A decisions, flags, and notes.
This script creates metadata/manifests only and never copies or changes images.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


FINAL_LABELS = {"intact", "damaged", "open_box", "ambiguous"}

# Blinded second-review results after viewing all 287 images. These identifiers
# are recorded only after the sheets were generated without Phase 8A metadata.
SECOND_REVIEW_DAMAGED = {"grp_00546", "grp_00682"}
SECOND_REVIEW_OPEN_BOX = {
    "grp_00531", "grp_00569", "grp_00696", "grp_00714", "grp_00722",
    "grp_00779", "grp_00783", "grp_00838", "grp_00851",
}
SECOND_REVIEW_IRREDUCIBLE = {
    "grp_00078", "grp_00841", "grp_00955", "grp_00956", "grp_01171",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def build_blinded_sheets(dataset: Path, output: Path, rows: list[dict[str, str]]) -> None:
    sheets = output / "blinded_second_review_sheets"
    sheets.mkdir(parents=True, exist_ok=True)
    shuffled = list(rows)
    random.Random(808).shuffle(shuffled)
    cell_w, cell_h, cols, rows_n = 220, 205, 5, 5
    for offset in range(0, len(shuffled), cols * rows_n):
        batch = shuffled[offset : offset + cols * rows_n]
        canvas = Image.new("RGB", (cell_w * cols, cell_h * rows_n), "white")
        draw = ImageDraw.Draw(canvas)
        for n, row in enumerate(batch):
            path = dataset / row["new_split"] / "intact" / f'{row["image_id"]}.jpg'
            with Image.open(path) as image:
                thumb = ImageOps.contain(image.convert("RGB"), (cell_w, 165), Image.Resampling.LANCZOS)
            x, y = (n % cols) * cell_w, (n // cols) * cell_h
            canvas.paste(thumb, (x + (cell_w - thumb.width) // 2, y))
            draw.rectangle((x, y, x + cell_w - 1, y + cell_h - 1), outline="#777777")
            draw.text((x + 4, y + 168), row["group_id"], fill="black")
            draw.text((x + 4, y + 184), row["image_id"], fill="black")
        number = offset // (cols * rows_n) + 1
        canvas.save(sheets / f"blinded_review_{number:02d}.jpg", quality=92)


def create_template(output: Path, rows: list[dict[str, str]]) -> Path:
    by_group: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_group[row["group_id"]].append(row)
    path = output / "manifests" / "second_review_group_decisions.csv"
    old = {r["group_id"]: r for r in read_csv(path)} if path.exists() else {}
    result = []
    for gid, members in sorted(by_group.items()):
        prior = old.get(gid, {})
        label, confidence, flags, notes = "intact", "medium", "closed_no_visible_damage", "All group images reviewed blind; no visible structural damage or open flap."
        if gid in SECOND_REVIEW_DAMAGED:
            label, confidence, flags, notes = "damaged", "high", "visible_crush_or_deformation", "All group images reviewed blind; visible structural deformation makes the parcel unacceptable."
        elif gid in SECOND_REVIEW_OPEN_BOX:
            label, confidence, flags, notes = "open_box", "high", "visible_open_flap_or_packing_state", "All group images reviewed blind; an open flap/carton or active open-box packing state is visible."
        elif gid in SECOND_REVIEW_IRREDUCIBLE:
            label, confidence, flags, notes = "ambiguous", "low", "nonparcel_or_insufficient_condition_evidence", "All group images reviewed blind; object identity or parcel condition remains insufficient for a defensible binary label."
        result.append({
            "group_id": gid, "image_count": str(len(members)),
            "final_adjudication_label": prior.get("final_adjudication_label") or label,
            "second_review_confidence": prior.get("second_review_confidence") or confidence,
            "second_review_flags": prior.get("second_review_flags") or flags,
            "second_review_notes": prior.get("second_review_notes") or notes,
            "reviewer": prior.get("reviewer") or "Codex blinded all-image contact-sheet review",
        })
    fields = ["group_id", "image_count", "final_adjudication_label", "second_review_confidence", "second_review_flags", "second_review_notes", "reviewer"]
    write_csv(path, result, fields)
    return path


def finalize(phase8a: list[dict[str, str]], decisions_path: Path, output: Path) -> dict[str, object]:
    decisions = {r["group_id"]: r for r in read_csv(decisions_path)}
    final_groups: dict[str, dict[str, str]] = {}
    for row in phase8a:
        if not row["group_id"]:
            continue
        if row["phase8a_label"] == "ambiguous" and row["audit_status"] != "excluded":
            decision = decisions[row["group_id"]]
            final_groups[row["group_id"]] = {
                "group_id": row["group_id"], "phase8a_label": "ambiguous",
                "final_adjudication_label": decision["final_adjudication_label"],
                "binary_qc_class": "damaged" if decision["final_adjudication_label"] in {"damaged", "open_box"} else "intact" if decision["final_adjudication_label"] == "intact" else "",
                "include_in_parcel_binary_v2": "yes" if decision["final_adjudication_label"] != "ambiguous" else "no",
                "exclusion_reason": "" if decision["final_adjudication_label"] != "ambiguous" else "irreducibly ambiguous after blinded second review",
                "review_confidence": decision["second_review_confidence"], "review_flags": decision["second_review_flags"],
                "review_notes": decision["second_review_notes"], "reviewer": decision["reviewer"],
            }
        elif row["audit_status"] != "excluded":
            label = row["phase8a_label"]
            final_groups.setdefault(row["group_id"], {
                "group_id": row["group_id"], "phase8a_label": label, "final_adjudication_label": label,
                "binary_qc_class": "damaged" if label in {"damaged", "open_box"} else "intact",
                "include_in_parcel_binary_v2": "yes", "exclusion_reason": "",
                "review_confidence": row["phase8a_confidence"], "review_flags": row["phase8a_visual_flags"],
                "review_notes": row["phase8a_review_notes"], "reviewer": row["phase8a_reviewer"],
            })
    # Preserve group-level provenance for groups excluded before Phase 8B.
    by_group_all: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in phase8a:
        if row["group_id"]:
            by_group_all[row["group_id"]].append(row)
    for gid, members in by_group_all.items():
        if gid not in final_groups:
            reasons = sorted({r["exclusion_reason"] for r in members if r["exclusion_reason"]})
            final_groups[gid] = {
                "group_id": gid, "phase8a_label": "excluded", "final_adjudication_label": "excluded",
                "binary_qc_class": "", "include_in_parcel_binary_v2": "no",
                "exclusion_reason": "; ".join(reasons) or "excluded before Phase 8B",
                "review_confidence": "", "review_flags": "", "review_notes": "No retained image in group", "reviewer": "",
            }
    group_fields = ["group_id", "phase8a_label", "final_adjudication_label", "binary_qc_class", "include_in_parcel_binary_v2", "exclusion_reason", "review_confidence", "review_flags", "review_notes", "reviewer"]
    write_csv(output / "manifests" / "final_group_decisions.csv", [final_groups[k] for k in sorted(final_groups)], group_fields)

    fields = list(phase8a[0]) + ["phase8b_final_label", "phase8b_binary_qc_class", "phase8b_include", "phase8b_exclusion_reason", "phase8b_review_confidence", "phase8b_review_flags", "phase8b_review_notes"]
    final_rows = []
    for row in phase8a:
        out = dict(row)
        if row["audit_status"] == "excluded":
            out.update({"phase8b_final_label": "excluded", "phase8b_binary_qc_class": "", "phase8b_include": "no", "phase8b_exclusion_reason": row["exclusion_reason"], "phase8b_review_confidence": "", "phase8b_review_flags": "", "phase8b_review_notes": "Phase 7 exclusion retained"})
        else:
            group = final_groups[row["group_id"]]
            out.update({"phase8b_final_label": group["final_adjudication_label"], "phase8b_binary_qc_class": group["binary_qc_class"], "phase8b_include": group["include_in_parcel_binary_v2"], "phase8b_exclusion_reason": group["exclusion_reason"], "phase8b_review_confidence": group["review_confidence"], "phase8b_review_flags": group["review_flags"], "phase8b_review_notes": group["review_notes"]})
        final_rows.append(out)
    write_csv(output / "manifests" / "parcel_binary_v2_inclusion_manifest.csv", final_rows, fields)

    included = [r for r in final_rows if r["phase8b_include"] == "yes"]
    second = [g for g in final_groups.values() if g["phase8a_label"] == "ambiguous"]
    included_groups = [g for g in final_groups.values() if g["include_in_parcel_binary_v2"] == "yes"]
    summary = {
        "binary_policy": "open_box maps to damaged/unacceptable; original adjudication label preserved",
        "second_review_groups": len(second), "second_review_images": sum(int(decisions[g["group_id"]]["image_count"]) for g in second),
        "second_review_group_labels": dict(Counter(g["final_adjudication_label"] for g in second)),
        "included_images_by_binary_class": dict(Counter(r["phase8b_binary_qc_class"] for r in included)),
        "included_groups_by_binary_class": dict(Counter(g["binary_qc_class"] for g in included_groups)),
        "included_images_by_final_adjudication": dict(Counter(r["phase8b_final_label"] for r in included)),
        "included_images_by_source_split_and_binary_class": {s: dict(Counter(r["phase8b_binary_qc_class"] for r in included if r["source_split"] == s)) for s in ("train", "valid", "test")},
        "included_images_by_phase7_split_and_binary_class": {s: dict(Counter(r["phase8b_binary_qc_class"] for r in included if r["new_split"] == s)) for s in ("train", "valid", "test")},
        "excluded_images_by_reason": dict(Counter(r["phase8b_exclusion_reason"] for r in final_rows if r["phase8b_include"] == "no")),
        "included_total": len(included), "excluded_total": len(final_rows) - len(included),
    }
    (output / "manifests" / "phase8b_class_count_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "README.md").write_text(
        "# Phase 8B Binary Policy Resolution\n\n"
        "Metadata staging for constructing a future frozen `parcel_binary_v2`; no images are "
        "copied here. `open_box` is preserved as an adjudication label and maps to binary "
        "`damaged` for warehouse QC. Only rows marked `phase8b_include=yes` are eligible. "
        "Re-split by `group_id` with seed 42 before freezing v2.\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("datasets/processed/parcel_binary_v1"))
    parser.add_argument("--phase8a", type=Path, default=Path("datasets/processed/parcel_binary_v2_review/manifests/reviewed_dataset_manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("datasets/processed/parcel_binary_v2_review/phase8b"))
    args = parser.parse_args()
    rows = read_csv(args.phase8a)
    ambiguous = [r for r in rows if r["phase8a_label"] == "ambiguous" and r["audit_status"] != "excluded"]
    args.output.mkdir(parents=True, exist_ok=True)
    build_blinded_sheets(args.dataset, args.output, ambiguous)
    decisions = create_template(args.output, ambiguous)
    print(json.dumps(finalize(rows, decisions, args.output), indent=2))


if __name__ == "__main__":
    main()
