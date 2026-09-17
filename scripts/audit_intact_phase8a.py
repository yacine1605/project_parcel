"""Create and finalize the Phase 8A nominal-intact adjudication manifest.

This script never changes raw data or an existing processed dataset. It creates
review sheets and propagates human group decisions to every retained nominal-
intact image. Run once to create templates, fill group_decisions.csv, then rerun.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


LABEL_DEFINITIONS = {
    "intact": "Parcel appears closed/sealed with no visible dent, compression, hole, tear, moisture damage, crushed corner, or open flap.",
    "damaged": "Visible structural or material damage: dent/compression, crush, hole/puncture, tear, wet damage/staining, or materially crushed corner.",
    "open_box": "One or more flaps/lids are visibly open or the package is presented as an open carton, without enough evidence to call structural damage.",
    "ambiguous": "Condition cannot be decided confidently because of resolution, occlusion, illustration/rendering, incomplete parcel view, borderline deformation, or conflicting views.",
}

# Explicit findings from the complete 25-sheet visual pass. Group identifiers
# make decisions reproducible and keep every augmentation sibling consistent.
OPEN_BOX_GROUPS = {
    "grp_00051", "grp_00058", "grp_00062", "grp_00063", "grp_00083",
    "grp_00541", "grp_00571", "grp_00583", "grp_00584", "grp_00593",
    "grp_00599", "grp_00600", "grp_00603", "grp_00607", "grp_00608",
    "grp_00640", "grp_00644", "grp_00654", "grp_00667", "grp_00675",
    "grp_00678", "grp_00679", "grp_00683", "grp_00684", "grp_00685",
    "grp_00687", "grp_00692", "grp_00694", "grp_00709", "grp_00735",
    "grp_00760", "grp_00762", "grp_00763", "grp_00770", "grp_00784",
    "grp_00808", "grp_00834", "grp_00840", "grp_00842", "grp_00847",
    "grp_00854", "grp_00868", "grp_00871", "grp_00875", "grp_00895",
    "grp_00903", "grp_00946", "grp_00950", "grp_01082", "grp_01110",
    "grp_01126", "grp_01143", "grp_01146", "grp_01163",
}
DAMAGED_GROUPS = {
    "grp_00059", "grp_00521", "grp_00533", "grp_00547", "grp_00578",
    "grp_00626", "grp_00627", "grp_00633", "grp_00652", "grp_00730",
    "grp_00738", "grp_00740", "grp_00781", "grp_00801", "grp_00865",
    "grp_00880", "grp_00920", "grp_00922", "grp_00924", "grp_00936",
}
AMBIGUOUS_GROUPS = {
    "grp_00045", "grp_00067", "grp_00078", "grp_00084", "grp_00529",
    "grp_00530", "grp_00531", "grp_00548", "grp_00569", "grp_00577",
    "grp_00615", "grp_00624", "grp_00645", "grp_00648", "grp_00668",
    "grp_00676", "grp_00682", "grp_00714", "grp_00722", "grp_00741",
    "grp_00766", "grp_00779", "grp_00783", "grp_00785", "grp_00837",
    "grp_00838", "grp_00841", "grp_00844", "grp_00848", "grp_00851",
    "grp_00857", "grp_00861", "grp_00868", "grp_00887", "grp_00893",
    "grp_00901", "grp_00913", "grp_00938", "grp_00955", "grp_00956",
    "grp_00957", "grp_01055", "grp_01056", "grp_01069", "grp_01104",
    "grp_01119", "grp_01139", "grp_01140", "grp_01141", "grp_01144",
    "grp_01147", "grp_01149", "grp_01150", "grp_01153", "grp_01154",
    "grp_01155", "grp_01166", "grp_01171",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def build_review_material(dataset: Path, output: Path, intact: list[dict[str, str]]) -> None:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in intact:
        groups[row["group_id"]].append(row)
    decisions_path = output / "manifests" / "group_decisions.csv"
    existing = {r["group_id"]: r for r in read_csv(decisions_path)} if decisions_path.exists() else {}
    decision_rows = []
    for gid, rows in sorted(groups.items()):
        representative = sorted(rows, key=lambda r: r["image_id"])[0]
        old = existing.get(gid, {})
        count = len(rows)
        label, confidence, flags, notes = "intact", "medium", "", "No visible damage or open flap in group representative; group has at most three source variants."
        if count > 3:
            label, confidence, flags, notes = "ambiguous", "low", "group_collision_risk", "More than three files share a conservative Phase 7 filename group; one representative cannot safely adjudicate all images."
        if gid in AMBIGUOUS_GROUPS:
            label, confidence, flags, notes = "ambiguous", "low", "visual_uncertainty", "Visual condition, parcel identity, or group consistency cannot be determined confidently."
        if gid in OPEN_BOX_GROUPS:
            label, confidence, flags, notes = "open_box", "high", "open_flap_or_lid", "One or more flaps/lids are visibly open; no structural-damage label inferred from opening alone."
        if gid in DAMAGED_GROUPS:
            label, confidence, flags, notes = "damaged", "high", "visible_structural_damage", "Visible crush, dent, tear, compression, or materially deformed corner."
        decision_rows.append({
            "group_id": gid, "representative_image_id": representative["image_id"],
            "representative_original_path": representative["original_path"], "image_count": str(len(rows)),
            "adjudicated_label": old.get("adjudicated_label") or label,
            "confidence": old.get("confidence") or confidence, "visual_flags": old.get("visual_flags") or flags,
            "review_notes": old.get("review_notes") or notes, "reviewer": old.get("reviewer") or "Codex visual contact-sheet review",
        })
    fields = ["group_id", "representative_image_id", "representative_original_path", "image_count", "adjudicated_label", "confidence", "visual_flags", "review_notes", "reviewer"]
    write_csv(decisions_path, decision_rows, fields)

    sheets = output / "review_sheets"
    sheets.mkdir(parents=True, exist_ok=True)
    cell_w, cell_h, cols, rows_n = 220, 205, 5, 5
    for offset in range(0, len(decision_rows), cols * rows_n):
        batch = decision_rows[offset : offset + cols * rows_n]
        canvas = Image.new("RGB", (cell_w * cols, cell_h * rows_n), "white")
        draw = ImageDraw.Draw(canvas)
        for n, row in enumerate(batch):
            source = dataset / row["new_split"] if "new_split" in row else None
            # The derived copy is found from the main manifest fields below.
            original = next(r for r in intact if r["image_id"] == row["representative_image_id"])
            path = dataset / original["new_split"] / "intact" / f'{original["image_id"]}.jpg'
            with Image.open(path) as image:
                thumb = ImageOps.contain(image.convert("RGB"), (cell_w, 165), Image.Resampling.LANCZOS)
            x, y = (n % cols) * cell_w, (n // cols) * cell_h
            canvas.paste(thumb, (x + (cell_w - thumb.width) // 2, y))
            draw.rectangle((x, y, x + cell_w - 1, y + cell_h - 1), outline="#777777")
            draw.text((x + 4, y + 168), f'{row["group_id"]}  n={row["image_count"]}', fill="black")
            draw.text((x + 4, y + 184), row["representative_image_id"], fill="black")
        number = offset // (cols * rows_n) + 1
        canvas.save(sheets / f"group_review_{number:02d}.jpg", quality=92)


def finalize(dataset: Path, output: Path, manifest: list[dict[str, str]]) -> dict[str, object]:
    intact = [r for r in manifest if r["binary_class"] == "intact" and r["audit_status"] != "excluded"]
    decisions = {r["group_id"]: r for r in read_csv(output / "manifests" / "group_decisions.csv")}
    fields = list(manifest[0]) + ["phase8a_label", "phase8a_confidence", "phase8a_visual_flags", "phase8a_review_notes", "phase8a_reviewer", "phase8a_status", "parcel_binary_v2_action"]
    reviewed = []
    for row in manifest:
        out = dict(row)
        if row["binary_class"] == "intact" and row["audit_status"] != "excluded":
            decision = decisions[row["group_id"]]
            label = decision["adjudicated_label"]
            out.update({
                "phase8a_label": label, "phase8a_confidence": decision["confidence"],
                "phase8a_visual_flags": decision["visual_flags"], "phase8a_review_notes": decision["review_notes"],
                "phase8a_reviewer": decision["reviewer"],
                "phase8a_status": "reviewed" if label in LABEL_DEFINITIONS else "pending_review",
                "parcel_binary_v2_action": {"intact": "include_as_intact", "damaged": "reclassify_as_damaged", "open_box": "exclude_pending_open_box_policy", "ambiguous": "exclude_pending_manual_re_review"}.get(label, "exclude_pending_review"),
            })
        else:
            action = "include_as_damaged" if row["binary_class"] == "damaged" and row["audit_status"] != "excluded" else "retain_exclusion"
            out.update({"phase8a_label": row["binary_class"] if row["audit_status"] != "excluded" else "excluded", "phase8a_confidence": "not_reviewed_phase8a", "phase8a_visual_flags": "", "phase8a_review_notes": "Phase 8A scope was nominal intact only", "phase8a_reviewer": "", "phase8a_status": "out_of_scope", "parcel_binary_v2_action": action})
        reviewed.append(out)
    write_csv(output / "manifests" / "reviewed_dataset_manifest.csv", reviewed, fields)
    group_counts = Counter(d["adjudicated_label"] or "pending_review" for d in decisions.values())
    image_counts = Counter(r["phase8a_label"] or "pending_review" for r in reviewed if r["binary_class"] == "intact" and r["audit_status"] != "excluded")
    summary = {"nominal_intact_images": len(intact), "nominal_intact_groups": len(decisions), "group_label_counts": dict(group_counts), "image_label_counts": dict(image_counts), "label_definitions": LABEL_DEFINITIONS}
    (output / "manifests" / "phase8a_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "README.md").write_text(
        "# Parcel Binary V2 Review Staging\n\n"
        "Phase 8A adjudication artifacts only; this is not a frozen parcel_binary_v2 dataset. "
        "Use `manifests/reviewed_dataset_manifest.csv` and its `parcel_binary_v2_action` field. "
        "Open-box images remain a separate policy label and ambiguous images remain excluded from "
        "automatic v2 construction. Re-split eligible groups deterministically; do not train from "
        "this staging directory.\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("datasets/processed/parcel_binary_v1"))
    parser.add_argument("--output", type=Path, default=Path("datasets/processed/parcel_binary_v2_review"))
    args = parser.parse_args()
    manifest = read_csv(args.dataset / "manifests" / "dataset_manifest.csv")
    intact = [r for r in manifest if r["binary_class"] == "intact" and r["audit_status"] != "excluded"]
    args.output.mkdir(parents=True, exist_ok=True)
    build_review_material(args.dataset, args.output, intact)
    print(json.dumps(finalize(args.dataset, args.output, manifest), indent=2))


if __name__ == "__main__":
    main()
