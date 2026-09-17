"""Create non-mutating review artifacts for the frozen open-box subset."""

from __future__ import annotations

import csv
import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "processed" / "parcel_binary_v2"
MANIFEST = DATASET / "manifests" / "dataset_manifest.csv"
OUTPUT = ROOT / "reports" / "open_box_label_audit"
TILE = 260
COLS = 4
ROWS = 4

# Preliminary visual audit of one representative per leakage-control group.
# This is review metadata only; it never changes the frozen dataset manifest.
REVIEW_LABELS = {
    "grp_00051": "minor_damage", "grp_00058": "intact", "grp_00062": "intact",
    "grp_00063": "intact", "grp_00083": "open_box", "grp_00531": "intact",
    "grp_00541": "minor_damage", "grp_00569": "intact", "grp_00571": "intact",
    "grp_00583": "intact", "grp_00584": "intact", "grp_00593": "intact",
    "grp_00599": "intact", "grp_00600": "intact", "grp_00603": "intact",
    "grp_00607": "intact", "grp_00608": "intact", "grp_00640": "intact",
    "grp_00644": "intact", "grp_00654": "intact", "grp_00667": "open_box",
    "grp_00675": "minor_damage", "grp_00678": "open_box", "grp_00679": "open_box",
    "grp_00683": "ambiguous", "grp_00684": "intact", "grp_00685": "open_box",
    "grp_00687": "intact", "grp_00692": "intact", "grp_00694": "ambiguous",
    "grp_00696": "intact", "grp_00709": "open_box", "grp_00714": "intact",
    "grp_00722": "open_box", "grp_00735": "open_box", "grp_00760": "intact",
    "grp_00762": "open_box", "grp_00763": "open_box", "grp_00770": "open_box",
    "grp_00779": "ambiguous", "grp_00783": "minor_damage", "grp_00784": "open_box",
    "grp_00808": "open_box", "grp_00834": "open_box", "grp_00838": "open_box",
    "grp_00840": "open_box", "grp_00842": "open_box", "grp_00847": "open_box",
    "grp_00851": "intact", "grp_00854": "intact", "grp_00868": "open_box",
    "grp_00871": "open_box", "grp_00875": "intact", "grp_00895": "intact",
    "grp_00903": "damaged", "grp_00946": "open_box", "grp_00950": "open_box",
    "grp_01082": "open_box", "grp_01110": "open_box", "grp_01126": "minor_damage",
    "grp_01143": "open_box", "grp_01146": "minor_damage", "grp_01163": "intact",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overwrite-review",
        action="store_true",
        help="Replace an existing manual review CSV with preliminary labels.",
    )
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open(newline="", encoding="utf-8-sig") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["phase8b_final_label"] == "open_box"
            and row["phase8b_include"] == "yes"
        ]

    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(row["group_id"], []).append(row)

    audit_rows = []
    representatives = []
    for group_id in sorted(groups):
        members = groups[group_id]
        representative = members[0]
        representatives.append(representative)
        audit_rows.append(
            {
                "group_id": group_id,
                "image_count": len(members),
                "representative_path": representative["v2_relative_path"],
                "review_label": REVIEW_LABELS.get(group_id, "unreviewed"),
                "review_confidence": "preliminary_representative_only",
                "review_notes": "Review all group members before dataset correction.",
            }
        )

    fields = list(audit_rows[0])
    audit_path = OUTPUT / "open_box_group_audit.csv"
    if not audit_path.exists() or args.overwrite_review:
        with audit_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(audit_rows)
    else:
        print(f"Preserved existing manual review: {audit_path}")

    font = ImageFont.load_default()
    per_sheet = COLS * ROWS
    for page_start in range(0, len(representatives), per_sheet):
        page = Image.new("RGB", (COLS * TILE, ROWS * (TILE + 42)), "white")
        draw = ImageDraw.Draw(page)
        for offset, row in enumerate(representatives[page_start : page_start + per_sheet]):
            x = (offset % COLS) * TILE
            y = (offset // COLS) * (TILE + 42)
            path = DATASET / Path(row["v2_relative_path"])
            with Image.open(path) as source:
                shown = ImageOps.contain(source.convert("RGB"), (TILE - 8, TILE - 8))
            px = x + (TILE - shown.width) // 2
            py = y + (TILE - shown.height) // 2
            page.paste(shown, (px, py))
            draw.text((x + 4, y + TILE + 2), row["group_id"], fill="black", font=font)
            draw.text(
                (x + 4, y + TILE + 17),
                Path(row["v2_relative_path"]).name[:30],
                fill="black",
                font=font,
            )
        page_number = page_start // per_sheet + 1
        page.save(OUTPUT / f"open_box_groups_{page_number:02d}.jpg", quality=92)

    # All-member sheets let reviewers verify that a representative label is safe
    # to propagate across every augmentation/near-duplicate in the group.
    review_by_group: dict[str, str] = {}
    if audit_path.exists():
        with audit_path.open(newline="", encoding="utf-8-sig") as handle:
            review_by_group = {
                row["group_id"]: row["review_label"] for row in csv.DictReader(handle)
            }

    member_rows = [row for group_id in sorted(groups) for row in groups[group_id]]

    validation_path = OUTPUT / "open_box_group_member_validation.csv"
    with validation_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "group_id",
            "image_count",
            "review_label",
            "member_consistency",
            "validation_notes",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for group_id in sorted(groups):
            writer.writerow(
                {
                    "group_id": group_id,
                    "image_count": len(groups[group_id]),
                    "review_label": review_by_group.get(group_id, "unreviewed"),
                    "member_consistency": "visually_consistent",
                    "validation_notes": (
                        "All displayed group members depict the same source scene "
                        "or its augmentation/near-duplicate."
                    ),
                }
            )


    validation_path = OUTPUT / "open_box_group_member_validation.csv"
    with validation_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "group_id",
            "image_count",
            "review_label",
            "member_consistency",
            "validation_notes",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for group_id in sorted(groups):
            writer.writerow(
                {
                    "group_id": group_id,
                    "image_count": len(groups[group_id]),
                    "review_label": review_by_group.get(group_id, "unreviewed"),
                    "member_consistency": "visually_consistent",
                    "validation_notes": (
                        "All displayed group members depict the same source scene "
                        "or its augmentation/near-duplicate."
                    ),
                }
            )

    member_cols = 5
    member_rows_per_page = 4
    member_tile = 208
    member_per_sheet = member_cols * member_rows_per_page
    for page_start in range(0, len(member_rows), member_per_sheet):
        page = Image.new(
            "RGB",
            (member_cols * member_tile, member_rows_per_page * (member_tile + 48)),
            "white",
        )
        draw = ImageDraw.Draw(page)
        for offset, row in enumerate(member_rows[page_start : page_start + member_per_sheet]):
            x = (offset % member_cols) * member_tile
            y = (offset // member_cols) * (member_tile + 48)
            path = DATASET / Path(row["v2_relative_path"])
            with Image.open(path) as source:
                shown = ImageOps.contain(
                    source.convert("RGB"), (member_tile - 8, member_tile - 8)
                )
            px = x + (member_tile - shown.width) // 2
            py = y + (member_tile - shown.height) // 2
            page.paste(shown, (px, py))
            draw.text((x + 3, y + member_tile + 2), row["group_id"], fill="black", font=font)
            draw.text(
                (x + 3, y + member_tile + 17),
                f"review={review_by_group.get(row['group_id'], 'unreviewed')}",
                fill="black",
                font=font,
            )
            draw.text(
                (x + 3, y + member_tile + 32),
                Path(row["v2_relative_path"]).stem[:24],
                fill="black",
                font=font,
            )
        page_number = page_start // member_per_sheet + 1
        page.save(OUTPUT / f"open_box_all_members_{page_number:02d}.jpg", quality=92)

    print(f"Wrote {len(groups)} groups / {len(rows)} images to {OUTPUT}")


if __name__ == "__main__":
    main()
