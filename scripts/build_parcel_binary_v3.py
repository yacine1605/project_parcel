"""Build corrected parcel_binary_v3 without modifying frozen v2."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "datasets" / "processed" / "parcel_binary_v2"
TARGET = ROOT / "datasets" / "processed" / "parcel_binary_v3"
SOURCE_MANIFEST = SOURCE / "manifests" / "dataset_manifest.csv"
REVIEW = ROOT / "reports" / "open_box_label_audit" / "open_box_group_audit.csv"
ALLOWED = {"open_box", "intact", "minor_damage", "damaged", "ambiguous"}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if TARGET.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {TARGET}")

    with REVIEW.open(newline="", encoding="utf-8-sig") as handle:
        review_rows = list(csv.DictReader(handle))
    if len(review_rows) != 63 or len({row["group_id"] for row in review_rows}) != 63:
        raise ValueError("Expected exactly 63 unique reviewed groups")
    invalid = [row for row in review_rows if row["review_label"] not in ALLOWED]
    if invalid:
        raise ValueError(f"Invalid reviewed labels: {invalid[:3]}")
    if any(row["review_label"] == "ambiguous" for row in review_rows):
        raise ValueError("Resolve or explicitly exclude ambiguous groups before building")
    review = {row["group_id"]: row["review_label"] for row in review_rows}

    with SOURCE_MANIFEST.open(newline="", encoding="utf-8-sig") as handle:
        source_rows = list(csv.DictReader(handle))
        source_fields = list(source_rows[0])

    target_rows: list[dict[str, str]] = []
    split_groups: dict[str, set[str]] = defaultdict(set)
    counts: Counter[tuple[str, str]] = Counter()
    corrected_groups: Counter[str] = Counter()

    for row in source_rows:
        old_binary = row["v2_binary_class"]
        reviewed_label = review.get(row["group_id"], "")
        new_binary = "intact" if reviewed_label == "intact" else old_binary
        if reviewed_label in {"open_box", "minor_damage", "damaged"}:
            new_binary = "damaged"

        relative = Path(row["v2_relative_path"])
        split = row["v2_split"]
        new_relative = Path(split) / new_binary / relative.name
        source_path = SOURCE / relative
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        if file_hash(source_path) != row["sha256"]:
            raise ValueError(f"Source hash mismatch: {source_path}")

        target_path = TARGET / new_relative
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)

        updated = dict(row)
        updated.update(
            {
                "v3_review_label": reviewed_label or row["phase8b_final_label"],
                "v3_binary_class": new_binary,
                "v3_relative_path": new_relative.as_posix(),
                "v3_label_changed": "yes" if new_binary != old_binary else "no",
                "v3_review_source": (
                    "open_box_group_audit" if reviewed_label else "unchanged_from_v2"
                ),
            }
        )
        target_rows.append(updated)
        split_groups[split].add(row["group_id"])
        counts[(split, new_binary)] += 1
        if new_binary != old_binary:
            corrected_groups[row["group_id"]] += 1

    group_splits: dict[str, set[str]] = defaultdict(set)
    hash_splits: dict[str, set[str]] = defaultdict(set)
    for row in target_rows:
        group_splits[row["group_id"]].add(row["v2_split"])
        hash_splits[row["sha256"]].add(row["v2_split"])
    group_leakage = {key: sorted(value) for key, value in group_splits.items() if len(value) > 1}
    hash_leakage = {key: sorted(value) for key, value in hash_splits.items() if len(value) > 1}
    if group_leakage or hash_leakage:
        raise ValueError("Source split integrity check failed")

    manifest_dir = TARGET / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    new_fields = source_fields + [
        "v3_review_label",
        "v3_binary_class",
        "v3_relative_path",
        "v3_label_changed",
        "v3_review_source",
    ]
    with (manifest_dir / "dataset_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=new_fields)
        writer.writeheader()
        writer.writerows(target_rows)

    summary = {
        "dataset_version": "parcel_binary_v3",
        "source_dataset": "parcel_binary_v2",
        "status": "development_only_requires_new_external_final_test",
        "images": len(target_rows),
        "groups": len(group_splits),
        "audited_groups": len(review),
        "label_changed_groups": len(corrected_groups),
        "label_changed_images": sum(corrected_groups.values()),
        "split_class_counts": {
            split: {label: counts[(split, label)] for label in ("damaged", "intact")}
            for split in ("train", "valid", "test")
        },
        "group_cross_split_leakage": len(group_leakage),
        "exact_hash_cross_split_leakage": len(hash_leakage),
        "source_hashes_verified": len(target_rows),
    }
    (manifest_dir / "audit_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (TARGET / "README.md").write_text(
        "# Parcel Binary V3\n\n"
        "Corrected development dataset derived from frozen `parcel_binary_v2`. "
        "The audited open-box groups use the manual decisions in "
        "`reports/open_box_label_audit/open_box_group_audit.csv`. Existing split "
        "assignments are preserved for traceability. Because prior test results "
        "influenced this correction phase, the `test` directory is not a fresh "
        "final test for a retrained model; final claims require new external data.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
