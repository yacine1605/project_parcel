"""
# ============================================================
# What does this script do?
# ============================================================
#
# This script created the frozen ``parcel_binary_v2`` image dataset from the
# Phase 8B adjudication manifest.
#
# Input:
#   - parcel_binary_v1 images
#   - a reviewed inclusion manifest containing labels and group IDs
#
# Output:
#   - train/valid/test image folders
#   - a dataset manifest containing paths and SHA-256 hashes
#   - a leakage-audit summary
#
# Models used:
#   None. This is dataset preparation, not machine learning training.
#
# Important:
#   Images with the same group ID stay in one split. This prevents related
#   images from appearing in both development and evaluation data.
#
# Pipeline:
#   reviewed manifest
#       -> keep approved rows
#       -> assign whole groups to splits
#       -> copy images
#       -> calculate hashes
#       -> verify no group/hash leakage
#       -> freeze dataset metadata
#
# The existing v2 dataset is frozen. This script refuses to overwrite it.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path


SPLITS = ("train", "valid", "test")
CLASSES = ("damaged", "intact")
RATIOS = {"train": 0.70, "valid": 0.15, "test": 0.15}


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read the CSV manifest as one dictionary per image.

    A manifest row is metadata: image ID, group ID, reviewed class, inclusion
    decision, and related audit fields. Reading metadata does not open images.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        return list[dict[str, str]](csv.DictReader(handle))


def assign_groups(rows: list[dict[str, str]], seed: int) -> dict[str, str]:
    """Assign each complete leakage-control group to exactly one split.

    Group assignment matters because augmented siblings or duplicate-related
    images must travel together. Splitting individual rows could let nearly the
    same parcel image appear in both training and validation/test data.
    """
    by_group: dict[str, list[dict[str, str]]] = defaultdict[str, list[dict[str, str]]](list)
    for row in rows:
        by_group[row["group_id"]].append(row)

    assignments: dict[str, str] = {}
    rng = random.Random(seed)

    for label in CLASSES:
        # Select groups whose every included member has this one reviewed label.
        # The explicit loop keeps the label-consistency rule visible.
        groups = []
        for group_id, members in by_group.items():
            labels_in_group = {
                member["phase8b_binary_qc_class"]
                for member in members
            }
            if labels_in_group == {label}:
                groups.append((group_id, members))

        # The seeded shuffle makes equal-sized groups reproducibly ordered. The
        # following stable sort places large groups first, improving balance.
        rng.shuffle(groups)
        groups.sort(key=lambda item: len(item[1]), reverse=True)

        total_images_for_label = sum(len(members) for _, members in groups)
        target = {
            split: RATIOS[split] * total_images_for_label
            for split in SPLITS
        }
        counts = Counter()

        for group_id, members in groups:
            # Choose the split furthest below its target. SPLITS order provides
            # the unchanged deterministic tie-break: train, then valid, then test.
            selected_split = max(
                SPLITS,
                key=lambda split: (
                    target[split] - counts[split],
                    -SPLITS.index(split),
                ),
            )
            assignments[group_id] = selected_split
            counts[selected_split] += len(members)

    if len(assignments) != len(by_group):
        raise ValueError("Some included groups have missing or conflicting binary labels")
    return assignments


def locate_source(root: Path, row: dict[str, str]) -> Path:
    """Find the one v1 image represented by a reviewed manifest row."""
    folder = root / row["new_split"] / row["binary_class"]
    matches = list(folder.glob(f'{row["image_id"]}.*'))
    if len(matches) != 1:
        raise FileNotFoundError(f'Expected one source for {row["image_id"]}, found {len(matches)}')
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the adjudicated, leakage-controlled parcel_binary_v2 dataset."
    )
    parser.add_argument("--source", type=Path, default=Path("datasets/processed/parcel_binary_v1"))
    parser.add_argument("--manifest", type=Path, default=Path("datasets/processed/parcel_binary_v2_review/phase8b/manifests/parcel_binary_v2_inclusion_manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("datasets/processed/parcel_binary_v2"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen-version target: {args.output}")

    # ---------------------------------------------------------
    # 1. Read adjudication metadata and retain approved images
    # ---------------------------------------------------------
    all_rows = read_rows(args.manifest)
    included_rows = []
    for row in all_rows:
        if row["phase8b_include"] == "yes":
            included_rows.append(row)

    # ---------------------------------------------------------
    # 2. Assign complete groups to train, validation, or test
    # ---------------------------------------------------------
    assignments = assign_groups(included_rows, args.seed)

    # ---------------------------------------------------------
    # 3. Create the new version's empty directory structure
    # ---------------------------------------------------------
    for split in SPLITS:
        for label in CLASSES:
            (args.output / split / label).mkdir(parents=True, exist_ok=False)
    (args.output / "manifests").mkdir()

    output_rows = []
    hashes: dict[str, list[tuple[str, str]]] = defaultdict(list)
    try:
        # -----------------------------------------------------
        # 4. Copy every included image and record its hash
        # -----------------------------------------------------
        for row in included_rows:
            source = locate_source(args.source, row)
            split = assignments[row["group_id"]]
            label = row["phase8b_binary_qc_class"]
            destination = args.output / split / label / f'{row["image_id"]}{source.suffix.lower()}'
            shutil.copy2(source, destination)

            # SHA-256 identifies byte-identical images. If one hash occurs in
            # multiple splits, exact image leakage has occurred.
            digest = hashlib.sha256(destination.read_bytes()).hexdigest()
            hashes[digest].append((split, row["image_id"]))

            out = dict(row)
            out["v2_split"] = split
            out["v2_binary_class"] = label
            out["v2_relative_path"] = destination.relative_to(args.output).as_posix()
            out["sha256"] = digest
            output_rows.append(out)

        # -----------------------------------------------------
        # 5. Audit group leakage and exact-hash leakage
        # -----------------------------------------------------
        group_splits: dict[str, set[str]] = defaultdict(set)
        for row in output_rows:
            group_splits[row["group_id"]].add(row["v2_split"])

        leaking_groups = {
            group_id: sorted(used_splits)
            for group_id, used_splits in group_splits.items()
            if len(used_splits) > 1
        }
        cross_hashes = {
            digest: locations
            for digest, locations in hashes.items()
            if len({split for split, _ in locations}) > 1
        }
        counts = {}
        for split in SPLITS:
            split_labels = [
                row["v2_binary_class"]
                for row in output_rows
                if row["v2_split"] == split
            ]
            counts[split] = dict(Counter(split_labels))

        # -----------------------------------------------------
        # 6. Save reproducibility metadata
        # -----------------------------------------------------
        summary = {
            "dataset_version": "parcel_binary_v2",
            "frozen": not leaking_groups and not cross_hashes,
            "seed": args.seed,
            "target_ratios": RATIOS,
            "included_images": len(output_rows),
            "included_groups": len(group_splits),
            "split_class_counts": counts,
            "group_leakage": leaking_groups,
            "exact_hash_cross_split_leakage": cross_hashes,
            "excluded_images_preserved_in_phase8b_manifest": len(all_rows) - len(included_rows),
        }
        fields = list(output_rows[0])
        with (args.output / "manifests" / "dataset_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(output_rows)
        (args.output / "manifests" / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (args.output / "README.md").write_text(
            "# Parcel Binary V2\n\nFrozen adjudicated damaged-vs-intact dataset. Open boxes map to "
            "`damaged` for warehouse QC while their original adjudication is preserved in the manifest. "
            "The seed-42 split is group-stratified; test must remain closed until final evaluation.\n",
            encoding="utf-8")
        if not summary["frozen"]:
            raise RuntimeError("Leakage audit failed; dataset must not be used")
        print(json.dumps(summary, indent=2))
    except Exception:
        # A partially built dataset could be mistaken for a valid frozen version.
        # Remove only the new output folder created by this attempted build.
        shutil.rmtree(args.output, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()


# ============================================================
# What you should understand after reading this script
# ============================================================
#
# 1. Why related images must remain in one split.
# 2. Why a fixed random seed makes split construction reproducible.
# 3. How SHA-256 detects exact cross-split leakage.
# 4. Why manifests make a derived dataset auditable.
# 5. Why the frozen output is never silently overwritten.
