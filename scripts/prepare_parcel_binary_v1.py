"""
# ============================================================
# What does this script do?
# ============================================================
#
# This script audits raw parcel images and created the first leakage-controlled
# damaged/intact dataset used by the classical baseline.
#
# Input:
#   Raw parcel images arranged by source split and source class.
#
# Output:
#   - a derived train/valid/test dataset;
#   - manifests describing every retained or excluded image;
#   - duplicate and near-duplicate audit records;
#   - contact sheets for human review.
#
# Models used:
#   None. Perceptual hashing is an image-similarity check, not a trained model.
#
# Important:
#   Raw files are never changed. Roboflow augmentation siblings, exact
#   duplicates, and conservative perceptual-near-duplicate matches are grouped
#   before the deterministic 70/15/15 split (seed 42).
#
# Pipeline:
#   raw folders -> inventory -> readability/hash audit -> leakage groups
#               -> group-aware split -> derived copies + manifests
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy.fft import dctn


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CLASS_MAP = {"damagedpackages": "damaged", "undamagedpackages": "intact"}
SPLITS = ("train", "valid", "test")
RF_SUFFIX = re.compile(r"\.rf\.[0-9a-f]{32}$", re.I)
VISUALLY_FLAGGED = {
    "f0a7b7c1c675f5d9": "open_box",
    "20e713e8a0c48054": "open_box",
    "786350603b7b8744": "open_box",
    "00524c5c115eb29f": "open_box",
    "f7efe7f26ff0aee2": "open_box",
    "6db320b2f99cfb4e": "open_box",
    "389cb7dbf96b6065": "open_box",
}


@dataclass
class Record:
    """All audit metadata associated with one source image.

    A dataclass makes these named fields easier to understand than positional
    lists. Fields such as SHA-256, dimensions, group, and split are filled as the
    image moves through the audit pipeline.
    """
    path: Path
    relative: str
    source_split: str
    original_class: str
    binary_class: str
    source_key: str
    sha256: str = ""
    phash: int = 0
    width: int = 0
    height: int = 0
    readable: bool = True
    group_id: str = ""
    new_split: str = ""
    duplicate_status: str = "unique"
    audit_status: str = "retained"
    exclusion_reason: str = ""


class UnionFind:
    """Keep related images in one leakage-control group.

    Union-find answers: "Do these two records belong to the same connected
    group?" If A matches B and B matches C, all three must remain together even
    if A was not directly compared with C.
    """
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[b] = a


def source_key(path: Path) -> str:
    # Roboflow preserves the pre-export source name and adds .rf.<32 hex>.
    return RF_SUFFIX.sub("", path.stem).casefold()


def perceptual_hash(image: Image.Image) -> int:
    """Create a compact fingerprint that tolerates small visual changes.

    SHA-256 changes when even one file byte changes. A perceptual hash instead
    summarizes low-frequency visual structure, helping flag resized or lightly
    transformed near duplicates. It is used only for leakage grouping, never to
    infer a damaged/intact label.
    """
    gray = ImageOps.grayscale(image).resize((32, 32), Image.Resampling.LANCZOS)
    coeff = dctn(np.asarray(gray, dtype=np.float32), norm="ortho")[:8, :8]
    flat = coeff.flatten()
    bits = flat > np.median(flat[1:])
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def inventory(raw_root: Path) -> tuple[list[Record], list[str]]:
    """List supported images and record unexpected files or folder layouts."""
    records: list[Record] = []
    unexpected: list[str] = []
    for path in sorted(p for p in raw_root.rglob("*") if p.is_file()):
        rel = path.relative_to(raw_root).as_posix()
        if path.suffix.casefold() not in IMAGE_EXTENSIONS:
            unexpected.append(rel)
            continue
        parts = path.relative_to(raw_root).parts
        if len(parts) < 3 or parts[0] not in SPLITS or parts[1] not in CLASS_MAP:
            unexpected.append(rel)
            continue
        records.append(Record(path, rel, parts[0], parts[1], CLASS_MAP[parts[1]], source_key(path)))
    return records, unexpected


def audit(records: list[Record]) -> tuple[UnionFind, list[tuple[int, int, int]]]:
    """Check readability and group exact, augmented, and near duplicates."""
    uf = UnionFind(len(records))
    by_key: dict[tuple[str, str], list[int]] = defaultdict(list)
    by_sha: dict[str, list[int]] = defaultdict(list)
    readable: list[int] = []
    for i, rec in enumerate(records):
        # Generic web-download names are reused across classes, so a filename
        # alone is not evidence that two differently labelled images match.
        by_key[(rec.binary_class, rec.source_key)].append(i)
        try:
            rec.sha256 = hashlib.sha256(rec.path.read_bytes()).hexdigest()
            with Image.open(rec.path) as image:
                image.load()
                rec.width, rec.height = image.size
                rec.phash = perceptual_hash(image.convert("RGB"))
            readable.append(i)
            by_sha[rec.sha256].append(i)
        except Exception as exc:  # manifest preserves the reason
            rec.readable = False
            rec.audit_status = "excluded"
            rec.exclusion_reason = f"unreadable: {type(exc).__name__}: {exc}"

    # The source key is the strongest grouping signal for the documented rotations.
    for indices in by_key.values():
        for i in indices[1:]:
            uf.union(indices[0], i)
        if len(indices) > 1:
            for i in indices:
                records[i].duplicate_status = "roboflow_augmentation_sibling"

    for indices in by_sha.values():
        for i in indices[1:]:
            uf.union(indices[0], i)
            records[i].duplicate_status = "exact_duplicate"
            records[i].audit_status = "excluded"
            records[i].exclusion_reason = "redundant byte-identical duplicate"
        if len(indices) > 1:
            records[indices[0]].duplicate_status = "exact_duplicate_canonical"

    # Conservative pHash threshold. Only compare similar aspect ratios and never
    # use these pairs to infer labels; they are solely leakage-control groups.
    near_pairs: list[tuple[int, int, int]] = []
    for pos, i in enumerate(readable):
        a = records[i]
        ratio_a = a.width / a.height
        for j in readable[pos + 1 :]:
            b = records[j]
            if a.source_key == b.source_key or a.sha256 == b.sha256:
                continue
            if abs(np.log(ratio_a / (b.width / b.height))) > 0.04:
                continue
            distance = (a.phash ^ b.phash).bit_count()
            if distance <= 3:
                uf.union(i, j)
                near_pairs.append((i, j, distance))
                if records[i].duplicate_status == "unique":
                    records[i].duplicate_status = "near_duplicate_grouped"
                if records[j].duplicate_status == "unique":
                    records[j].duplicate_status = "near_duplicate_grouped"
    return uf, near_pairs


def finalize_groups(records: list[Record], uf: UnionFind) -> dict[str, list[int]]:
    roots: dict[int, list[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        if rec.readable:
            roots[uf.find(i)].append(i)
    groups: dict[str, list[int]] = {}
    for number, (_, indices) in enumerate(sorted(roots.items(), key=lambda x: min(records[i].relative for i in x[1])), 1):
        gid = f"grp_{number:05d}"
        groups[gid] = indices
        classes = {records[i].binary_class for i in indices}
        for i in indices:
            records[i].group_id = gid
        if len(classes) > 1:
            for i in indices:
                records[i].audit_status = "excluded"
                records[i].exclusion_reason = "conflicting classes within leakage-control group"
    return groups


def assign_splits(records: list[Record], groups: dict[str, list[int]], seed: int) -> None:
    """Assign whole groups—not individual images—to reproducible splits."""
    rng = random.Random(seed)
    ratios = {"train": 0.70, "valid": 0.15, "test": 0.15}
    for label in ("damaged", "intact"):
        candidates = []
        for gid, idx in groups.items():
            kept = [i for i in idx if records[i].audit_status != "excluded"]
            if kept and {records[i].binary_class for i in kept} == {label}:
                candidates.append((gid, kept))
        rng.shuffle(candidates)
        candidates.sort(key=lambda item: len(item[1]), reverse=True)
        total = sum(len(idx) for _, idx in candidates)
        targets = {s: ratios[s] * total for s in SPLITS}
        counts = Counter()
        for gid, indices in candidates:
            split = max(SPLITS, key=lambda s: (targets[s] - counts[s], -SPLITS.index(s)))
            for i in indices:
                records[i].new_split = split
                if label == "intact":
                    records[i].audit_status = "retained_label_unverified"
            counts[split] += len(indices)


def image_id(rec: Record) -> str:
    return hashlib.sha256(rec.relative.encode("utf-8")).hexdigest()[:16]


def write_csvs(records: list[Record], near_pairs: list[tuple[int, int, int]], output: Path) -> None:
    manifests = output / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    fields = ["image_id", "original_path", "source_split", "original_class", "binary_class", "group_id", "new_split", "duplicate_status", "audit_status", "exclusion_reason"]
    with (manifests / "dataset_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rec in records:
            writer.writerow({"image_id": image_id(rec), "original_path": rec.relative, "source_split": rec.source_split, "original_class": rec.original_class, "binary_class": rec.binary_class, "group_id": rec.group_id, "new_split": rec.new_split, "duplicate_status": rec.duplicate_status, "audit_status": rec.audit_status, "exclusion_reason": rec.exclusion_reason})

    with (manifests / "near_duplicate_pairs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_id_a", "image_id_b", "phash_hamming_distance", "class_a", "class_b", "group_id"])
        for i, j, distance in near_pairs:
            writer.writerow([image_id(records[i]), image_id(records[j]), distance, records[i].binary_class, records[j].binary_class, records[i].group_id])

    with (manifests / "manual_review_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["image_id", "original_path", "binary_class", "new_split", "review_status", "visual_flags", "review_notes"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rec in records:
            if rec.binary_class == "intact" and rec.audit_status != "excluded":
                iid = image_id(rec)
                flag = VISUALLY_FLAGGED.get(iid, "")
                writer.writerow({"image_id": iid, "original_path": rec.relative, "binary_class": rec.binary_class, "new_split": rec.new_split, "review_status": "flagged_manual_review" if flag else "pending_manual_review", "visual_flags": flag, "review_notes": "Visually flagged in contact-sheet review" if flag else "Check dents/compression; holes; tears; wet damage; crushed corners; open box; ambiguity"})


def copy_dataset(records: list[Record], output: Path) -> None:
    for split in SPLITS:
        for label in ("damaged", "intact"):
            (output / split / label).mkdir(parents=True, exist_ok=True)
    for rec in records:
        if rec.audit_status == "excluded" or not rec.new_split:
            continue
        name = f"{image_id(rec)}{rec.path.suffix.lower()}"
        shutil.copy2(rec.path, output / rec.new_split / rec.binary_class / name)


def contact_sheets(records: list[Record], output: Path, per_sheet: int = 100) -> int:
    review_dir = output / "manifests" / "intact_review_sheets"
    review_dir.mkdir(parents=True, exist_ok=True)
    intact = [r for r in records if r.binary_class == "intact" and r.audit_status != "excluded"]
    thumb = (120, 100)
    cols, rows = 10, 10
    count = 0
    for offset in range(0, len(intact), per_sheet):
        batch = intact[offset : offset + per_sheet]
        sheet = Image.new("RGB", (cols * thumb[0], rows * 125), "white")
        draw = ImageDraw.Draw(sheet)
        for n, rec in enumerate(batch):
            with Image.open(rec.path) as im:
                im = ImageOps.fit(im.convert("RGB"), thumb, method=Image.Resampling.LANCZOS)
            x, y = (n % cols) * thumb[0], (n // cols) * 125
            sheet.paste(im, (x, y))
            draw.text((x + 2, y + 102), image_id(rec), fill="black")
        count += 1
        sheet.save(review_dir / f"intact_review_{count:02d}.jpg", quality=90)
    return count


def write_readme_and_summary(records: list[Record], unexpected: list[str], near_pairs: list[tuple[int, int, int]], output: Path, sheets: int) -> dict:
    retained = [r for r in records if r.audit_status != "excluded"]
    summary = {
        "source_images": len(records),
        "unexpected_files": unexpected,
        "unreadable": sum(not r.readable for r in records),
        "excluded": len(records) - len(retained),
        "retained": len(retained),
        "exact_duplicate_files": sum(r.duplicate_status.startswith("exact_duplicate") for r in records),
        "augmentation_sibling_files": sum(r.duplicate_status == "roboflow_augmentation_sibling" for r in records),
        "near_duplicate_pairs": len(near_pairs),
        "groups": len({r.group_id for r in retained}),
        "split_counts": {s: dict(Counter(r.binary_class for r in retained if r.new_split == s)) for s in SPLITS},
        "source_counts": {s: dict(Counter(r.binary_class for r in records if r.source_split == s)) for s in SPLITS},
        "review_sheets": sheets,
    }
    (output / "manifests" / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "README.md").write_text(
        "# Parcel Binary V1\n\n"
        "Derived image-level damaged/intact dataset for Phase 7. Raw files are unchanged.\n\n"
        "The split is deterministic (seed 42) and group-stratified. Roboflow rotation siblings "
        "identified by the pre-`.rf.<hash>` filename, exact hashes, and conservative pHash matches "
        "remain in one split. Derived filenames are stable manifest IDs.\n\n"
        "The intact class comes from source folder labels and is not visually certified. Review "
        "`manifests/manual_review_manifest.csv` and `manifests/intact_review_sheets/`. HOG+SVM is "
        "an image classifier only and does not localize damage.\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=Path("datasets/raw/Damaged_Box_Detection"))
    parser.add_argument("--output", type=Path, default=Path("datasets/processed/parcel_binary_v1"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    # Reproducible rebuild: remove only this script's known derived artifacts.
    for name in (*SPLITS, "manifests"):
        target = args.output / name
        if target.exists():
            shutil.rmtree(target)
    # Each function below represents one visible dataset-preparation stage. No
    # model training occurs, and the raw input tree is never modified.
    records, unexpected = inventory(args.raw)
    uf, near_pairs = audit(records)
    groups = finalize_groups(records, uf)
    assign_splits(records, groups, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csvs(records, near_pairs, args.output)
    copy_dataset(records, args.output)
    sheets = contact_sheets(records, args.output)
    summary = write_readme_and_summary(records, unexpected, near_pairs, args.output, sheets)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


# ============================================================
# What you should understand after reading this script
# ============================================================
#
# 1. Why dataset leakage can produce misleading validation/test performance.
# 2. The difference between exact hashes and perceptual hashes.
# 3. Why related images are split as groups rather than individual files.
# 4. How a manifest preserves inclusion, exclusion, and split decisions.
# 5. Why human review is still needed after automated integrity checks.
