"""Create parcel_damage_v3 from an approved compressed-class review.

The script refuses to mutate V2 and defaults to a validation-only dry run.
Use --apply only after every checklist row has an approved decision.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "datasets" / "processed" / "parcel_damage_v2"
TARGET = ROOT / "datasets" / "processed" / "parcel_damage_v3"
CHECKLIST = ROOT / "reports" / "compressed_ontology_review" / "compressed_review_checklist.csv"
CHANGELOG = ROOT / "reports" / "dataset_v3_changes.json"
DECISIONS = {
    "keep_compressed",
    "relabel_hole",
    "relabel_minor_damage",
    "relabel_wet",
    "exclude_unverifiable",
}
REMAP = {"relabel_hole": 2, "relabel_minor_damage": 0, "relabel_wet": 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Create V3 after validation passes")
    return parser.parse_args()


def load_review() -> list[dict[str, str]]:
    with CHECKLIST.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate(rows: list[dict[str, str]]) -> tuple[Counter, list[str]]:
    counts = Counter((row.get("review_decision") or "").strip() for row in rows)
    errors = []
    incomplete = counts[""] + counts["needs_domain_review"]
    invalid = sorted(decision for decision in counts if decision and decision not in DECISIONS and decision != "needs_domain_review")
    if incomplete:
        errors.append(f"{incomplete} rows are incomplete or still need domain review")
    if invalid:
        errors.append(f"invalid decisions: {', '.join(invalid)}")
    return counts, errors


def rewrite_compressed_rows(label_path: Path, decision: str) -> tuple[int, int]:
    lines = label_path.read_text(encoding="utf-8").splitlines()
    output = []
    changed = 0
    removed = 0
    for line in lines:
        fields = line.split()
        if not fields or int(fields[0]) != 1:
            output.append(line)
            continue
        if decision == "exclude_unverifiable":
            removed += 1
            continue
        if decision in REMAP:
            fields[0] = str(REMAP[decision])
            line = " ".join(fields)
            changed += 1
        output.append(line)
    label_path.write_text("\n".join(output) + ("\n" if output else ""), encoding="utf-8")
    return changed, removed


def main() -> int:
    args = parse_args()
    rows = load_review()
    counts, errors = validate(rows)
    print(f"Checklist rows: {len(rows)}")
    for decision, count in sorted(counts.items()):
        print(f"  {decision or '<blank>'}: {count}")
    if errors:
        for error in errors:
            print(f"BLOCKED: {error}")
        return 2
    if not args.apply:
        print("READY: checklist is complete; rerun with --apply to create V3")
        return 0
    if TARGET.exists():
        raise FileExistsError(f"Refusing to overwrite existing target: {TARGET}")

    shutil.copytree(SOURCE, TARGET)
    changes = []
    for row in rows:
        decision = row["review_decision"].strip()
        if decision == "keep_compressed":
            continue
        source_label = ROOT / row["label"]
        target_label = TARGET / source_label.relative_to(SOURCE)
        changed, removed = rewrite_compressed_rows(target_label, decision)
        changes.append({
            "label": target_label.relative_to(ROOT).as_posix(),
            "decision": decision,
            "boxes_relabelled": changed,
            "boxes_removed": removed,
            "notes": row.get("review_notes", ""),
        })

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": SOURCE.relative_to(ROOT).as_posix(),
        "target": TARGET.relative_to(ROOT).as_posix(),
        "decision_counts": dict(counts),
        "files_changed": len(changes),
        "changes": changes,
    }
    CHANGELOG.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Created {TARGET}")
    print(f"Wrote {CHANGELOG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
