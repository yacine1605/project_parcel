"""Create manually reviewable parcel-state crops from the raw YOLO dataset."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prototype.parcel_state_review import build_review_candidates


if __name__ == "__main__":
    path = build_review_candidates()
    print(f"Review candidates created: {path}")
