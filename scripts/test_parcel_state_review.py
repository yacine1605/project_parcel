"""CPU-only tests for parcel-state review preparation and folder assignment."""

import tempfile
import sys
import unittest
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prototype.parcel_state_review import assign_candidate, build_review_candidates, load_manifest


class ParcelStateReviewTests(unittest.TestCase):
    def test_prepare_and_assign(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, review, export = root / "raw", root / "review", root / "export"
            for split in ("train", "valid"):
                (source / split / "images").mkdir(parents=True)
                (source / split / "labels").mkdir(parents=True)
                Image.new("RGB", (100, 80), "white").save(source / split / "images" / f"{split}.jpg")
            (source / "train" / "labels" / "train.txt").write_text(
                "2 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8\n", encoding="utf-8"
            )
            (source / "valid" / "labels" / "valid.txt").write_text(
                "1 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n", encoding="utf-8"
            )

            manifest = build_review_candidates(source, review, padding=0)
            rows = load_manifest(manifest)
            self.assertEqual([row["suggested_label"] for row in rows], ["open_box", "exclude"])
            with Image.open(review / rows[0]["crop_path"]) as crop:
                self.assertEqual(crop.size, (60, 48))

            updated = assign_candidate(rows[0]["candidate_id"], "closed_box", manifest, export)
            self.assertEqual(updated["decision"], "closed_box")
            self.assertTrue((export / "train" / "closed_box" / Path(rows[0]["crop_path"]).name).is_file())

            assign_candidate(rows[0]["candidate_id"], "open_box", manifest, export)
            self.assertFalse((export / "train" / "closed_box" / Path(rows[0]["crop_path"]).name).exists())
            self.assertTrue((export / "train" / "open_box" / Path(rows[0]["crop_path"]).name).is_file())


if __name__ == "__main__":
    unittest.main()
