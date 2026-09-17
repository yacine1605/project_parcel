"""CPU-only branch tests for the multi-parcel inference pipeline."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prototype.inference_parcel_detector import crop_parcel
from prototype.inference_parcel_state import load_parcel_state_classifier
from prototype.inspection_pipeline import map_bbox_to_original, run_pipeline


def state(box, state="closed_box", confidence=0.95):
    return {"parcel_bbox": box, "parcel_confidence": 0.97,
            "mock_state": state, "mock_state_confidence": confidence}


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (100, 100), "white")
        self.models = {
            "parcel_detector": object(), "parcel_state_classifier": object(), "classifier": object(),
            "damage_detector": object(), "device": object(), "classifier_threshold": 0.434584259987,
        }
        self.classifier_calls = 0
        self.damage_calls = 0

    def detector(self, detections):
        return lambda *_: {"detections": detections, "detection_count": len(detections), "inference_ms": 1.0}

    def classifier(self, label="intact", probability=0.1):
        def predict(*_):
            self.classifier_calls += 1
            return {"label": label, "damaged_probability": probability,
                    "threshold": 0.434584259987, "inference_ms": 2.0}
        return predict

    def damage_detector(self, detections=None):
        def predict(*_):
            self.damage_calls += 1
            return {"detections": detections or [], "detection_count": len(detections or []), "inference_ms": 3.0}
        return predict

    def run_case(self, detections, classifier=None, damage=None):
        states = iter(detections)
        def state_classifier(*_):
            item = next(states)
            return {"parcel_state": item["mock_state"],
                    "parcel_state_confidence": item["mock_state_confidence"],
                    "warning": "parcel_localized_but_state_not_classified" if item["mock_state"] == "unknown" else None,
                    "inference_ms": 1.5}
        return run_pipeline(
            self.image, self.models, parcel_detector_fn=self.detector(detections),
            state_classifier_fn=state_classifier,
            classifier_fn=classifier or self.classifier(), damage_detector_fn=damage or self.damage_detector(),
        )[0]

    def test_no_parcel_detected(self):
        result = self.run_case([])
        self.assertEqual(result["status"], "no_parcel_detected")
        self.assertEqual(result["parcels"], [])
        self.assertEqual((self.classifier_calls, self.damage_calls), (0, 0))
        self.assertFalse(result["classifier_executed"])
        self.assertFalse(result["damage_detector_executed"])

    def test_one_open_parcel_skips_both_damage_models(self):
        result = self.run_case([state([10, 10, 90, 90], "open_box")])
        self.assertEqual(result["parcels"][0]["final_status"], "open_box")
        self.assertEqual((self.classifier_calls, self.damage_calls), (0, 0))

    def test_one_closed_intact_parcel_skips_damage_yolo(self):
        result = self.run_case([state([10, 10, 90, 90])])
        self.assertEqual(result["parcels"][0]["final_status"], "closed_intact")
        self.assertEqual((self.classifier_calls, self.damage_calls), (1, 0))
        self.assertTrue(result["classifier_executed"])
        self.assertFalse(result["damage_detector_executed"])

    def test_unknown_state_skips_both_damage_models(self):
        result = self.run_case([state([10, 10, 90, 90], "unknown", None)])
        parcel = result["parcels"][0]
        self.assertEqual(parcel["final_status"], "state_unknown")
        self.assertEqual(parcel["warning"], "parcel_localized_but_state_not_classified")
        self.assertEqual((self.classifier_calls, self.damage_calls), (0, 0))

    def test_one_closed_damaged_parcel_maps_damage(self):
        damage = [{"class_name": "hole", "confidence": 0.8,
                   "bounding_box_xyxy": [1, 2, 3, 4]}]
        result = self.run_case([state([10, 10, 90, 90])], self.classifier("damaged", 0.9), self.damage_detector(damage))
        parcel = result["parcels"][0]
        self.assertEqual(parcel["final_status"], "closed_damaged")
        self.assertEqual(parcel["damages"][0]["bbox"], [7, 8, 9, 10])
        self.assertEqual((self.classifier_calls, self.damage_calls), (1, 1))
        self.assertTrue(result["classifier_executed"])
        self.assertTrue(result["damage_detector_executed"])

    def test_multiple_parcels_are_processed_independently(self):
        predictions = iter([
            {"label": "intact", "damaged_probability": 0.1, "inference_ms": 1.0},
            {"label": "damaged", "damaged_probability": 0.9, "inference_ms": 1.0},
        ])
        def classifier(*_):
            self.classifier_calls += 1
            return next(predictions)
        result = self.run_case([
            state([0, 0, 30, 30]), state([35, 0, 65, 30], "open_box"), state([70, 0, 99, 30])
        ], classifier, self.damage_detector())
        self.assertEqual([p["final_status"] for p in result["parcels"]],
                         ["closed_intact", "open_box", "closed_damaged"])
        self.assertEqual((self.classifier_calls, self.damage_calls), (2, 1))

    def test_coordinate_conversion(self):
        self.assertEqual(map_bbox_to_original([2, 3, 8, 9], [100, 200, 300, 400]), [102, 203, 108, 209])

    def test_invalid_crop_is_reported(self):
        result = self.run_case([state([200, 200, 210, 210])])
        self.assertEqual(result["parcels"][0]["final_status"], "invalid_crop")
        self.assertEqual((self.classifier_calls, self.damage_calls), (0, 0))

    def test_damaged_with_no_regions_keeps_damaged_warning(self):
        result = self.run_case([state([10, 10, 90, 90])], self.classifier("damaged", 0.91), self.damage_detector())
        parcel = result["parcels"][0]
        self.assertEqual(parcel["damage_status"], "damaged")
        self.assertEqual(parcel["damages"], [])
        self.assertEqual(parcel["warning"], "classified_damaged_but_no_damage_region_detected")

    def test_crop_clamps_to_image(self):
        crop, bbox = crop_parcel(self.image, [-10, -10, 110, 110])
        self.assertEqual(bbox, [0, 0, 100, 100])
        self.assertEqual(crop.size, (100, 100))

    def test_missing_state_weights_fail_clearly(self):
        missing = ROOT / "models" / "parcel_state_classifier" / "missing_test_checkpoint.pt"
        with self.assertRaisesRegex(FileNotFoundError, "Open/closed"):
            load_parcel_state_classifier(missing)


if __name__ == "__main__":
    unittest.main()
