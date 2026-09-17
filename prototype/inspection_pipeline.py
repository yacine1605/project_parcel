"""Multi-parcel state -> classification -> conditional damage pipeline."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

try:
    from .inference_classifier import load_frozen_classifier, predict_damage_classification, select_device
    from .inference_parcel_detector import crop_parcel, detect_parcels, load_parcel_detector
    from .inference_parcel_state import classify_parcel_state, load_parcel_state_classifier
    from .inference_yolo import load_frozen_yolo, predict_damage_regions
except ImportError:
    from inference_classifier import load_frozen_classifier, predict_damage_classification, select_device
    from inference_parcel_detector import crop_parcel, detect_parcels, load_parcel_detector
    from inference_parcel_state import classify_parcel_state, load_parcel_state_classifier
    from inference_yolo import load_frozen_yolo, predict_damage_regions


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LOGGER = logging.getLogger(__name__)


def create_image_id(image: Image.Image) -> str:
    rgb = image.convert("RGB")
    digest = hashlib.sha256(f"{rgb.width}x{rgb.height}".encode("ascii") + rgb.tobytes())
    return digest.hexdigest()[:16]


def map_bbox_to_original(bbox: list[float], crop_bbox: list[int]) -> list[float]:
    """Translate a crop-relative xyxy box to original-image coordinates."""
    if len(bbox) != 4 or len(crop_bbox) != 4:
        raise ValueError("Both bounding boxes must contain four coordinates.")
    return [bbox[0] + crop_bbox[0], bbox[1] + crop_bbox[1],
            bbox[2] + crop_bbox[0], bbox[3] + crop_bbox[1]]


def classify_damage_status(parcel_crop: Image.Image, loaded_models: dict) -> dict:
    return predict_damage_classification(
        image=parcel_crop, model=loaded_models["classifier"],
        preprocessing=loaded_models["preprocessing"], device=loaded_models["device"],
        frozen_threshold=loaded_models["classifier_threshold"],
    )


def detect_damage(parcel_crop: Image.Image, loaded_models: dict) -> dict:
    return predict_damage_regions(
        image=parcel_crop, model=loaded_models["damage_detector"], device=loaded_models["device"]
    )


def process_parcel(
    image: Image.Image,
    detection: dict,
    parcel_index: int,
    loaded_models: dict,
    state_classifier_fn=classify_parcel_state,
    classifier_fn=classify_damage_status,
    damage_detector_fn=detect_damage,
) -> tuple[dict, Image.Image | None]:
    """Process one parcel independently, enforcing the open/closed gates."""
    base = {
        "parcel_index": parcel_index,
        "parcel_bbox": detection["parcel_bbox"],
        "parcel_confidence": detection["parcel_confidence"],
        "parcel_state": "unknown",
        "parcel_state_confidence": None,
        "open_box_probability": None,
        "open_box_threshold": None,
        "crop_bbox": None,
        "final_status": None,
        "damage_status": None,
        "damage_confidence": None,
        "classifier_executed": False,
        "damage_detector_executed": False,
        "damages": [],
        "warning": None,
        "timing_ms": {"state_classifier": 0.0, "classifier": 0.0, "damage_detector": 0.0},
    }
    crop, crop_bbox = crop_parcel(image, detection["parcel_bbox"])
    base["crop_bbox"] = crop_bbox
    state_result = state_classifier_fn(crop, loaded_models)
    state = state_result["parcel_state"]
    base["parcel_state"] = state
    base["parcel_state_confidence"] = state_result["parcel_state_confidence"]
    base["open_box_probability"] = state_result.get("open_box_probability")
    base["open_box_threshold"] = state_result.get("open_box_threshold")
    base["timing_ms"]["state_classifier"] = state_result.get("inference_ms", 0.0)
    if state_result.get("warning"):
        base["warning"] = state_result["warning"]
    LOGGER.info("Parcel %d\nstate: %s (%s)", parcel_index, state,
                f"{base['parcel_state_confidence']:.2f}" if base["parcel_state_confidence"] is not None else "no prediction")
    if state == "open_box":
        base["final_status"] = "open_box"
        LOGGER.info("Skipping damage models.")
        return base, crop
    if state != "closed_box":
        base["final_status"] = "state_unknown"
        LOGGER.info("Skipping damage models because parcel state is unknown.")
        return base, crop

    classifier = classifier_fn(crop, loaded_models)
    base["classifier_executed"] = True
    base["damage_status"] = classifier["label"]
    base["damage_confidence"] = classifier["damaged_probability"]
    base["classifier"] = classifier
    base["timing_ms"]["classifier"] = classifier.get("inference_ms", 0.0)
    LOGGER.info("MobileNet: %s (%.2f)", classifier["label"], classifier["damaged_probability"])
    if classifier["label"] == "intact":
        base["final_status"] = "closed_intact"
        return base, crop

    damage_result = damage_detector_fn(crop, loaded_models)
    base["damage_detector_executed"] = True
    base["timing_ms"]["damage_detector"] = damage_result.get("inference_ms", 0.0)
    for item in damage_result["detections"]:
        mapped = map_bbox_to_original(item["bounding_box_xyxy"], crop_bbox)
        base["damages"].append({
            "class": item["class_name"], "confidence": item["confidence"], "bbox": mapped
        })
        LOGGER.info("damage: %s %.2f", item["class_name"], item["confidence"])
    base["final_status"] = "closed_damaged"
    if not base["damages"]:
        base["warning"] = "classified_damaged_but_no_damage_region_detected"
    return base, crop


def _legacy_summary(parcels: list[dict], threshold: float) -> dict:
    """Return compatibility fields without hiding whether a model was skipped.

    The original SQLite schema requires an intact/damaged label and a numeric
    probability. Those compatibility values remain for old readers, while the
    explicit execution flags tell new readers whether they represent a real
    model result. The per-parcel list remains the authoritative evidence.
    """
    classified = [p for p in parcels if p["damage_status"] is not None]
    damaged = [p for p in classified if p["damage_status"] == "damaged"]
    damages = [d for parcel in parcels for d in parcel["damages"]]
    classifier_executed = bool(classified)
    damage_detector_executed = any(p.get("damage_detector_executed", False) for p in parcels)
    label = "damaged" if damaged else "intact"
    probability = max((p["damage_confidence"] for p in classified), default=0.0)
    yolo_detections = [{"class_name": d["class"], "confidence": d["confidence"],
                        "bounding_box_xyxy": d["bbox"]} for d in damages]
    accept = bool(parcels) and all(p["final_status"] == "closed_intact" for p in parcels)
    return {
        "classifier_label": label, "classifier_probability": probability,
        "classifier_executed": classifier_executed,
        "damage_detector_executed": damage_detector_executed,
        "classifier_threshold": threshold, "yolo_detection_count": len(damages),
        "yolo_damage_types": [d["class"] for d in damages],
        "yolo_confidences": [d["confidence"] for d in damages],
        "classifier": {"label": label, "damaged_probability": probability, "threshold": threshold},
        "yolo": {"detections": yolo_detections, "detection_count": len(damages)},
        "prototype_decision": "ACCEPT" if accept else "REVIEW",
        "prototype_decision_reason": "Every detected parcel is closed and intact." if accept else "One or more parcels require review.",
    }


def draw_results(image: Image.Image, result: dict) -> Image.Image:
    annotated = image.convert("RGB").copy()
    draw, font = ImageDraw.Draw(annotated), ImageFont.load_default()
    damage_colors = {"minor_damage": "#ff9800", "compressed": "#e91e63", "hole": "#f44336", "wet": "#2196f3"}
    for parcel in result["parcels"]:
        box = parcel["parcel_bbox"]
        state_colors = {
            "closed_box": "#00e676",
            "open_box": "#ff5252",
            "unknown": "#ffc107",
        }
        state_color = state_colors.get(parcel["parcel_state"], "#ffc107")
        draw.rectangle(box, outline=state_color, width=max(2, annotated.width // 250))
        state_confidence = parcel["parcel_state_confidence"]
        label = parcel["parcel_state"] + (f" {state_confidence:.2f}" if state_confidence is not None else "")
        draw.text((box[0] + 3, max(0, box[1] + 3)), label, fill=state_color, font=font)
        for damage in parcel["damages"]:
            color = damage_colors.get(damage["class"], "#ffeb3b")
            draw.rectangle(damage["bbox"], outline=color, width=max(2, annotated.width // 300))
            draw.text((damage["bbox"][0] + 3, damage["bbox"][1] + 3),
                      f"{damage['class']} {damage['confidence']:.2f}", fill=color, font=font)
    return annotated


def _save_debug_images(debug_dir: Path, original: Image.Image, crops: list[Image.Image | None],
                       state_image: Image.Image, final_image: Image.Image) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    original.save(debug_dir / "original.jpg")
    for index, crop in enumerate(crops):
        if crop is not None:
            crop.save(debug_dir / f"parcel_{index}_crop.jpg")
    state_image.save(debug_dir / "parcel_state_predictions.jpg")
    final_image.save(debug_dir / "final_predictions.jpg")


def run_pipeline(
    image: Image.Image,
    loaded_models: dict,
    debug: bool = False,
    debug_dir: Path | None = None,
    parcel_detector_fn=detect_parcels,
    state_classifier_fn=classify_parcel_state,
    classifier_fn=classify_damage_status,
    damage_detector_fn=detect_damage,
) -> tuple[dict, Image.Image]:
    started = time.perf_counter()
    rgb = image.convert("RGB")
    localization_result = parcel_detector_fn(rgb, loaded_models["parcel_detector"], loaded_models["device"])
    parcels, crops = [], []
    for index, detection in enumerate(localization_result["detections"]):
        try:
            parcel, crop = process_parcel(rgb, detection, index, loaded_models, state_classifier_fn,
                                          classifier_fn, damage_detector_fn)
        except ValueError as error:
            parcel = {
                "parcel_index": index, "parcel_bbox": detection.get("parcel_bbox"),
                "parcel_confidence": detection.get("parcel_confidence"),
                "parcel_state": "unknown", "parcel_state_confidence": None,
                "open_box_probability": None, "open_box_threshold": None,
                "crop_bbox": None, "final_status": "invalid_crop", "damage_status": None,
                "damage_confidence": None, "classifier_executed": False,
                "damage_detector_executed": False, "damages": [], "warning": str(error),
                "timing_ms": {"state_classifier": 0.0, "classifier": 0.0, "damage_detector": 0.0},
            }
            crop = None
        parcels.append(parcel)
        crops.append(crop)
    status = "ok" if parcels else "no_parcel_detected"
    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "image_id": create_image_id(rgb),
        "status": status, "parcels": parcels,
        "parcel_detector": localization_result,
        "timing_ms": {
            "parcel_detection": localization_result.get("inference_ms", 0.0),
            "parcel_state_classification": sum(p["timing_ms"]["state_classifier"] for p in parcels),
            "classifier_forward": sum(p["timing_ms"]["classifier"] for p in parcels),
            "yolo_inference": sum(p["timing_ms"]["damage_detector"] for p in parcels),
            "total_pipeline": 0.0,
        },
        "policy_status": "prototype_only_not_validated_for_production",
    }
    result.update(_legacy_summary(parcels, loaded_models["classifier_threshold"]))
    result["timing_ms"]["total_pipeline"] = (time.perf_counter() - started) * 1000
    state_only = draw_results(rgb, {"parcels": [{**p, "damages": []} for p in parcels]})
    annotated = draw_results(rgb, result)
    if debug:
        _save_debug_images(Path(debug_dir or PROJECT_ROOT / "debug"), rgb, crops, state_only, annotated)
    return result, annotated


def inspect_parcel_image(image: Image.Image, loaded_models: dict, **kwargs) -> tuple[dict, Image.Image]:
    return run_pipeline(image, loaded_models, **kwargs)


def load_prototype_models():
    """Load all four models once for reuse by UI, CLI, or webcam."""
    device = select_device()
    parcel_detector = load_parcel_detector()
    parcel_state_classifier = load_parcel_state_classifier()
    classifier, preprocessing, threshold, contract = load_frozen_classifier(device)
    damage_detector = load_frozen_yolo()
    return {
        "device": device, "classifier": classifier, "preprocessing": preprocessing,
        "classifier_threshold": threshold, "classifier_contract": contract,
        "parcel_detector": parcel_detector,
        "parcel_state_classifier": parcel_state_classifier,
        "damage_detector": damage_detector,
        "yolo": damage_detector,
    }


def open_supported_image(path: Path) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(f"Image does not exist: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported image type '{path.suffix}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}")
    try:
        with Image.open(path) as image:
            return image.convert("RGB")
    except Exception as error:
        raise ValueError(f"Image could not be opened: {path}. {error}") from error


def save_evidence_json(evidence: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")


def save_completed_inspection(evidence: dict, image_path: str | Path,
                              annotated_image_path: str | Path | None,
                              package_id: str | None = None, database_path: Path | None = None) -> int:
    try:
        from .database import DEFAULT_DATABASE_PATH, save_inspection
    except ImportError:
        from database import DEFAULT_DATABASE_PATH, save_inspection
    return save_inspection(evidence, image_path, annotated_image_path, package_id,
                           Path(database_path) if database_path else DEFAULT_DATABASE_PATH)
