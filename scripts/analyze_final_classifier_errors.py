"""
# ============================================================
# What are we doing in this script?
# ============================================================
#
# Metrics tell us HOW OFTEN the model is wrong.
# Error analysis helps us understand WHY the model is wrong.
#
# This script reads the predictions permanently saved in Phase 11. It does
# not load the neural network, run inference, train, or select a threshold.
# It creates four outcome tables, probability plots, and image grids that
# make recurring visual failure patterns easier to inspect.
#
# We are NOT changing the model, labels, preprocessing, or threshold. Using
# final-test errors to understand limitations is valid. Changing the system
# and testing it again on the same test set would contaminate the evaluation.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

# Error-analysis figures are saved to files, so no interactive desktop window
# is needed. ``Agg`` makes plotting reproducible on machines without a GUI.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps


# ------------------------------------------------------------
# 1. Fixed input and output paths
# ------------------------------------------------------------
# The prediction CSV is the permanent Phase 11 audit trail. Reading it avoids
# any possibility that a new inference run produces different predictions.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "datasets" / "processed" / "parcel_binary_v2"
MODEL_DIR = PROJECT_ROOT / "models" / "selected_transfer_model"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"

PREDICTIONS_PATH = MODEL_DIR / "final_test_predictions.csv"
METRICS_PATH = MODEL_DIR / "final_test_metrics.json"
ANALYSIS_PATH = MODEL_DIR / "error_analysis.csv"
TAXONOMY_PATH = MODEL_DIR / "error_taxonomy.json"

EXPECTED_ROWS = 592
EXPECTED_THRESHOLD = 0.43458425998687744
EXPECTED_CONFUSION = {"true_negative": 157, "false_positive": 43,
                      "false_negative": 42, "true_positive": 350}


def sha256(path: Path) -> str:
    """Return a file hash so the analysis records its exact Phase 11 input."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_and_verify_predictions() -> pd.DataFrame:
    """Load saved predictions and fail if their frozen facts have changed."""
    frame = pd.read_csv(PREDICTIONS_PATH)
    required = {
        "image_id", "relative_path", "group_id", "true_label",
        "damaged_probability", "frozen_threshold", "predicted_label",
        "correct", "subgroup", "open_box",
    }
    if set(frame.columns) != required:
        raise RuntimeError(f"Unexpected prediction columns: {list(frame.columns)}")
    if len(frame) != EXPECTED_ROWS or frame["image_id"].nunique() != EXPECTED_ROWS:
        raise RuntimeError("The Phase 11 prediction row count or IDs changed.")
    thresholds = frame["frozen_threshold"].unique()
    if len(thresholds) != 1 or not np.isclose(thresholds[0], EXPECTED_THRESHOLD, atol=1e-15):
        raise RuntimeError("The saved threshold does not match the frozen threshold.")

    # The margin measures distance from the frozen decision boundary:
    # margin = damaged probability - threshold.
    # A false negative at -0.0146 was close to becoming damaged; one at -0.35
    # was a much more confident miss. Positive margins produce damaged labels.
    frame["prediction_margin"] = (
        frame["damaged_probability"] - frame["frozen_threshold"]
    )

    conditions = [
        (frame.true_label == "intact") & (frame.predicted_label == "intact"),
        (frame.true_label == "intact") & (frame.predicted_label == "damaged"),
        (frame.true_label == "damaged") & (frame.predicted_label == "intact"),
        (frame.true_label == "damaged") & (frame.predicted_label == "damaged"),
    ]
    frame["outcome"] = np.select(
        conditions,
        ["true_negative", "false_positive", "false_negative", "true_positive"],
        default="invalid",
    )
    counts = frame.outcome.value_counts().to_dict()
    if counts != EXPECTED_CONFUSION:
        raise RuntimeError(f"Frozen confusion matrix changed: {counts}")

    paths = frame.relative_path.map(lambda value: DATASET_ROOT / value)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"Referenced test images are missing: {missing[:3]}")
    frame["absolute_image_path"] = paths.map(str)
    return frame


def annotate_failure_modes(frame: pd.DataFrame) -> pd.DataFrame:
    """Add conservative, human-reviewed visual tags to error rows.

    Related crops/rotations share a frozen ``group_id``. Assigning the same
    group-level interpretation prevents us from pretending that near-duplicate
    images are independent visual findings. Multiple tags are allowed because,
    for example, a label and a perspective edge may both contribute.
    """
    frame = frame.copy()
    frame["failure_tags"] = ""

    # Every missed open box has the semantic open/closed ambiguity. Extra tags
    # below describe visible context, not a newly invented numerical feature.
    open_miss = (frame.outcome == "false_negative") & frame.open_box
    frame.loc[open_miss, "failure_tags"] = "open_box_semantic_ambiguity"
    open_context = {
        "grp_00583": ["partial_parcel_visibility", "perspective_or_orientation"],
        "grp_00599": ["perspective_or_orientation", "shadow_or_lighting"],
        "grp_00603": ["background_or_clutter", "perspective_or_orientation"],
        "grp_00667": ["partial_parcel_visibility", "perspective_or_orientation"],
        "grp_00683": ["seam_or_corner_confusion", "perspective_or_orientation"],
        "grp_00687": ["perspective_or_orientation"],
        "grp_00735": ["shadow_or_lighting", "seam_or_corner_confusion"],
        "grp_00770": ["partial_parcel_visibility", "perspective_or_orientation"],
        "grp_00783": ["partial_parcel_visibility", "perspective_or_orientation"],
        "grp_00838": ["background_or_clutter", "packaging_or_branding"],
        "grp_00854": ["background_or_clutter", "partial_parcel_visibility"],
        "grp_00868": ["background_or_clutter", "perspective_or_orientation"],
        "grp_00950": ["background_or_clutter", "perspective_or_orientation"],
        "grp_01082": ["background_or_clutter", "packaging_or_branding"],
        "grp_01110": ["perspective_or_orientation", "packaging_or_branding"],
    }
    for group_id, tags in open_context.items():
        mask = open_miss & (frame.group_id == group_id)
        frame.loc[mask, "failure_tags"] += ";" + ";".join(tags)

    # The seven ordinary-damage misses were reviewed separately. Some visual
    # evidence is ambiguous, so ``other_or_uncertain`` is retained where needed.
    ordinary_context = {
        "grp_00626": ["subtle_or_localized_damage", "packaging_or_branding", "background_or_clutter"],
        "grp_00682": ["subtle_or_localized_damage", "background_or_clutter"],
        "grp_00924": ["subtle_or_localized_damage", "packaging_or_branding", "other_or_uncertain"],
        "grp_00976": ["subtle_or_localized_damage", "partial_parcel_visibility"],
    }
    for group_id, tags in ordinary_context.items():
        mask = (frame.outcome == "false_negative") & (frame.group_id == group_id)
        frame.loc[mask, "failure_tags"] = ";".join(tags)

    # False-positive tags record intact structures that visually resemble cues
    # for damage. They are qualitative observations, not causal proof.
    fp_context = {
        "grp_00047": ["tape_or_label_confusion", "shadow_or_lighting"],
        "grp_00049": ["packaging_or_branding", "seam_or_corner_confusion", "perspective_or_orientation"],
        "grp_00053": ["tape_or_label_confusion", "perspective_or_orientation"],
        "grp_00055": ["seam_or_corner_confusion", "perspective_or_orientation"],
        "grp_00065": ["tape_or_label_confusion", "background_or_clutter"],
        "grp_00067": ["seam_or_corner_confusion", "perspective_or_orientation"],
        "grp_00071": ["packaging_or_branding", "perspective_or_orientation"],
        "grp_00514": ["packaging_or_branding", "seam_or_corner_confusion"],
        "grp_00515": ["tape_or_label_confusion", "packaging_or_branding"],
        "grp_00545": ["tape_or_label_confusion", "shadow_or_lighting"],
        "grp_00582": ["tape_or_label_confusion", "packaging_or_branding"],
        "grp_00585": ["background_or_clutter", "packaging_or_branding"],
        "grp_00623": ["seam_or_corner_confusion", "perspective_or_orientation"],
        "grp_00650": ["packaging_or_branding", "seam_or_corner_confusion"],
        "grp_00662": ["packaging_or_branding", "perspective_or_orientation"],
        "grp_00745": ["packaging_or_branding", "seam_or_corner_confusion"],
        "grp_00753": ["tape_or_label_confusion", "background_or_clutter"],
        "grp_00816": ["tape_or_label_confusion", "packaging_or_branding"],
        "grp_00830": ["seam_or_corner_confusion", "perspective_or_orientation"],
        "grp_00870": ["background_or_clutter", "perspective_or_orientation"],
        "grp_00874": ["tape_or_label_confusion", "background_or_clutter"],
        "grp_00885": ["seam_or_corner_confusion", "perspective_or_orientation"],
        "grp_00915": ["tape_or_label_confusion", "background_or_clutter"],
        "grp_00942": ["background_or_clutter", "perspective_or_orientation"],
        "grp_01071": ["tape_or_label_confusion", "packaging_or_branding"],
        "grp_01118": ["tape_or_label_confusion", "packaging_or_branding"],
        "grp_01136": ["seam_or_corner_confusion", "perspective_or_orientation"],
        "grp_01162": ["seam_or_corner_confusion", "perspective_or_orientation"],
        "grp_01169": ["tape_or_label_confusion", "shadow_or_lighting"],
    }
    for group_id, tags in fp_context.items():
        mask = (frame.outcome == "false_positive") & (frame.group_id == group_id)
        frame.loc[mask, "failure_tags"] = ";".join(tags)

    errors = frame.outcome.isin(["false_negative", "false_positive"])
    if (frame.loc[errors, "failure_tags"] == "").any():
        raise RuntimeError("A reviewed error row has no taxonomy tag.")
    return frame


def save_outcome_tables(frame: pd.DataFrame) -> None:
    """Save one comprehensive table and four easy-to-filter outcome tables."""
    columns = [
        "image_id", "relative_path", "group_id", "true_label",
        "predicted_label", "damaged_probability", "frozen_threshold",
        "prediction_margin", "subgroup", "open_box", "outcome", "failure_tags",
    ]
    frame[columns].to_csv(ANALYSIS_PATH, index=False, float_format="%.12f")
    for outcome in EXPECTED_CONFUSION:
        output = MODEL_DIR / f"{outcome}s.csv"
        frame.loc[frame.outcome == outcome, columns].to_csv(
            output, index=False, float_format="%.12f"
        )


def make_image_grid(
    rows: pd.DataFrame,
    output_path: Path,
    title: str,
    columns: int = 6,
) -> None:
    """Create a readable grid with fixed Phase 11 probability and threshold."""
    thumb_w, thumb_h, label_h = 230, 170, 68
    grid_rows = int(np.ceil(len(rows) / columns))
    canvas = Image.new("RGB", (columns * thumb_w, 58 + grid_rows * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((12, 12), title, fill="black", font=font)
    draw.text((12, 32), f"Frozen threshold = {EXPECTED_THRESHOLD:.4f}; n = {len(rows)}", fill="black", font=font)

    for index, (_, sample) in enumerate(rows.iterrows()):
        col, row = index % columns, index // columns
        x, y = col * thumb_w, 58 + row * (thumb_h + label_h)
        with Image.open(sample.absolute_image_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail((thumb_w - 10, thumb_h - 10))
            tile = Image.new("RGB", (thumb_w - 4, thumb_h), "#dddddd")
            tile.paste(image, ((tile.width - image.width) // 2, (tile.height - image.height) // 2))
        canvas.paste(tile, (x + 2, y))
        lines = [
            f"ID {sample.image_id[:10]} | {sample.subgroup}",
            f"truth={sample.true_label} pred={sample.predicted_label}",
            f"p={sample.damaged_probability:.3f} margin={sample.prediction_margin:+.3f}",
        ]
        draw.multiline_text((x + 5, y + thumb_h + 3), "\n".join(lines), fill="black", font=font, spacing=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def plot_probability_distributions(frame: pd.DataFrame) -> None:
    """Compare score and margin distributions without moving the threshold."""
    colors = {
        "true_positive": "#2ca02c", "true_negative": "#1f77b4",
        "false_positive": "#ff7f0e", "false_negative": "#d62728",
    }
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    bins = np.linspace(0, 1, 21)
    for outcome, color in colors.items():
        values = frame.loc[frame.outcome == outcome, "damaged_probability"]
        axes[0, 0].hist(values, bins=bins, alpha=0.48, label=f"{outcome} (n={len(values)})", color=color)
    axes[0, 0].axvline(EXPECTED_THRESHOLD, color="black", linestyle="--", label="frozen threshold")
    axes[0, 0].set(title="Probability by outcome", xlabel="Damaged probability", ylabel="Samples")
    axes[0, 0].legend(fontsize=8)

    damaged = frame[frame.true_label == "damaged"]
    for subgroup, color in [("ordinary_damaged", "#9467bd"), ("open_box", "#8c564b")]:
        values = damaged.loc[damaged.subgroup == subgroup, "damaged_probability"]
        axes[0, 1].hist(values, bins=bins, alpha=0.55, label=f"{subgroup} (n={len(values)})", color=color)
    axes[0, 1].axvline(EXPECTED_THRESHOLD, color="black", linestyle="--")
    axes[0, 1].set(title="Damaged subgroups", xlabel="Damaged probability", ylabel="Samples")
    axes[0, 1].legend()

    errors = frame[frame.outcome.isin(["false_positive", "false_negative"])]
    axes[1, 0].hist(errors.prediction_margin, bins=20, color="#7f7f7f", edgecolor="white")
    axes[1, 0].axvline(0, color="black", linestyle="--")
    axes[1, 0].set(title="Error distance from frozen threshold", xlabel="Probability - threshold", ylabel="Errors")

    data = [frame.loc[frame.outcome == name, "damaged_probability"] for name in colors]
    axes[1, 1].boxplot(data, tick_labels=[name.replace("_", "\n") for name in colors], showfliers=True)
    axes[1, 1].axhline(EXPECTED_THRESHOLD, color="black", linestyle="--")
    axes[1, 1].set(title="Probability spread by outcome", ylabel="Damaged probability")
    fig.suptitle("Phase 12: frozen MobileNetV3-Large probability analysis")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "phase12_probability_distributions.png", dpi=160)
    plt.close(fig)


def write_taxonomy(frame: pd.DataFrame) -> None:
    """Write taxonomy definitions, reviewed assignments, and tag frequencies."""
    categories = {
        "open_box_semantic_ambiguity": "Opening is a structural/semantic state with little classic damage texture.",
        "subtle_or_localized_damage": "Visible damage is small, weak, or localized.",
        "partial_parcel_visibility": "Only part of the parcel is visible.",
        "perspective_or_orientation": "Viewpoint hides or imitates the relevant structure.",
        "tape_or_label_confusion": "Tape or labels resemble learned damage cues.",
        "seam_or_corner_confusion": "Normal seams, folds, or corners resemble damage geometry.",
        "background_or_clutter": "Surroundings compete with parcel evidence.",
        "shadow_or_lighting": "Illumination creates or hides damage-like contrast.",
        "packaging_or_branding": "Printing or unusual packaging changes visual appearance.",
        "other_or_uncertain": "Visual evidence is insufficient for a confident cause.",
    }
    errors = frame[frame.outcome.isin(["false_negative", "false_positive"])]
    tag_counts: dict[str, int] = {}
    for tags in errors.failure_tags:
        for tag in tags.split(";"):
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    assignments = {
        row.image_id: row.failure_tags.split(";")
        for row in errors.itertuples()
    }
    payload = {
        "analysis_type": "descriptive_only",
        "predictions_sha256": sha256(PREDICTIONS_PATH),
        "frozen_threshold": EXPECTED_THRESHOLD,
        "categories": categories,
        "assignment_policy": "Multiple tags allowed; uncertain cases are not forced into a category.",
        "important_caution": "Counts overlap because one image may have multiple visually supported tags; tags are descriptive observations, not proven causes.",
        "tag_counts_across_error_images": dict(sorted(tag_counts.items(), key=lambda item: (-item[1], item[0]))),
        "assignments_by_image_id": assignments,
        "aggregate_review": {
            "open_box_false_negatives": int(((frame.outcome == "false_negative") & frame.open_box).sum()),
            "ordinary_damage_false_negatives": int(((frame.outcome == "false_negative") & ~frame.open_box).sum()),
            "false_positives": int((frame.outcome == "false_positive").sum()),
        },
    }
    TAXONOMY_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def print_summary(frame: pd.DataFrame) -> None:
    """Print results in plain language rather than only raw objects."""
    fn = frame[frame.outcome == "false_negative"]
    fp = frame[frame.outcome == "false_positive"]
    near_band = 0.10
    print("\nPhase 12 descriptive artifacts complete.")
    print(f"False negatives: {len(fn)}; open-box misses: {int(fn.open_box.sum())}.")
    print(f"False positives: {len(fp)}.")
    print(f"Errors within +/-{near_band:.2f} of threshold: {int((pd.concat([fn, fp]).prediction_margin.abs() <= near_band).sum())} of {len(fn)+len(fp)}.")
    print("The model and frozen threshold were not loaded or modified.")


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_and_verify_predictions()
    frame = annotate_failure_modes(frame)
    save_outcome_tables(frame)
    plot_probability_distributions(frame)

    fn = frame[frame.outcome == "false_negative"].sort_values("damaged_probability")
    fp = frame[frame.outcome == "false_positive"].sort_values("damaged_probability", ascending=False)
    open_box_fn = fn[fn.open_box]
    open_box_tp = frame[(frame.outcome == "true_positive") & frame.open_box].sort_values("damaged_probability", ascending=False)

    make_image_grid(fn, FIGURE_DIR / "phase12_false_negatives.png", "False negatives: damaged parcels predicted intact")
    make_image_grid(fp, FIGURE_DIR / "phase12_false_positives.png", "False positives: intact parcels predicted damaged")
    make_image_grid(open_box_fn, FIGURE_DIR / "phase12_open_box_errors.png", "Open-box false negatives")
    make_image_grid(open_box_tp, FIGURE_DIR / "phase12_open_box_correct.png", "Correctly detected open boxes")
    write_taxonomy(frame)
    print_summary(frame)


if __name__ == "__main__":
    main()
