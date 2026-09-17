"""Post-hoc diagnostic: unchanged V2 MobileNet against corrected V3 validation labels."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision.models import mobilenet_v3_large
from torchvision.transforms import v2


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "processed" / "parcel_binary_v3"
MANIFEST = DATASET / "manifests" / "dataset_manifest.csv"
CHECKPOINT = ROOT / "models" / "transfer_learning" / "mobilenet_v3_large_frozen" / "best_model.pt"
OUTPUT = ROOT / "models" / "v3_label_correction_diagnostic"
EXPECTED_HASH = "f255600ad26289e7a3cc3b277926ffd91c8384a013a4203808b12d4ea77629d1"
THRESHOLD = 0.43458425998687744


class Images(Dataset):
    def __init__(self, rows, transform):
        self.rows, self.transform = rows, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(DATASET / row["v3_relative_path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, int(row["v3_binary_class"] == "damaged"), row["image_id"]


def main() -> None:
    if hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() != EXPECTED_HASH:
        raise ValueError("Frozen checkpoint hash mismatch")
    with MANIFEST.open(newline="", encoding="utf-8-sig") as handle:
        rows = [row for row in csv.DictReader(handle) if row["v2_split"] == "valid"]
    transform = v2.Compose([
        v2.Resize(256), v2.CenterCrop(224), v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    loader = DataLoader(Images(rows, transform), batch_size=64, shuffle=False, num_workers=0)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = mobilenet_v3_large(weights=None)
    model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, 1)
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    labels, probabilities, ids = [], [], []
    with torch.inference_mode():
        for images, batch_labels, batch_ids in loader:
            probabilities.extend(torch.sigmoid(model(images.to(device)).squeeze(1)).cpu().tolist())
            labels.extend(batch_labels.tolist())
            ids.extend(batch_ids)
    y = np.asarray(labels); p = np.asarray(probabilities); pred = p >= THRESHOLD
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    metrics = {
        "scope": "post_hoc_label_correction_diagnostic_not_model_selection",
        "dataset": "parcel_binary_v3 validation labels with preserved V2 split",
        "checkpoint_sha256": EXPECTED_HASH, "threshold": THRESHOLD, "images": len(y),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "recall": tp / (tp + fn), "specificity": tn / (tn + fp),
        "precision": tp / (tp + fp), "accuracy": (tp + tn) / len(y),
        "f1": 2 * tp / (2 * tp + fp + fn),
        "roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    with (OUTPUT / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["image_id", "true_label", "probability", "prediction"])
        writer.writerows(zip(ids, y.tolist(), p.tolist(), pred.astype(int).tolist()))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
