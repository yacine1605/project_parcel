"""Train and calibrate a strict-frozen MobileNetV3 head on parcel_binary_v3."""

from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import mobilenet_v3_large
from torchvision.transforms import v2


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "processed" / "parcel_binary_v3"
SOURCE_CHECKPOINT = ROOT / "models" / "transfer_learning" / "mobilenet_v3_large_frozen" / "best_model.pt"
OUTPUT = ROOT / "models" / "mobilenet_v3_v3_development"
SEED, EPOCHS, BATCH_SIZE, LR = 42, 10, 32, 0.001


def seed_everything() -> None:
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = p >= threshold
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    return {
        "threshold": float(threshold), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "recall": tp / (tp + fn), "specificity": tn / (tn + fp),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "accuracy": (tp + tn) / len(y),
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
    }


def infer(model, loader, device):
    model.eval(); labels, probabilities = [], []
    with torch.inference_mode():
        for images, target in loader:
            probabilities.extend(torch.sigmoid(model(images.to(device)).squeeze(1)).cpu().tolist())
            labels.extend(target.tolist())
    return np.asarray(labels), np.asarray(probabilities)


def main() -> None:
    seed_everything()
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    transform = v2.Compose([
        v2.Resize(256), v2.CenterCrop(224), v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_set = ImageFolder(DATASET / "train", transform=transform)
    valid_set = ImageFolder(DATASET / "valid", transform=transform)
    if train_set.class_to_idx != {"damaged": 0, "intact": 1}:
        raise ValueError(train_set.class_to_idx)
    # ImageFolder is alphabetical; convert intact=1 to damaged=1.
    def collate(batch):
        images, labels = zip(*batch)
        return torch.stack(images), 1 - torch.tensor(labels, dtype=torch.float32)
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True, num_workers=0,
                              collate_fn=collate, generator=generator)
    valid_loader = DataLoader(valid_set, BATCH_SIZE, shuffle=False, num_workers=0,
                              collate_fn=collate)

    source = torch.load(SOURCE_CHECKPOINT, map_location="cpu", weights_only=False)
    model = mobilenet_v3_large(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 1)
    model.load_state_dict(source["model_state_dict"])
    for parameter in model.parameters(): parameter.requires_grad = False
    torch.manual_seed(SEED)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 1)
    for parameter in model.classifier[3].parameters(): parameter.requires_grad = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.classifier[3].parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()
    history, best_loss, started = [], float("inf"), time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.eval()  # keep frozen BatchNorm and Dropout deterministic
        train_loss, seen = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images).squeeze(1); loss = criterion(logits, labels)
            loss.backward(); optimizer.step()
            train_loss += loss.item() * len(labels); seen += len(labels)
        y, p = infer(model, valid_loader, device)
        eps = 1e-7
        val_loss = float(-(y*np.log(p+eps)+(1-y)*np.log(1-p+eps)).mean())
        row = {"epoch": epoch, "train_loss": train_loss/seen, "validation_loss": val_loss,
               **{f"validation_{k}": v for k, v in metrics(y, p, 0.5).items() if k not in {"threshold","tp","tn","fp","fn"}}}
        history.append(row); print(json.dumps(row))
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({"architecture":"mobilenet_v3_large", "model_state_dict":model.state_dict(),
                        "epoch":epoch, "validation_loss":val_loss, "backbone_frozen":True,
                        "dataset_version":"parcel_binary_v3"}, OUTPUT/"best_model.pt")

    best = torch.load(OUTPUT/"best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"]); y, p = infer(model, valid_loader, device)
    candidates = np.unique(np.r_[p, 0.0, 1.0])
    eligible = [metrics(y, p, t) for t in candidates if metrics(y, p, t)["recall"] >= 0.90]
    calibrated = max(eligible, key=lambda m:(m["specificity"],m["precision"],m["f1"],-abs(m["threshold"]-0.5)))
    result = {"status":"development_only_no_fresh_final_test", "dataset":"parcel_binary_v3",
              "train_images":len(train_set), "validation_images":len(valid_set), "test_accessed":False,
              "seed":SEED, "epochs":EPOCHS, "batch_size":BATCH_SIZE, "learning_rate":LR,
              "best_epoch":best["epoch"], "best_validation_loss":best_loss,
              "default_metrics":metrics(y,p,0.5), "calibrated_metrics":calibrated,
              "elapsed_seconds":time.perf_counter()-started}
    (OUTPUT/"results.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    with (OUTPUT/"training_history.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=history[0]); writer.writeheader(); writer.writerows(history)
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
