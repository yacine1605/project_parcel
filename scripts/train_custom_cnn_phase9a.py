"""Train an educational custom CNN on parcel_binary_v2 train/validation only.

This is a new model family after the closed Phase 8 HOG experiment. The test
split is deliberately excluded; it will require a separate frozen evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score, average_precision_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# ---------------------------------------------------------
# 1. Reproducibility
# ---------------------------------------------------------
# A fixed seed makes weight initialization, data shuffling, and other random
# operations repeatable as far as the installed hardware allows.
def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------
# 2. Dataset and DataLoader
# ---------------------------------------------------------
# The Dataset tells PyTorch how to load one image and label. The DataLoader
# later groups samples into batches. Only train/validation rows are accepted;
# the code never builds a list of test samples.
class ParcelDataset(Dataset):
    def __init__(self, dataset_root: Path, rows: list[dict[str, str]], transform) -> None:
        self.dataset_root = dataset_root
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(self.dataset_root / row["v2_relative_path"]) as image:
            image = image.convert("RGB")
            image_tensor = self.transform(image)

        # One floating-point label is used because BCEWithLogitsLoss expects
        # binary targets such as 0.0 (intact) and 1.0 (damaged).
        label = torch.tensor(1.0 if row["v2_binary_class"] == "damaged" else 0.0)
        return image_tensor, label, row["image_id"]


# ---------------------------------------------------------
# 3. Custom convolutional neural network
# ---------------------------------------------------------
class ParcelDamageCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # RGB images have 3 input channels. The first layer learns 32 filters;
        # early filters commonly learn edges, corners, and texture changes.
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 224x224 -> 112x112
        )

        # Deeper filters combine simple edges into cardboard boundaries,
        # folds, surface changes, and possible openings.
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 112x112 -> 56x56
        )

        # The third block can learn more complex structures such as holes,
        # tears, deformation, crushed corners, and package openings.
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 56x56 -> 28x28
        )

        # Global average pooling summarizes each of the 128 feature maps with
        # one number. This avoids a very large, difficult-to-follow dense layer.
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.30)
        self.output = nn.Linear(128, 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # Expected shapes for batch size B:
        # input [B,3,224,224] -> block1 [B,32,112,112]
        # -> block2 [B,64,56,56] -> block3 [B,128,28,28]
        features = self.block1(images)
        features = self.block2(features)
        features = self.block3(features)
        features = self.global_pool(features)       # [B,128,1,1]
        features = torch.flatten(features, start_dim=1)  # [B,128]
        features = self.dropout(features)
        logits = self.output(features).squeeze(1)   # [B]

        # We return raw logits, not sigmoid probabilities. BCEWithLogitsLoss
        # combines sigmoid and binary cross-entropy more stably during training.
        return logits


def calculate_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> dict[str, object]:
    predictions = probabilities >= threshold
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        # Recall: of all truly damaged parcels, how many did we detect?
        "damaged_precision": float(precision),
        "damaged_recall": float(recall),
        "damaged_f1": float(f1),
        # Specificity: of all truly intact parcels, how many stayed intact?
        "specificity": float(tn / (tn + fp)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


# ---------------------------------------------------------
# 4. One training epoch
# ---------------------------------------------------------
# One epoch means the model sees every training image once, in shuffled batches.
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    # train() enables training behavior in Dropout and lets BatchNorm update its
    # running statistics.
    total_loss = 0.0
    all_labels, all_probabilities = [], []

    for images, labels, _ in loader:
        # Step 1-2: load the batch and move tensors to the same CPU/GPU as model.
        # images: [batch_size,3,224,224]; labels: [batch_size]
        images = images.to(device)
        labels = labels.to(device)

        # Step 3: PyTorch accumulates gradients by default. Clear gradients from
        # the previous batch before calculating gradients for this batch.
        optimizer.zero_grad()

        # Step 4: forward pass. One raw logit is produced per image: [batch_size].
        logits = model(images)

        # Step 5: BCEWithLogitsLoss combines sigmoid and binary cross-entropy.
        # It is numerically safer than manually using sigmoid followed by BCELoss.
        loss = criterion(logits, labels)

        # Step 6: backpropagation calculates how every trainable parameter
        # contributed to the error.
        loss.backward()

        # Step 7: Adam uses those gradients to update the model weights.
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        # Sigmoid is appropriate here because metrics need probabilities [0,1].
        probabilities = torch.sigmoid(logits.detach())
        all_labels.extend(labels.detach().cpu().numpy())
        all_probabilities.extend(probabilities.cpu().numpy())

    metrics = calculate_metrics(np.asarray(all_labels), np.asarray(all_probabilities))
    return total_loss / len(loader.dataset), metrics


# ---------------------------------------------------------
# 5. Validation epoch
# ---------------------------------------------------------
def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    # eval() disables Dropout and tells BatchNorm to use learned running values.
    total_loss = 0.0
    all_labels, all_probabilities, all_ids = [], [], []

    with torch.no_grad():
        # Validation makes predictions only. Gradients and weight updates are not
        # needed, so no_grad() saves GPU memory and computation.
        for images, labels, image_ids in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            probabilities = torch.sigmoid(logits)
            total_loss += loss.item() * images.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_ids.extend(image_ids)

    labels_array = np.asarray(all_labels)
    probabilities_array = np.asarray(all_probabilities)
    metrics = calculate_metrics(labels_array, probabilities_array)
    return total_loss / len(loader.dataset), metrics, all_ids, labels_array, probabilities_array


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("datasets/processed/parcel_binary_v2"))
    parser.add_argument("--output", type=Path, default=Path("models/phase9a_custom_cnn"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite experiment: {args.output}")
    args.output.mkdir(parents=True)

    seed = 42
    learning_rate = 1e-3
    # 1e-3 is a common baseline learning rate for Adam. It is a simple starting
    # value, not a claim that validation has proven it optimal.
    set_random_seeds(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # The model and input tensors must be on the same device.
    print(f"Selected device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # No random augmentation is used in this first controlled CNN baseline.
    # Both transforms resize deterministically and normalize RGB values using
    # common ImageNet channel statistics. A later augmentation experiment can
    # change this one variable without changing the baseline retrospectively.
    training_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    validation_transform = transforms.Compose([
        # Validation has no random augmentation: metrics must describe the same
        # images on every epoch.
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    with (args.dataset / "manifests/dataset_manifest.csv").open(newline="", encoding="utf-8") as handle:
        # Rows are retained only when split is train/valid. Test rows are not put
        # into any Dataset or DataLoader in this development phase.
        development_rows = [r for r in csv.DictReader(handle) if r["v2_split"] in {"train", "valid"}]
    train_rows = [r for r in development_rows if r["v2_split"] == "train"]
    valid_rows = [r for r in development_rows if r["v2_split"] == "valid"]
    if len(train_rows) != 2765 or len(valid_rows) != 592:
        raise RuntimeError("Frozen train/validation counts do not match Phase 8C")
    train_groups = {r["group_id"] for r in train_rows}
    valid_groups = {r["group_id"] for r in valid_rows}
    if train_groups & valid_groups:
        raise RuntimeError("Group leakage between train and validation")

    train_dataset = ParcelDataset(args.dataset, train_rows, training_transform)
    valid_dataset = ParcelDataset(args.dataset, valid_rows, validation_transform)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, generator=generator)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = ParcelDamageCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    # The first baseline uses unweighted loss so it stays simple and comparable.
    # Class weighting can be a separately documented experiment later.
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Print important shapes once so a learner can follow the model.
    with torch.no_grad():
        example = torch.zeros(2, 3, 224, 224, device=device)
        x1 = model.block1(example); x2 = model.block2(x1); x3 = model.block3(x2)
        pooled = model.global_pool(x3).flatten(1); logits = model(example)
    print("Tensor shape walkthrough:")
    print(f"  Input batch: {list(example.shape)}")
    print(f"  After Conv Block 1: {list(x1.shape)}")
    print(f"  After Conv Block 2: {list(x2.shape)}")
    print(f"  After Conv Block 3: {list(x3.shape)}")
    print(f"  After Global Average Pooling: {list(pooled.shape)}")
    print(f"  Final logits: {list(logits.shape)}")

    history = []
    best_validation_loss = float("inf")
    best_epoch = 0
    training_started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        valid_loss, valid_metrics, ids, labels, probabilities = validate_one_epoch(model, valid_loader, criterion, device)
        row = {"epoch": epoch, "train_loss": train_loss, "validation_loss": valid_loss,
               **{f"train_{k}": v for k, v in train_metrics.items() if k != "confusion_matrix"},
               **{f"validation_{k}": v for k, v in valid_metrics.items() if k != "confusion_matrix"}}
        history.append(row)
        print(f"Epoch {epoch:02d}/{args.epochs} | train loss {train_loss:.4f} | valid loss {valid_loss:.4f} | valid recall {valid_metrics['damaged_recall']:.2%} | valid specificity {valid_metrics['specificity']:.2%}")

        # Validation loss is the predeclared checkpoint criterion. Saving only
        # model weights does not train on validation; it selects an epoch.
        if valid_loss < best_validation_loss:
            best_validation_loss = valid_loss
            best_epoch = epoch
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch,
                        "validation_loss": valid_loss, "architecture": "ParcelDamageCNN",
                        "threshold": 0.5}, args.output / "best_model.pt")

    # Reload the chosen validation-loss checkpoint for permanent validation output.
    checkpoint = torch.load(args.output / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    final_loss, final_metrics, ids, labels, probabilities = validate_one_epoch(model, valid_loader, criterion, device)
    with (args.output / "validation_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["image_id", "true_label", "damaged_probability", "threshold", "predicted_label"])
        for iid, label, probability in zip(ids, labels, probabilities):
            writer.writerow([iid, int(label), float(probability), 0.5, int(probability >= 0.5)])
    with (args.output / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0])); writer.writeheader(); writer.writerows(history)

    metadata = {"phase": "9A", "experiment": "custom_cnn_no_augmentation", "dataset": "parcel_binary_v2",
                "splits_used": ["train", "valid"], "test_accessed": False, "seed": seed,
                "epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": learning_rate,
                "optimizer": "Adam", "loss": "BCEWithLogitsLoss", "checkpoint_criterion": "minimum validation loss",
                "best_epoch": best_epoch, "feature_learning": "learned CNN features", "validation_threshold": 0.5,
                "validation_loss": final_loss, "validation_metrics": final_metrics,
                "training_seconds": time.perf_counter() - training_started,
                "torch": torch.__version__, "torchvision": __import__("torchvision").__version__,
                "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None}
    (args.output / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = [r["epoch"] for r in history]
    axes[0].plot(epochs, [r["train_loss"] for r in history], label="Train")
    axes[0].plot(epochs, [r["validation_loss"] for r in history], label="Validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="BCEWithLogitsLoss"); axes[0].legend(); axes[0].grid(alpha=.25)
    axes[1].plot(epochs, [r["validation_damaged_recall"] for r in history], label="Damaged recall")
    axes[1].plot(epochs, [r["validation_specificity"] for r in history], label="Specificity")
    axes[1].set(title="Validation metrics at probability 0.5", xlabel="Epoch", ylabel="Rate"); axes[1].legend(); axes[1].grid(alpha=.25)
    fig.tight_layout(); fig.savefig(args.output / "training_curves.png", dpi=160); plt.close(fig)

    print("\nTraining complete.")
    print(f"Best validation epoch: {best_epoch}")
    print(f"Validation damaged recall: {final_metrics['damaged_recall']:.1%}")
    print(f"Validation specificity: {final_metrics['specificity']:.1%}")
    print(f"The model detected about {round(final_metrics['damaged_recall'] * 100)} out of every 100 damaged parcelys.")
    print(f"It correctly left about {round(final_metrics['specificity'] * 100)} out of every 100 intact parcels as intact.")
    print("The test split was not accessed.")


if __name__ == "__main__":
    main()
