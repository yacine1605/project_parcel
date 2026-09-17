"""Phase 9B: train the Phase 9A CNN with realistic training augmentation.

Only the training transform changes. The architecture, optimizer, learning
rate, loss, input size, seed, epoch budget, and validation protocol remain the
same so Phase 9A versus 9B measures augmentation as independently as possible.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

# Reusing these Phase 9A definitions guarantees that the model architecture and
# explicit train/validation logic are identical. That source explains model.train(),
# model.eval(), tensor shapes, BCEWithLogitsLoss, backpropagation, and optimizer.step().
from train_custom_cnn_phase9a import (
    ParcelDamageCNN,
    ParcelDataset,
    set_random_seeds,
    train_one_epoch,
    validate_one_epoch,
)


# ---------------------------------------------------------
# 1. Training-only augmentation
# ---------------------------------------------------------
def create_training_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),

        transforms.RandomAffine(
            degrees=8,
            # A parcel can be slightly rotated because of camera placement or
            # conveyor orientation. Eight degrees stays realistic.
            translate=(0.05, 0.05),
            # Translation moves the parcel by at most 5% of image width/height,
            # representing small framing changes without cropping most of it.
            scale=(0.95, 1.05),
            # Mild zoom represents small camera-distance variation.
            interpolation=InterpolationMode.BILINEAR,
            fill=127,
        ),

        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        # Warehouse illumination and camera exposure can change brightness and
        # contrast. The ±15% range is mild and does not invent damage colors.

        transforms.RandomHorizontalFlip(p=0.5),
        # Left/right orientation does not change whether a parcel is damaged or
        # open, so horizontal flipping is semantically valid. Vertical flipping
        # is excluded because upside-down warehouse views are less realistic.

        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def create_validation_transform():
    # Validation is deterministic. Random augmentation here would make metrics
    # change between epochs and would no longer measure original validation images.
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def save_augmentation_grid(dataset_root: Path, train_rows: list[dict[str, str]], output: Path) -> None:
    # This preview uses the same random geometric/photometric operations but omits
    # normalization so ordinary RGB images can be displayed correctly.
    preview = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomAffine(8, translate=(0.05, 0.05), scale=(0.95, 1.05), interpolation=InterpolationMode.BILINEAR, fill=127),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.RandomHorizontalFlip(p=0.5),
    ])
    fig, axes = plt.subplots(3, 4, figsize=(10, 8))
    for axis, row in zip(axes.flat, train_rows[:12]):
        with Image.open(dataset_root / row["v2_relative_path"]) as image:
            augmented = preview(image.convert("RGB"))
        axis.imshow(augmented); axis.set_title(row["v2_binary_class"]); axis.axis("off")
    fig.suptitle("Phase 9B mild training-only augmentation examples")
    fig.tight_layout(); fig.savefig(output, dpi=150); plt.close(fig)


# ---------------------------------------------------------
# 2. Main experiment
# ---------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("datasets/processed/parcel_binary_v2"))
    parser.add_argument("--output", type=Path, default=Path("models/phase9b_custom_cnn"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite experiment: {args.output}")
    args.output.mkdir(parents=True)

    seed = 42
    learning_rate = 1e-3
    set_random_seeds(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Selected device: {device}")
    if device.type == "cuda": print(f"GPU: {torch.cuda.get_device_name(0)}")

    with (args.dataset / "manifests/dataset_manifest.csv").open(newline="", encoding="utf-8") as handle:
        # Only train/validation rows are retained. No test Dataset is constructed.
        rows = [r for r in csv.DictReader(handle) if r["v2_split"] in {"train", "valid"}]
    train_rows = [r for r in rows if r["v2_split"] == "train"]
    valid_rows = [r for r in rows if r["v2_split"] == "valid"]
    if len(train_rows) != 2765 or len(valid_rows) != 592:
        raise RuntimeError("Frozen development split counts changed")
    if {r["group_id"] for r in train_rows} & {r["group_id"] for r in valid_rows}:
        raise RuntimeError("Train/validation group leakage")

    save_augmentation_grid(args.dataset, train_rows, args.output / "augmentation_examples.png")
    # Reset seeds after generating the preview so the preview does not change the
    # random sequence used during actual training.
    set_random_seeds(seed)

    train_dataset = ParcelDataset(args.dataset, train_rows, create_training_transform())
    valid_dataset = ParcelDataset(args.dataset, valid_rows, create_validation_transform())
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, generator=generator)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # This instantiates the exact Phase 9A architecture from random weights.
    # It does not load Phase 9A or any pretrained weights.
    model = ParcelDamageCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    # BCEWithLogitsLoss expects raw logits [batch_size] and labels [batch_size].
    # It combines sigmoid and binary cross-entropy in a numerically stable form.
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Print the same shape walkthrough as Phase 9A.
    with torch.no_grad():
        example = torch.zeros(2, 3, 224, 224, device=device)
        x1=model.block1(example); x2=model.block2(x1); x3=model.block3(x2)
        pooled=model.global_pool(x3).flatten(1); logits=model(example)
    print("Tensor shape walkthrough:")
    print(f"  Input batch: {list(example.shape)}")
    print(f"  After Conv Block 1: {list(x1.shape)}")
    print(f"  After Conv Block 2: {list(x2.shape)}")
    print(f"  After Conv Block 3: {list(x3.shape)}")
    print(f"  After Global Average Pooling: {list(pooled.shape)}")
    print(f"  Final logits: {list(logits.shape)}")

    history=[]; best_validation_loss=float("inf"); best_epoch=0; started=time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        # train_one_epoch performs: load batch, move to GPU, zero gradients,
        # forward pass, BCE loss, backward pass, optimizer step, and metrics.
        train_loss, train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        # validate_one_epoch uses model.eval() and torch.no_grad(): no learning or
        # weight update occurs, and sigmoid converts logits to probabilities.
        valid_loss, valid_metrics, ids, labels, probabilities = validate_one_epoch(model, valid_loader, criterion, device)
        row={"epoch":epoch,"train_loss":train_loss,"validation_loss":valid_loss,
             **{f"train_{k}":v for k,v in train_metrics.items() if k!="confusion_matrix"},
             **{f"validation_{k}":v for k,v in valid_metrics.items() if k!="confusion_matrix"}}
        history.append(row)
        print(f"Epoch {epoch:02d}/{args.epochs} | train loss {train_loss:.4f} | valid loss {valid_loss:.4f} | valid recall {valid_metrics['damaged_recall']:.2%} | valid specificity {valid_metrics['specificity']:.2%}")
        if valid_loss < best_validation_loss:
            best_validation_loss=valid_loss; best_epoch=epoch
            torch.save({"model_state_dict":model.state_dict(),"epoch":epoch,"validation_loss":valid_loss,
                        "architecture":"ParcelDamageCNN","threshold":0.5,"augmentation":"phase9b_mild"},args.output/"best_model.pt")

    checkpoint=torch.load(args.output/"best_model.pt",map_location=device,weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    final_loss, metrics, ids, labels, probabilities=validate_one_epoch(model,valid_loader,criterion,device)
    tn,fp,fn,tp=np.asarray(metrics["confusion_matrix"]).ravel()
    metrics.update({"balanced_accuracy":float((metrics["damaged_recall"]+metrics["specificity"])/2),
                    "tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp)})

    # Frozen subgroup definitions come from metadata created before this experiment.
    by_id={r["image_id"]:r for r in valid_rows}; subgroup_counts=defaultdict(lambda:{"support":0,"detected":0})
    for iid,label,probability in zip(ids,labels,probabilities):
        if label != 1: continue
        name="open_box" if by_id[iid]["phase8b_final_label"]=="open_box" else "ordinary_damaged"
        subgroup_counts[name]["support"]+=1; subgroup_counts[name]["detected"]+=int(probability>=0.5)
    subgroups={k:{**v,"recall":v["detected"]/v["support"]} for k,v in subgroup_counts.items()}

    with (args.output/"validation_predictions.csv").open("w",newline="",encoding="utf-8") as handle:
        w=csv.writer(handle); w.writerow(["image_id","true_label","damaged_probability","threshold","predicted_label","subgroup"])
        for iid,label,probability in zip(ids,labels,probabilities):
            subgroup="open_box" if by_id[iid]["phase8b_final_label"]=="open_box" else "ordinary_damaged" if label==1 else "intact"
            w.writerow([iid,int(label),float(probability),0.5,int(probability>=0.5),subgroup])
    with (args.output/"training_history.csv").open("w",newline="",encoding="utf-8") as handle:
        w=csv.DictWriter(handle,fieldnames=list(history[0])); w.writeheader(); w.writerows(history)

    total_seconds=time.perf_counter()-started
    metadata={"phase":"9B","experiment":"same_custom_cnn_with_mild_training_augmentation","dataset":"parcel_binary_v2",
              "splits_used":["train","valid"],"test_accessed":False,"trained_from_scratch":True,"seed":seed,
              "epochs":args.epochs,"batch_size":args.batch_size,"learning_rate":learning_rate,"optimizer":"Adam",
              "loss":"BCEWithLogitsLoss","checkpoint_criterion":"minimum validation loss","best_epoch":best_epoch,
              "best_validation_loss":best_validation_loss,"validation_threshold":0.5,"validation_metrics":metrics,
              "subgroup_results":subgroups,"training_seconds":total_seconds,
              "augmentations":{"rotation_degrees":8,"translation_fraction":[0.05,0.05],"scale_range":[0.95,1.05],
                               "brightness":0.15,"contrast":0.15,"horizontal_flip_probability":0.5},
              "torch":torch.__version__,"torchvision":__import__("torchvision").__version__,"device":str(device),
              "gpu":torch.cuda.get_device_name(0) if device.type=="cuda" else None}
    (args.output/"experiment_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")

    fig,axes=plt.subplots(1,2,figsize=(11,4)); epochs=[r["epoch"] for r in history]
    axes[0].plot(epochs,[r["train_loss"] for r in history],label="Train"); axes[0].plot(epochs,[r["validation_loss"] for r in history],label="Validation"); axes[0].axvline(best_epoch,color="gray",linestyle="--",label=f"Best epoch {best_epoch}"); axes[0].set(title="Loss",xlabel="Epoch",ylabel="BCEWithLogitsLoss"); axes[0].legend(); axes[0].grid(alpha=.25)
    axes[1].plot(epochs,[r["validation_damaged_recall"] for r in history],label="Damaged recall"); axes[1].plot(epochs,[r["validation_specificity"] for r in history],label="Specificity"); axes[1].set(title="Validation metrics at probability 0.5",xlabel="Epoch",ylabel="Rate"); axes[1].legend(); axes[1].grid(alpha=.25)
    fig.tight_layout(); fig.savefig(args.output/"training_curves.png",dpi=160); plt.close(fig)

    print("\nTraining complete.")
    print(f"Best validation epoch: {best_epoch}")
    print(f"Validation damaged recall: {metrics['damaged_recall']:.1%}")
    print(f"Validation specificity: {metrics['specificity']:.1%}")
    print(f"The model detected about {round(metrics['damaged_recall']*100)} out of every 100 damaged parcels.")
    print(f"It correctly left about {round(metrics['specificity']*100)} out of every 100 intact parcels as intact.")
    print("The test split was not accessed.")


if __name__ == "__main__":
    main()
