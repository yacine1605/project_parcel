"""Phase 10A: partially fine-tune one frozen-backbone Experiment A model.

# ============================================================
# What are we doing in this script?
# ============================================================
#
# We start from a completed Experiment A checkpoint. Its ImageNet backbone was
# frozen while a parcel classification head learned. We now allow only the last
# feature blocks and the classification head to learn at small learning rates.
#
# The purpose is to test whether late pretrained features can adapt to parcel-
# specific patterns such as crushed geometry, tears, holes, open flaps, tape,
# seams, labels, cardboard texture, and deformation.
#
# We are NOT training from scratch, changing dataset splits, calibrating a
# threshold, or opening the test split. Validation checkpoint selection uses
# minimum validation loss, and metrics are reported at threshold 0.5.
"""

from __future__ import annotations

import argparse, csv, hashlib, json, platform, random, sys, time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


def set_seeds(seed: int) -> None:
    """Make initialization and batch shuffling repeatable where practical."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------
# 1. Dataset and DataLoader
# ---------------------------------------------------------
# A Dataset explains how to load ONE sample. A DataLoader groups samples into
# batches. For batch size 16, image tensors have shape [16,3,224,224]:
# 16 images, 3 RGB channels, height 224, width 224.
class ParcelDataset(Dataset):
    def __init__(self, root: Path, rows: list[dict[str, str]], transform) -> None:
        self.root, self.rows, self.transform = root, rows, transform

    def __len__(self) -> int: return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(self.root / row["v2_relative_path"]) as image:
            image_tensor = self.transform(image.convert("RGB"))
        # BCEWithLogitsLoss expects a floating target: 0.0 intact, 1.0 damaged.
        label = torch.tensor(1.0 if row["v2_binary_class"] == "damaged" else 0.0)
        return image_tensor, label, row["image_id"]


def imagenet_transform():
    # Pretrained ImageNet filters expect this scale and channel normalization.
    # Train and validation are deterministic: augmentation is not a new variable.
    return transforms.Compose([
        transforms.Resize(256),       # Preserve aspect ratio; shorter side -> 256.
        transforms.CenterCrop(224),   # Produce the expected [3,224,224] input.
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])


# ---------------------------------------------------------
# 2. Load Experiment A and choose late trainable blocks
# ---------------------------------------------------------
def build_model(architecture: str, experiment_a_checkpoint: Path):
    # Transfer learning reuses knowledge learned from millions of ImageNet images.
    # We recreate the standard torchvision architecture and then load the exact
    # Experiment A parcel head and frozen backbone state.
    if architecture == "resnet50":
        model=models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc=nn.Linear(model.fc.in_features,1)
        trainable_prefixes=("layer4.","fc.")
        explanation="ResNet layer4 (last residual stage) plus classification head"
    elif architecture == "efficientnet_b0":
        model=models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1]=nn.Linear(model.classifier[1].in_features,1)
        trainable_prefixes=("features.7.","features.8.","classifier.")
        explanation="EfficientNet final two feature blocks plus classifier"
    elif architecture == "mobilenet_v3_large":
        model=models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        model.classifier[3]=nn.Linear(model.classifier[3].in_features,1)
        trainable_prefixes=("features.15.","features.16.","classifier.")
        explanation="MobileNet final two feature blocks plus classifier"
    else: raise ValueError(architecture)

    checkpoint=torch.load(experiment_a_checkpoint,map_location="cpu",weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    # requires_grad=False means backpropagation will not calculate an update for
    # this parameter. requires_grad=True means the parameter may learn.
    for parameter in model.parameters(): parameter.requires_grad=False
    for name,parameter in model.named_parameters():
        if name.startswith(trainable_prefixes): parameter.requires_grad=True
    return model,trainable_prefixes,explanation


def put_frozen_blocks_in_eval_mode(model: nn.Module) -> None:
    # model.train() is needed for trainable late blocks. However, early frozen
    # blocks must not accidentally change BatchNorm running means/variances or
    # activate stochastic training behavior. Any complete submodule with no
    # trainable parameters is therefore returned to eval mode.
    for module in model.modules():
        parameters=list(module.parameters())
        if parameters and not any(parameter.requires_grad for parameter in parameters):
            module.eval()


def parameter_counts(model):
    total=sum(p.numel() for p in model.parameters()); trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total,total-trainable,trainable,100*trainable/total


# ---------------------------------------------------------
# 3. Metrics
# ---------------------------------------------------------
def calculate_metrics(labels,probabilities,threshold=0.5):
    predictions=probabilities>=threshold
    precision,recall,f1,_=precision_recall_fscore_support(labels,predictions,average="binary",zero_division=0)
    tn,fp,fn,tp=confusion_matrix(labels,predictions,labels=[0,1]).ravel(); specificity=tn/(tn+fp)
    # Recall: of all damaged parcels, how many were detected? Specificity: of all
    # intact parcels, how many remained intact? Low specificity creates reviews.
    return {"accuracy":float(accuracy_score(labels,predictions)),"damaged_precision":float(precision),
            "damaged_recall":float(recall),"damaged_f1":float(f1),"specificity":float(specificity),
            "balanced_accuracy":float((recall+specificity)/2),"roc_auc":float(roc_auc_score(labels,probabilities)),
            "pr_auc":float(average_precision_score(labels,probabilities)),"tp":int(tp),"tn":int(tn),"fp":int(fp),"fn":int(fn),
            "confusion_matrix":[[int(tn),int(fp)],[int(fn),int(tp)]]}


# ---------------------------------------------------------
# 4. One explicit fine-tuning epoch
# ---------------------------------------------------------
def train_one_epoch(model,loader,criterion,optimizer,device):
    # An epoch means one pass through every training image.
    model.train(); put_frozen_blocks_in_eval_mode(model)
    running_loss=0.0; labels_all=[]; probabilities_all=[]
    for images,labels,_ in loader:
        # Step 1: DataLoader supplies a batch.
        # Step 2: model and tensors must share the CPU/GPU device.
        images=images.to(device); labels=labels.to(device)

        # Step 3: PyTorch accumulates gradients by default. Clear old batch
        # gradients before calculating gradients for this batch.
        optimizer.zero_grad()

        # Step 4: Forward pass. Images [B,3,224,224] pass through the CNN and
        # produce logits [B], one unrestricted raw score per parcel.
        logits=model(images).squeeze(1)

        # Step 5: BCEWithLogitsLoss combines sigmoid and binary cross-entropy in
        # a numerically stable operation. The model should output logits rather
        # than apply sigmoid internally during training.
        loss=criterion(logits,labels)

        # Step 6: Backpropagation calculates gradients for trainable late blocks
        # and the head. Frozen early parameters receive no gradients.
        loss.backward()

        # Step 7: Adam applies the parameter-group learning rates to update the
        # late backbone gently and the classification head somewhat faster.
        optimizer.step()

        # Step 8: Sigmoid converts logits to probabilities for readable metrics.
        running_loss+=loss.item()*images.size(0); probabilities=torch.sigmoid(logits.detach())
        labels_all.extend(labels.detach().cpu().numpy()); probabilities_all.extend(probabilities.cpu().numpy())
    return running_loss/len(loader.dataset),calculate_metrics(np.asarray(labels_all),np.asarray(probabilities_all))


# ---------------------------------------------------------
# 5. Validation without learning
# ---------------------------------------------------------
def validate(model,loader,criterion,device):
    model.eval()
    # eval() makes Dropout deterministic and BatchNorm use saved running values.
    running_loss=0.0; labels_all=[]; probabilities_all=[]; ids_all=[]
    with torch.no_grad():
        # Validation does not update weights. no_grad() saves memory/computation.
        for images,labels,image_ids in loader:
            images=images.to(device); labels=labels.to(device); logits=model(images).squeeze(1)
            loss=criterion(logits,labels); probabilities=torch.sigmoid(logits)
            running_loss+=loss.item()*images.size(0); labels_all.extend(labels.cpu().numpy()); probabilities_all.extend(probabilities.cpu().numpy()); ids_all.extend(image_ids)
    labels_array=np.asarray(labels_all); probabilities_array=np.asarray(probabilities_all)
    return running_loss/len(loader.dataset),calculate_metrics(labels_array,probabilities_array),ids_all,labels_array,probabilities_array


def measure_latency(model,loader,device):
    model.eval(); elapsed=0.0; count=0
    with torch.no_grad():
        for images,_,_ in loader:
            images=images.to(device)
            if device.type=="cuda": torch.cuda.synchronize()
            started=time.perf_counter(); model(images)
            if device.type=="cuda": torch.cuda.synchronize()
            elapsed+=time.perf_counter()-started; count+=images.size(0)
    return 1000*elapsed/count


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--architecture",required=True,choices=["resnet50","efficientnet_b0","mobilenet_v3_large"]); parser.add_argument("--dataset",type=Path,default=Path("datasets/processed/parcel_binary_v2")); parser.add_argument("--experiment-a-root",type=Path,default=Path("models/transfer_learning")); parser.add_argument("--output-root",type=Path,default=Path("models/transfer_learning_finetuned")); parser.add_argument("--epochs",type=int,default=10); parser.add_argument("--batch-size",type=int,default=16); args=parser.parse_args()
    output=args.output_root/f"{args.architecture}_partial_finetune"
    if output.exists(): raise FileExistsError(f"Refusing to overwrite experiment: {output}")
    output.mkdir(parents=True)
    experiment_a=args.experiment_a_root/f"{args.architecture}_frozen"/"best_model.pt"
    if not experiment_a.exists(): raise FileNotFoundError(experiment_a)
    seed=42; backbone_lr=1e-4; head_lr=5e-4
    # The backbone already contains useful knowledge, so 1e-4 makes small updates
    # that are less likely to destroy it. The new head may adapt faster at 5e-4.
    set_seeds(seed); torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f"Architecture: {args.architecture}\nDevice: {device}")
    if device.type=="cuda": print(f"GPU: {torch.cuda.get_device_name(0)}")

    with (args.dataset/"manifests/dataset_manifest.csv").open(newline="",encoding="utf-8") as handle:
        development=[r for r in csv.DictReader(handle) if r["v2_split"] in {"train","valid"}]
    train_rows=[r for r in development if r["v2_split"]=="train"]; valid_rows=[r for r in development if r["v2_split"]=="valid"]
    if len(train_rows)!=2765 or len(valid_rows)!=592: raise RuntimeError("Frozen development counts changed")
    if {r["group_id"] for r in train_rows}&{r["group_id"] for r in valid_rows}: raise RuntimeError("Group leakage")
    transform=imagenet_transform(); generator=torch.Generator().manual_seed(seed)
    train_loader=DataLoader(ParcelDataset(args.dataset,train_rows,transform),batch_size=args.batch_size,shuffle=True,num_workers=0,generator=generator)
    valid_loader=DataLoader(ParcelDataset(args.dataset,valid_rows,transform),batch_size=args.batch_size,shuffle=False,num_workers=0)

    model,prefixes,explanation=build_model(args.architecture,experiment_a); total,frozen,trainable,percent=parameter_counts(model); model=model.to(device)
    trainable_names=[name for name,p in model.named_parameters() if p.requires_grad]
    print(f"Fine-tuning: {explanation}\nTotal parameters: {total:,}\nFrozen: {frozen:,}\nTrainable: {trainable:,} ({percent:.2f}%)")
    print("First trainable parameter names:"); [print(" ",name) for name in trainable_names[:12]]

    # BatchNorm has learnable scale/shift (affine parameters) plus non-gradient
    # running mean/variance buffers. BN in frozen blocks stays eval/fixed. BN in
    # explicitly trainable late blocks may learn affine values and running stats.
    head_prefix="fc." if args.architecture=="resnet50" else "classifier."
    backbone_parameters=[p for name,p in model.named_parameters() if p.requires_grad and not name.startswith(head_prefix)]
    head_parameters=[p for name,p in model.named_parameters() if p.requires_grad and name.startswith(head_prefix)]
    optimizer=torch.optim.Adam([{"params":backbone_parameters,"lr":backbone_lr},{"params":head_parameters,"lr":head_lr}])
    criterion=nn.BCEWithLogitsLoss(); history=[]; best_loss=float("inf"); best_epoch=0; started=time.perf_counter()
    for epoch in range(1,args.epochs+1):
        train_loss,train_metrics=train_one_epoch(model,train_loader,criterion,optimizer,device)
        valid_loss,valid_metrics,ids,labels,probabilities=validate(model,valid_loader,criterion,device)
        row={"epoch":epoch,"train_loss":train_loss,"validation_loss":valid_loss,**{f"train_{k}":v for k,v in train_metrics.items() if k!="confusion_matrix"},**{f"validation_{k}":v for k,v in valid_metrics.items() if k!="confusion_matrix"}}; history.append(row)
        print(f"Epoch {epoch:02d}/{args.epochs} | train loss {train_loss:.4f} | valid loss {valid_loss:.4f} | recall {valid_metrics['damaged_recall']:.2%} | specificity {valid_metrics['specificity']:.2%}")
        if valid_loss<best_loss:
            best_loss=valid_loss; best_epoch=epoch; torch.save({"architecture":args.architecture,"model_state_dict":model.state_dict(),"epoch":epoch,"validation_loss":valid_loss,"threshold":0.5,"partial_finetune":True,"trainable_prefixes":prefixes},output/"best_model.pt")

    training_seconds=time.perf_counter()-started; checkpoint=torch.load(output/"best_model.pt",map_location=device,weights_only=True); model.load_state_dict(checkpoint["model_state_dict"])
    validation_loss,validation_metrics,ids,labels,probabilities=validate(model,valid_loader,criterion,device); latency=measure_latency(model,valid_loader,device)
    by_id={r["image_id"]:r for r in valid_rows}; groups=defaultdict(lambda:{"support":0,"detected":0})
    for iid,label,probability in zip(ids,labels,probabilities):
        if label!=1: continue
        group="open_box" if by_id[iid]["phase8b_final_label"]=="open_box" else "ordinary_damaged"; groups[group]["support"]+=1; groups[group]["detected"]+=int(probability>=0.5)
    subgroups={k:{**v,"recall":v["detected"]/v["support"]} for k,v in groups.items()}
    with (output/"training_history.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(history[0])); writer.writeheader(); writer.writerows(history)
    with (output/"validation_predictions.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.writer(handle); writer.writerow(["image_id","true_label","damaged_probability","threshold","predicted_label","subgroup"])
        for iid,label,probability in zip(ids,labels,probabilities):
            group="open_box" if by_id[iid]["phase8b_final_label"]=="open_box" else "ordinary_damaged" if label==1 else "intact"; writer.writerow([iid,int(label),float(probability),0.5,int(probability>=0.5),group])
    metadata={"phase":"10A","experiment":"partial_fine_tuning","architecture":args.architecture,"initial_checkpoint":experiment_a.as_posix(),"initial_checkpoint_sha256":sha256(experiment_a),"dataset":"parcel_binary_v2","splits_used":["train","valid"],"test_accessed":False,"input":"224x224 RGB after resize 256 and center crop","augmentation":"none","normalization":"ImageNet mean/std","fine_tuned_parts":explanation,"trainable_prefixes":prefixes,"frozen_batchnorm_policy":"Frozen blocks eval; trainable late-block BN affine parameters and running statistics update deliberately","seed":seed,"epochs":args.epochs,"batch_size":args.batch_size,"optimizer":"Adam with two parameter groups","backbone_learning_rate":backbone_lr,"head_learning_rate":head_lr,"loss":"BCEWithLogitsLoss","checkpoint_rule":"minimum validation loss","best_epoch":best_epoch,"best_validation_loss":best_loss,"validation_threshold":0.5,"validation_metrics":validation_metrics,"subgroup_results":subgroups,"total_parameters":total,"frozen_parameters":frozen,"trainable_parameters":trainable,"percent_trainable":percent,"trainable_parameter_names":trainable_names,"checkpoint_size_mib":(output/"best_model.pt").stat().st_size/1024**2,"training_seconds":training_seconds,"validation_inference_ms_per_image":latency,"torch":torch.__version__,"torchvision":torchvision.__version__,"python":sys.version,"platform":platform.platform(),"device":str(device),"gpu":torch.cuda.get_device_name(0) if device.type=="cuda" else None}
    (output/"experiment_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    cm=np.asarray(validation_metrics["confusion_matrix"]); fig,ax=plt.subplots(figsize=(5,4)); im=ax.imshow(cm,cmap="Blues"); ax.set(xticks=[0,1],yticks=[0,1],xticklabels=["intact","damaged"],yticklabels=["intact","damaged"],xlabel="Predicted",ylabel="True",title=f"{args.architecture} partial fine-tune");
    for i in range(2):
        for j in range(2): ax.text(j,i,str(cm[i,j]),ha="center",va="center",color="white" if cm[i,j]>cm.max()/2 else "black")
    fig.colorbar(im,ax=ax); fig.tight_layout(); fig.savefig(output/"confusion_matrix.png",dpi=160); plt.close(fig)
    epochs=[r["epoch"] for r in history]; fig,axes=plt.subplots(1,2,figsize=(11,4)); axes[0].plot(epochs,[r["train_loss"] for r in history],label="Train"); axes[0].plot(epochs,[r["validation_loss"] for r in history],label="Validation"); axes[0].axvline(best_epoch,color="gray",linestyle="--",label=f"Best {best_epoch}"); axes[0].set(title="Loss",xlabel="Epoch"); axes[0].legend(); axes[0].grid(alpha=.25); axes[1].plot(epochs,[r["validation_damaged_recall"] for r in history],label="Recall"); axes[1].plot(epochs,[r["validation_specificity"] for r in history],label="Specificity"); axes[1].set(title="Validation at threshold 0.5",xlabel="Epoch"); axes[1].legend(); axes[1].grid(alpha=.25); fig.tight_layout(); fig.savefig(output/"training_curves.png",dpi=160); plt.close(fig)
    print("\nFine-tuning complete."); print(f"Best epoch: {best_epoch}\nRecall: {validation_metrics['damaged_recall']:.1%}\nSpecificity: {validation_metrics['specificity']:.1%}\nLatency: {latency:.3f} ms/image\nTest accessed: no")


if __name__=="__main__": main()
