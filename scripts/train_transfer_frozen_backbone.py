"""Train one ImageNet-pretrained model with its backbone frozen.

Examples:
  python scripts/train_transfer_frozen_backbone.py --architecture resnet50
  python scripts/train_transfer_frozen_backbone.py --architecture efficientnet_b0
  python scripts/train_transfer_frozen_backbone.py --architecture mobilenet_v3_large

The script uses train/validation only. It never constructs a test Dataset.
"""

from __future__ import annotations

import argparse, csv, json, platform, random, sys, time
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


# ---------------------------------------------------------
# 1. Reproducibility
# ---------------------------------------------------------
def set_seeds(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------
# 2. Dataset and ImageNet preprocessing
# ---------------------------------------------------------
class ParcelDataset(Dataset):
    """Load one RGB parcel image, binary label, and stable image ID."""
    def __init__(self, root: Path, rows: list[dict[str, str]], transform) -> None:
        self.root, self.rows, self.transform = root, rows, transform

    def __len__(self) -> int: return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(self.root / row["v2_relative_path"]) as image:
            image_tensor = self.transform(image.convert("RGB"))
        label = torch.tensor(1.0 if row["v2_binary_class"] == "damaged" else 0.0)
        return image_tensor, label, row["image_id"]


def imagenet_transform():
    # The official pretrained networks learned from ImageNet images prepared in
    # this way. Using the expected normalization keeps RGB values on the scale
    # that the pretrained filters saw during their original training.
    return transforms.Compose([
        transforms.Resize(256),
        # Resize the shorter side to 256 while preserving aspect ratio.
        transforms.CenterCrop(224),
        # All three backbones expect a 224x224 RGB input for this baseline.
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ---------------------------------------------------------
# 3. Pretrained model, frozen backbone, and new head
# ---------------------------------------------------------
def build_model(architecture: str):
    # Transfer learning starts from filters learned on ImageNet. These filters
    # already recognize broadly useful edges, textures, shapes, and object parts.
    if architecture == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        head_input_features = model.fc.in_features
    elif architecture == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=weights)
        head_input_features = model.classifier[1].in_features
    elif architecture == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT
        model = models.mobilenet_v3_large(weights=weights)
        head_input_features = model.classifier[3].in_features
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    # The backbone is the convolutional feature extractor. Setting
    # requires_grad=False means backward() will not calculate or store gradients
    # for those parameters, so pretrained backbone weights cannot change.
    for parameter in model.parameters():
        parameter.requires_grad = False

    # The classification head converts learned image features into one parcel
    # damage logit. This new layer defaults to requires_grad=True and is the only
    # part updated during Experiment A.
    if architecture == "resnet50": model.fc = nn.Linear(head_input_features, 1)
    elif architecture == "efficientnet_b0": model.classifier[1] = nn.Linear(head_input_features, 1)
    else: model.classifier[3] = nn.Linear(head_input_features, 1)
    return model, weights


def parameter_counts(model):
    total=sum(p.numel() for p in model.parameters()); trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, total-trainable, trainable


def metrics(labels, probabilities, threshold=0.5):
    predictions=probabilities>=threshold
    precision,recall,f1,_=precision_recall_fscore_support(labels,predictions,average="binary",zero_division=0)
    tn,fp,fn,tp=confusion_matrix(labels,predictions,labels=[0,1]).ravel(); specificity=tn/(tn+fp)
    return {"accuracy":float(accuracy_score(labels,predictions)),"damaged_precision":float(precision),
            "damaged_recall":float(recall),"damaged_f1":float(f1),"specificity":float(specificity),
            "balanced_accuracy":float((recall+specificity)/2),"roc_auc":float(roc_auc_score(labels,probabilities)),
            "pr_auc":float(average_precision_score(labels,probabilities)),"tp":int(tp),"tn":int(tn),
            "fp":int(fp),"fn":int(fn),"confusion_matrix":[[int(tn),int(fp)],[int(fn),int(tp)]]}


# ---------------------------------------------------------
# 4. Explicit head-training epoch
# ---------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.eval()
    # The entire frozen model stays in evaluation mode while the new linear head
    # learns. This is stricter than requires_grad=False alone: it also prevents
    # BatchNorm running statistics and stochastic backbone layers from changing.
    # nn.Linear behaves the same in train/eval mode, so its gradients and Adam
    # updates still work normally. The pretrained representation remains fixed.
    running_loss=0.0; labels_all=[]; probabilities_all=[]
    for images, labels, _ in loader:
        # 1-2. Load one [B,3,224,224] batch and move it to the model's device.
        images=images.to(device); labels=labels.to(device)
        # 3. Clear head gradients left from the previous batch.
        optimizer.zero_grad()
        # 4. Forward propagation: backbone features flow into the new head.
        # Output is one raw logit per image, shape [B].
        logits=model(images).squeeze(1)
        # 5. BCEWithLogitsLoss combines sigmoid and binary cross-entropy safely.
        # The model returns logits rather than applying sigmoid internally.
        loss=criterion(logits,labels)
        # 6. Backpropagation computes gradients for trainable head parameters.
        loss.backward()
        # 7. Adam updates only parameters included in the optimizer: the head.
        optimizer.step()
        running_loss+=loss.item()*images.size(0)
        probabilities=torch.sigmoid(logits.detach())
        labels_all.extend(labels.detach().cpu().numpy()); probabilities_all.extend(probabilities.cpu().numpy())
    return running_loss/len(loader.dataset),metrics(np.asarray(labels_all),np.asarray(probabilities_all))


# ---------------------------------------------------------
# 5. Validation: prediction without learning
# ---------------------------------------------------------
def validate(model, loader, criterion, device):
    model.eval()
    # eval() makes BatchNorm use learned running statistics and disables Dropout.
    running_loss=0.0; labels_all=[]; probabilities_all=[]; ids_all=[]
    with torch.no_grad():
        # Validation does not learn, so gradients would waste memory/computation.
        for images,labels,image_ids in loader:
            images=images.to(device); labels=labels.to(device)
            logits=model(images).squeeze(1); loss=criterion(logits,labels)
            # Sigmoid converts raw logits to damaged probabilities in [0,1].
            probabilities=torch.sigmoid(logits)
            running_loss+=loss.item()*images.size(0)
            labels_all.extend(labels.cpu().numpy()); probabilities_all.extend(probabilities.cpu().numpy()); ids_all.extend(image_ids)
    labels_array=np.asarray(labels_all); probabilities_array=np.asarray(probabilities_all)
    return running_loss/len(loader.dataset),metrics(labels_array,probabilities_array),ids_all,labels_array,probabilities_array


def measure_latency(model, loader, device):
    # Measure GPU forward-pass time only, after images have moved to GPU. CUDA is
    # asynchronous, so synchronize before and after timing each batch.
    model.eval(); total_seconds=0.0; total_images=0
    with torch.no_grad():
        for images,_,_ in loader:
            images=images.to(device)
            if device.type=="cuda": torch.cuda.synchronize()
            started=time.perf_counter(); model(images)
            if device.type=="cuda": torch.cuda.synchronize()
            total_seconds+=time.perf_counter()-started; total_images+=images.size(0)
    return 1000*total_seconds/total_images


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--architecture",required=True,choices=["resnet50","efficientnet_b0","mobilenet_v3_large"]); parser.add_argument("--dataset",type=Path,default=Path("datasets/processed/parcel_binary_v2")); parser.add_argument("--output-root",type=Path,default=Path("models/transfer_learning")); parser.add_argument("--epochs",type=int,default=10); parser.add_argument("--batch-size",type=int,default=32); args=parser.parse_args()
    output=args.output_root/f"{args.architecture}_frozen"
    if output.exists(): raise FileExistsError(f"Refusing to overwrite experiment: {output}")
    output.mkdir(parents=True)
    seed=42; learning_rate=1e-3; set_seeds(seed); torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f"Architecture: {args.architecture}\nDevice: {device}");
    if device.type=="cuda": print(f"GPU: {torch.cuda.get_device_name(0)}")

    with (args.dataset/"manifests/dataset_manifest.csv").open(newline="",encoding="utf-8") as handle:
        development=[r for r in csv.DictReader(handle) if r["v2_split"] in {"train","valid"}]
    train_rows=[r for r in development if r["v2_split"]=="train"]; valid_rows=[r for r in development if r["v2_split"]=="valid"]
    if len(train_rows)!=2765 or len(valid_rows)!=592: raise RuntimeError("Frozen development counts changed")
    if {r["group_id"] for r in train_rows}&{r["group_id"] for r in valid_rows}: raise RuntimeError("Group leakage")
    transform=imagenet_transform(); train_dataset=ParcelDataset(args.dataset,train_rows,transform); valid_dataset=ParcelDataset(args.dataset,valid_rows,transform)
    generator=torch.Generator().manual_seed(seed); train_loader=DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,num_workers=0,generator=generator); valid_loader=DataLoader(valid_dataset,batch_size=args.batch_size,shuffle=False,num_workers=0)

    model,weights=build_model(args.architecture); total,frozen,trainable=parameter_counts(model); model=model.to(device)
    print(f"Total parameters: {total:,}\nFrozen parameters: {frozen:,}\nTrainable head parameters: {trainable:,}")
    trainable_names=[name for name,p in model.named_parameters() if p.requires_grad]; print("Trainable tensors:",trainable_names)
    # Passing only requires_grad=True parameters makes it explicit that Adam can
    # update only the new classification head.
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=learning_rate)
    criterion=nn.BCEWithLogitsLoss(); history=[]; best_loss=float("inf"); best_epoch=0; started=time.perf_counter()
    for epoch in range(1,args.epochs+1):
        train_loss,train_metrics=train_one_epoch(model,train_loader,criterion,optimizer,device)
        valid_loss,valid_metrics,ids,labels,probabilities=validate(model,valid_loader,criterion,device)
        row={"epoch":epoch,"train_loss":train_loss,"validation_loss":valid_loss,**{f"train_{k}":v for k,v in train_metrics.items() if k!="confusion_matrix"},**{f"validation_{k}":v for k,v in valid_metrics.items() if k!="confusion_matrix"}}; history.append(row)
        print(f"Epoch {epoch:02d}/{args.epochs} | train loss {train_loss:.4f} | valid loss {valid_loss:.4f} | recall {valid_metrics['damaged_recall']:.2%} | specificity {valid_metrics['specificity']:.2%}")
        if valid_loss<best_loss:
            best_loss=valid_loss; best_epoch=epoch; torch.save({"architecture":args.architecture,"model_state_dict":model.state_dict(),"epoch":epoch,"validation_loss":valid_loss,"threshold":0.5,"backbone_frozen":True,"weights":str(weights)},output/"best_model.pt")

    training_seconds=time.perf_counter()-started; checkpoint=torch.load(output/"best_model.pt",map_location=device,weights_only=True); model.load_state_dict(checkpoint["model_state_dict"])
    validation_loss,validation_metrics,ids,labels,probabilities=validate(model,valid_loader,criterion,device); latency=measure_latency(model,valid_loader,device)
    by_id={r["image_id"]:r for r in valid_rows}; groups=defaultdict(lambda:{"support":0,"detected":0})
    for iid,label,probability in zip(ids,labels,probabilities):
        if label!=1: continue
        name="open_box" if by_id[iid]["phase8b_final_label"]=="open_box" else "ordinary_damaged"; groups[name]["support"]+=1; groups[name]["detected"]+=int(probability>=0.5)
    subgroup={k:{**v,"recall":v["detected"]/v["support"]} for k,v in groups.items()}
    with (output/"training_history.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(history[0])); writer.writeheader(); writer.writerows(history)
    with (output/"validation_predictions.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.writer(handle); writer.writerow(["image_id","true_label","damaged_probability","threshold","predicted_label","subgroup"])
        for iid,label,probability in zip(ids,labels,probabilities):
            group="open_box" if by_id[iid]["phase8b_final_label"]=="open_box" else "ordinary_damaged" if label==1 else "intact"; writer.writerow([iid,int(label),float(probability),0.5,int(probability>=0.5),group])
    model_size=(output/"best_model.pt").stat().st_size/1024**2
    metadata={"experiment":"transfer_learning_frozen_backbone","architecture":args.architecture,"pretrained_weights":str(weights),"dataset":"parcel_binary_v2","splits_used":["train","valid"],"test_accessed":False,"input":"224x224 RGB after resize-short-side 256 and center crop","augmentation":"none","normalization":"ImageNet mean/std","backbone_frozen":True,"trained_from_scratch":False,"seed":seed,"epochs":args.epochs,"batch_size":args.batch_size,"optimizer":"Adam","learning_rate":learning_rate,"loss":"BCEWithLogitsLoss","checkpoint_rule":"minimum validation loss","best_epoch":best_epoch,"best_validation_loss":best_loss,"validation_threshold":0.5,"validation_metrics":validation_metrics,"subgroup_results":subgroup,"total_parameters":total,"frozen_parameters":frozen,"trainable_parameters":trainable,"trainable_parameter_names":trainable_names,"checkpoint_size_mib":model_size,"training_seconds":training_seconds,"validation_inference_ms_per_image":latency,"torch":torch.__version__,"torchvision":torchvision.__version__,"python":sys.version,"platform":platform.platform(),"device":str(device),"gpu":torch.cuda.get_device_name(0) if device.type=="cuda" else None}
    (output/"experiment_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    cm=np.asarray(validation_metrics["confusion_matrix"]); fig,ax=plt.subplots(figsize=(5,4)); im=ax.imshow(cm,cmap="Blues"); ax.set(xticks=[0,1],yticks=[0,1],xticklabels=["intact","damaged"],yticklabels=["intact","damaged"],xlabel="Predicted",ylabel="True",title=f"{args.architecture} validation");
    for i in range(2):
        for j in range(2): ax.text(j,i,str(cm[i,j]),ha="center",va="center",color="white" if cm[i,j]>cm.max()/2 else "black")
    fig.colorbar(im,ax=ax); fig.tight_layout(); fig.savefig(output/"confusion_matrix.png",dpi=160); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4)); epoch_values=[r["epoch"] for r in history]; axes[0].plot(epoch_values,[r["train_loss"] for r in history],label="Train"); axes[0].plot(epoch_values,[r["validation_loss"] for r in history],label="Validation"); axes[0].axvline(best_epoch,color="gray",linestyle="--",label=f"Best {best_epoch}"); axes[0].set(title="Loss",xlabel="Epoch"); axes[0].legend(); axes[0].grid(alpha=.25); axes[1].plot(epoch_values,[r["validation_damaged_recall"] for r in history],label="Recall"); axes[1].plot(epoch_values,[r["validation_specificity"] for r in history],label="Specificity"); axes[1].set(title="Validation at threshold 0.5",xlabel="Epoch"); axes[1].legend(); axes[1].grid(alpha=.25); fig.tight_layout(); fig.savefig(output/"training_curves.png",dpi=160); plt.close(fig)
    print("\nTraining complete."); print(f"Best epoch: {best_epoch}\nValidation recall: {validation_metrics['damaged_recall']:.1%}\nValidation specificity: {validation_metrics['specificity']:.1%}\nInference latency: {latency:.3f} ms/image\nTest split accessed: no")


if __name__=="__main__": main()
