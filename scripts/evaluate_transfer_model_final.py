"""Phase 11: one-time final test evaluation of transfer-mobile-v1.

# ============================================================
# What are we doing in this script?
# ============================================================
#
# Phase 10D froze one complete inference system: MobileNetV3-Large checkpoint,
# RGB/ImageNet preprocessing, and probability threshold 0.434584259987.
# This script first checks that every frozen input is unchanged. Only after all
# checks pass does it open test images and calculate the one-time final metrics.
#
# We are NOT training, recalibrating, changing labels, choosing another model,
# or updating weights from test data. Test results describe generalization; they
# are not development feedback for this same frozen system.
"""

from __future__ import annotations

import argparse, csv, hashlib, json, math, platform, sys, time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import torchvision
from PIL import Image, ImageDraw, ImageOps
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, precision_recall_curve, precision_recall_fscore_support, roc_auc_score, roc_curve
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


EXPECTED_CHECKPOINT_SHA256 = "f255600ad26289e7a3cc3b277926ffd91c8384a013a4203808b12d4ea77629d1"
EXPECTED_MANIFEST_SHA256 = "79f4307694e5a9415523e72dfe15af7720af566f32af907db767a89a8ea87284"
EXPECTED_THRESHOLD = 0.43458425998687744
RECALL_TARGET = 0.90
EXPECTED_TEST_IMAGES = 592


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    """Calculate a practical 95% binomial confidence interval."""
    if total == 0: return [0.0, 0.0]
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    half_width = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - half_width), min(1.0, center + half_width)]


# ---------------------------------------------------------
# 1. Test Dataset and deterministic preprocessing
# ---------------------------------------------------------
class TestParcelDataset(Dataset):
    # A Dataset loads one sample. It never changes model weights.
    def __init__(self, root: Path, rows: list[dict[str, str]], transform) -> None:
        self.root, self.rows, self.transform = root, rows, transform

    def __len__(self) -> int: return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(self.root / row["v2_relative_path"]) as image:
            image_tensor = self.transform(image.convert("RGB"))
        label = torch.tensor(1.0 if row["v2_binary_class"] == "damaged" else 0.0)
        return image_tensor, label, row["image_id"]


def create_transform():
    # This exactly matches frozen validation preprocessing. No random transform
    # is allowed because predictions must be deterministic and comparable.
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])


def build_frozen_model(checkpoint_path: Path):
    # weights=None avoids downloading or silently substituting another weight
    # version. The saved checkpoint contains the complete frozen model state.
    model = models.mobilenet_v3_large(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 1)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def calculate_metrics(labels, probabilities, predictions):
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0,1]).ravel()
    specificity = tn / (tn + fp)
    return {"accuracy":float(accuracy_score(labels,predictions)),"damaged_precision":float(precision),
            "damaged_recall":float(recall),"damaged_recall_95pct_wilson_ci":wilson_interval(int(tp),int(tp+fn)),
            "damaged_f1":float(f1),"specificity":float(specificity),
            "specificity_95pct_wilson_ci":wilson_interval(int(tn),int(tn+fp)),
            "balanced_accuracy":float((recall+specificity)/2),"roc_auc":float(roc_auc_score(labels,probabilities)),
            "pr_auc":float(average_precision_score(labels,probabilities)),"tp":int(tp),"tn":int(tn),"fp":int(fp),"fn":int(fn),
            "confusion_matrix":[[int(tn),int(fp)],[int(fn),int(tp)]]}


def save_error_sheet(rows: list[dict], dataset_root: Path, output: Path, title: str) -> None:
    # Contact sheets are created only after permanent metrics/predictions exist.
    shown=rows[:50]; columns=5; cell_w,cell_h=240,215; row_count=max(1,math.ceil(len(shown)/columns))
    canvas=Image.new("RGB",(columns*cell_w,row_count*cell_h),"white"); draw=ImageDraw.Draw(canvas); draw.text((5,3),title,fill="black")
    for index,row in enumerate(shown):
        with Image.open(dataset_root/row["relative_path"]) as image:
            thumb=ImageOps.contain(image.convert("RGB"),(cell_w-8,165),Image.Resampling.LANCZOS)
        x=(index%columns)*cell_w; y=(index//columns)*cell_h; canvas.paste(thumb,(x+(cell_w-thumb.width)//2,y+18)); draw.rectangle((x,y,x+cell_w-1,y+cell_h-1),outline="#777")
        draw.text((x+4,y+185),f'{row["image_id"]} {row["subgroup"]}',fill="black"); draw.text((x+4,y+199),f'p={float(row["damaged_probability"]):.3f}',fill="black")
    canvas.save(output,quality=92)


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--dataset",type=Path,default=Path("datasets/processed/parcel_binary_v2")); parser.add_argument("--freeze-manifest",type=Path,default=Path("models/selected_transfer_model/selected_transfer_model.json")); parser.add_argument("--output",type=Path,default=Path("models/selected_transfer_model")); parser.add_argument("--batch-size",type=int,default=32); args=parser.parse_args()
    required_outputs=["final_test_metrics.json","final_test_predictions.csv","final_confusion_matrix.png","final_roc_curve.png","final_pr_curve.png","final_subgroup_results.json","final_integrity_check.json","final_reproducibility_metadata.json"]
    existing=[name for name in required_outputs if (args.output/name).exists()]
    if existing: raise FileExistsError(f"One-time final artifacts already exist; refusing overwrite: {existing}")

    freeze=json.loads(args.freeze_manifest.read_text(encoding="utf-8")); checkpoint=Path(freeze["checkpoint_path"]); manifest_path=args.dataset/"manifests/dataset_manifest.csv"; audit_path=args.dataset/"manifests/audit_summary.json"
    with manifest_path.open(newline="",encoding="utf-8") as handle: rows=list(csv.DictReader(handle))
    audit=json.loads(audit_path.read_text(encoding="utf-8"))

    # ---------------------------------------------------------
    # 2. Integrity checks BEFORE any test image is decoded
    # ---------------------------------------------------------
    listed={row["v2_relative_path"] for row in rows}; actual={p.relative_to(args.dataset).as_posix() for split in ("train","valid","test") for label in ("damaged","intact") for p in (args.dataset/split/label).iterdir() if p.is_file()}
    group_splits=defaultdict(set); hash_splits=defaultdict(set); changed=[]
    for row in rows:
        group_splits[row["group_id"]].add(row["v2_split"]); hash_splits[row["sha256"]].add(row["v2_split"])
        path=args.dataset/row["v2_relative_path"]
        if not path.is_file() or sha256(path)!=row["sha256"]: changed.append(row["v2_relative_path"])
    test_rows=[row for row in rows if row["v2_split"]=="test"]
    checks={"checkpoint_hash_matches":sha256(checkpoint)==EXPECTED_CHECKPOINT_SHA256==freeze["checkpoint_sha256"],
            "dataset_version_matches":freeze["dataset_version"]=="parcel_binary_v2" and audit["dataset_version"]=="parcel_binary_v2",
            "manifest_hash_matches_frozen_record":sha256(manifest_path)==EXPECTED_MANIFEST_SHA256,
            "expected_test_size":len(test_rows)==EXPECTED_TEST_IMAGES,
            "file_inventory_exact":listed==actual,"all_file_hashes_match_manifest":not changed,
            "no_groups_cross_splits":not any(len(value)>1 for value in group_splits.values()),
            "no_exact_hashes_cross_splits":not any(len(value)>1 for value in hash_splits.values()),
            "phase8c_audit_frozen":audit["frozen"] is True,
            "threshold_matches_freeze":abs(float(freeze["frozen_threshold"])-EXPECTED_THRESHOLD)<5e-13,
            "model_status_frozen_before_test":freeze["status"]=="frozen_before_test" and freeze["test_accessed"] is False}
    integrity={"passed":all(checks.values()),"checked_before_test_image_access":True,"checks":checks,
               "checkpoint_sha256":sha256(checkpoint),"dataset_manifest_sha256":sha256(manifest_path),"freeze_manifest_sha256":sha256(args.freeze_manifest),
               "expected_test_images":EXPECTED_TEST_IMAGES,"observed_test_images":len(test_rows),"added_files":sorted(actual-listed),"missing_files":sorted(listed-actual),"modified_files":changed,"timestamp_utc":datetime.now(timezone.utc).isoformat()}
    (args.output/"final_integrity_check.json").write_text(json.dumps(integrity,indent=2),encoding="utf-8")
    if not integrity["passed"]: raise RuntimeError(f"Integrity failed before test inference: {checks}")

    # ---------------------------------------------------------
    # 3. One-time frozen test inference
    # ---------------------------------------------------------
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=build_frozen_model(checkpoint).to(device); model.eval()
    # model.eval() disables Dropout and uses fixed BatchNorm statistics. Test
    # data must not update model weights or running statistics.
    loader=DataLoader(TestParcelDataset(args.dataset,test_rows,create_transform()),batch_size=args.batch_size,shuffle=False,num_workers=0)
    labels_all=[]; probabilities_all=[]; ids_all=[]; forward_seconds=0.0
    print(f"Device: {device}")
    print("Expected shapes: image [3,224,224], batch [B,3,224,224], logits [B]")
    with torch.no_grad():
        # no_grad() disables gradient tracking because evaluation does not learn.
        # This saves memory and guarantees there is no backpropagation/update.
        for images,labels,image_ids in loader:
            images=images.to(device)
            if device.type=="cuda": torch.cuda.synchronize()
            started=time.perf_counter()
            logits=model(images).squeeze(1)
            if device.type=="cuda": torch.cuda.synchronize()
            forward_seconds+=time.perf_counter()-started
            # A logit is an unrestricted raw score. Sigmoid converts it to a
            # probability from 0 to 1 before applying the frozen threshold.
            probabilities=torch.sigmoid(logits)
            labels_all.extend(labels.numpy()); probabilities_all.extend(probabilities.cpu().numpy()); ids_all.extend(image_ids)
    labels=np.asarray(labels_all,dtype=np.uint8); probabilities=np.asarray(probabilities_all); predictions=probabilities>=EXPECTED_THRESHOLD
    metrics=calculate_metrics(labels,probabilities,predictions); metrics.update({"model_version":"transfer-mobile-v1","dataset":"parcel_binary_v2","test_images":len(labels),"frozen_threshold":EXPECTED_THRESHOLD,"recall_target":RECALL_TARGET,"recall_target_achieved":metrics["damaged_recall"]>=RECALL_TARGET,"average_forward_latency_ms_per_image":1000*forward_seconds/len(labels),"forward_throughput_images_per_second":len(labels)/forward_seconds,"checkpoint_size_mib":checkpoint.stat().st_size/1024**2,"total_parameters":freeze["total_parameters"],"gpu":torch.cuda.get_device_name(0) if device.type=="cuda" else None,"test_driven_modification":False})
    (args.output/"final_test_metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")

    by_id={row["image_id"]:row for row in test_rows}; prediction_rows=[]
    for iid,label,probability,prediction in zip(ids_all,labels,probabilities,predictions):
        row=by_id[iid]; subgroup="open_box" if row["phase8b_final_label"]=="open_box" else "ordinary_damaged" if label==1 else "intact"
        prediction_rows.append({"image_id":iid,"relative_path":row["v2_relative_path"],"group_id":row["group_id"],"true_label":"damaged" if label else "intact","damaged_probability":float(probability),"frozen_threshold":EXPECTED_THRESHOLD,"predicted_label":"damaged" if prediction else "intact","correct":bool(prediction==bool(label)),"subgroup":subgroup,"open_box":row["phase8b_final_label"]=="open_box"})
    with (args.output/"final_test_predictions.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(prediction_rows[0])); writer.writeheader(); writer.writerows(prediction_rows)

    subgroup_results={}
    for subgroup in ("ordinary_damaged","open_box"):
        selected=[row for row in prediction_rows if row["subgroup"]==subgroup]; detected=sum(row["predicted_label"]=="damaged" for row in selected)
        subgroup_results[subgroup]={"support":len(selected),"detected":detected,"false_negatives":len(selected)-detected,"recall":detected/len(selected),"recall_95pct_wilson_ci":wilson_interval(detected,len(selected))}
    (args.output/"final_subgroup_results.json").write_text(json.dumps(subgroup_results,indent=2),encoding="utf-8")

    # Primary metrics and predictions are permanent before plots/error inspection.
    cm=np.asarray(metrics["confusion_matrix"]); fig,axis=plt.subplots(figsize=(5,4)); image=axis.imshow(cm,cmap="Blues"); axis.set(xticks=[0,1],yticks=[0,1],xticklabels=["intact","damaged"],yticklabels=["intact","damaged"],xlabel="Predicted",ylabel="True",title="transfer-mobile-v1 final test");
    for i in range(2):
        for j in range(2): axis.text(j,i,str(cm[i,j]),ha="center",va="center",color="white" if cm[i,j]>cm.max()/2 else "black")
    fig.colorbar(image,ax=axis); fig.tight_layout(); fig.savefig(args.output/"final_confusion_matrix.png",dpi=160); plt.close(fig)
    fpr,tpr,_=roc_curve(labels,probabilities); fig,axis=plt.subplots(figsize=(5,4)); axis.plot(fpr,tpr,label=f"AUC={metrics['roc_auc']:.3f}"); axis.plot([0,1],[0,1],"--",color="gray"); axis.set(title="Final test ROC",xlabel="False-positive rate",ylabel="True-positive rate"); axis.grid(alpha=.25); axis.legend(); fig.tight_layout(); fig.savefig(args.output/"final_roc_curve.png",dpi=160); plt.close(fig)
    precision_values,recall_values,_=precision_recall_curve(labels,probabilities); fig,axis=plt.subplots(figsize=(5,4)); axis.plot(recall_values,precision_values,label=f"AP={metrics['pr_auc']:.3f}"); axis.set(title="Final test precision-recall",xlabel="Recall",ylabel="Precision"); axis.grid(alpha=.25); axis.legend(); fig.tight_layout(); fig.savefig(args.output/"final_pr_curve.png",dpi=160); plt.close(fig)
    false_negatives=sorted([row for row in prediction_rows if row["true_label"]=="damaged" and not row["correct"]],key=lambda row:row["damaged_probability"],reverse=True)
    false_positives=sorted([row for row in prediction_rows if row["true_label"]=="intact" and not row["correct"]],key=lambda row:row["damaged_probability"])
    save_error_sheet(false_negatives,args.dataset,args.output/"final_false_negative_contact_sheet.jpg","Final false negatives (closest to threshold first)")
    save_error_sheet(false_positives,args.dataset,args.output/"final_false_positive_contact_sheet.jpg","Final false positives (closest to threshold first)")
    reproducibility={"one_time_final_evaluation":True,"model_version":"transfer-mobile-v1","timestamp_utc":datetime.now(timezone.utc).isoformat(),"python":sys.version,"platform":platform.platform(),"torch":torch.__version__,"torchvision":torchvision.__version__,"numpy":np.__version__,"pillow":PIL.__version__,"device":str(device),"gpu":torch.cuda.get_device_name(0) if device.type=="cuda" else None,"input_shape":[args.batch_size,3,224,224],"model_retrained":False,"weights_updated":False,"threshold_changed":False,"preprocessing_changed":False,"test_driven_modification":False}
    (args.output/"final_reproducibility_metadata.json").write_text(json.dumps(reproducibility,indent=2),encoding="utf-8")
    print(json.dumps({"integrity_passed":True,"metrics":metrics,"subgroups":subgroup_results},indent=2))


if __name__=="__main__": main()
