"""One-time Phase 8F evaluation of the frozen HOG+LinearSVC system."""

from __future__ import annotations

import argparse, csv, hashlib, json, math, platform, sys, time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import joblib, matplotlib, numpy as np, sklearn, skimage, PIL
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageOps
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from skimage.feature import hog

THRESHOLD = -0.462288626498
TARGET_RECALL = 0.90
IMAGE_SIZE = (128, 128)
HOG_PARAMS = {"orientations": 9, "pixels_per_cell": (8, 8), "cells_per_block": (2, 2), "block_norm": "L2-Hys"}


def sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if not total: return [0.0, 0.0]
    p=successes/total; d=1+z*z/total; c=(p+z*z/(2*total))/d; h=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/d
    return [max(0.0,c-h),min(1.0,c+h)]


def extract(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        gray=ImageOps.grayscale(image); gray=ImageOps.fit(gray,IMAGE_SIZE,method=Image.Resampling.LANCZOS); array=np.asarray(gray,dtype=np.float32)/255.0
    vector=hog(array,feature_vector=True,**HOG_PARAMS).astype(np.float32)
    if vector.shape != (8100,) or not np.isfinite(vector).all(): raise ValueError(f"Invalid HOG vector: {vector.shape}")
    return vector


def contact_sheet(rows: list[dict], dataset: Path, output: Path, title: str) -> None:
    cell_w,cell_h,cols=240,215,5; shown=rows[:50]; nrows=max(1,math.ceil(len(shown)/cols)); canvas=Image.new("RGB",(cell_w*cols,cell_h*nrows),"white"); draw=ImageDraw.Draw(canvas); draw.text((5,3),title,fill="black")
    for n,row in enumerate(shown):
        with Image.open(dataset/row["relative_path"]) as im: thumb=ImageOps.contain(im.convert("RGB"),(cell_w-8,165),Image.Resampling.LANCZOS)
        x=(n%cols)*cell_w; y=(n//cols)*cell_h; canvas.paste(thumb,(x+(cell_w-thumb.width)//2,y+18)); draw.rectangle((x,y,x+cell_w-1,y+cell_h-1),outline="#777"); draw.text((x+4,y+185),f'{row["image_id"]} {row["subgroup"]}',fill="black"); draw.text((x+4,y+199),f'score={float(row["decision_score"]):.3f}',fill="black")
    canvas.save(output,quality=92)


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--dataset",type=Path,default=Path("datasets/processed/parcel_binary_v2")); p.add_argument("--model",type=Path,default=Path("models/hog_svm_phase8d_v2.joblib")); p.add_argument("--threshold-artifact",type=Path,default=Path("models/phase8e_artifacts/selected_threshold.json")); p.add_argument("--output",type=Path,default=Path("models/phase8f_artifacts")); a=p.parse_args()
    if a.output.exists(): raise FileExistsError(f"One-time final output already exists; refusing overwrite: {a.output}")
    a.output.mkdir(parents=True)
    try:
        manifest_path=a.dataset/"manifests/dataset_manifest.csv"; audit_path=a.dataset/"manifests/audit_summary.json"; threshold_data=json.loads(a.threshold_artifact.read_text(encoding="utf-8")); audit=json.loads(audit_path.read_text(encoding="utf-8"))
        with manifest_path.open(newline="",encoding="utf-8") as h: rows=list(csv.DictReader(h))
        # Integrity is completed before any image decoding/feature extraction.
        listed={r["v2_relative_path"] for r in rows}; actual={p.relative_to(a.dataset).as_posix() for s in ("train","valid","test") for c in ("damaged","intact") for p in (a.dataset/s/c).iterdir() if p.is_file()}
        group_splits: dict[str,set[str]]=defaultdict(set); hash_splits: dict[str,set[str]]=defaultdict(set); changed=[]
        for r in rows:
            group_splits[r["group_id"]].add(r["v2_split"]); hash_splits[r["sha256"]].add(r["v2_split"])
            path=a.dataset/r["v2_relative_path"]
            if not path.is_file() or sha(path)!=r["sha256"]: changed.append(r["v2_relative_path"])
        test_rows=[r for r in rows if r["v2_split"]=="test"]
        model_hash=sha(a.model); checks={
            "phase8c_audit_frozen_true":audit.get("frozen") is True,"phase8c_expected_total":audit.get("included_images")==3949,
            "manifest_rows_3949":len(rows)==3949,"test_rows_592":len(test_rows)==592,"file_inventory_exact":listed==actual,
            "all_manifest_file_hashes_match":not changed,"no_groups_cross_splits":not any(len(v)>1 for v in group_splits.values()),
            "no_exact_hashes_cross_splits":not any(len(v)>1 for v in hash_splits.values()),
            "model_hash_matches_phase8e":model_hash==threshold_data["selected_model_sha256"],
            "threshold_matches_frozen":abs(float(threshold_data["selected_threshold"])-THRESHOLD)<5e-13,
            "threshold_frozen_before_test":threshold_data.get("threshold_frozen_before_test") is True and threshold_data.get("test_accessed") is False,
        }
        integrity={"passed":all(checks.values()),"checks":checks,"manifest_sha256_at_final_evaluation":sha(manifest_path),"phase8c_audit_sha256":sha(audit_path),"phase8d_model_sha256":model_hash,"phase8e_threshold_artifact_sha256_at_final_evaluation":sha(a.threshold_artifact),"expected_test_images":592,"observed_test_images":len(test_rows),"added_files":sorted(actual-listed),"missing_files":sorted(listed-actual),"modified_files":changed,"checked_before_test_image_access":True,"timestamp_utc":datetime.now(timezone.utc).isoformat()}
        (a.output/"integrity_check.json").write_text(json.dumps(integrity,indent=2),encoding="utf-8")
        if not integrity["passed"]: raise RuntimeError(f"Integrity failed before evaluation: {checks}")
        bundle=joblib.load(a.model)
        if bundle["selected_config"]!={"C":0.0001,"class_weight":None} or bundle.get("threshold_calibrated") is not False: raise RuntimeError("Frozen model bundle configuration mismatch")
        started=time.perf_counter(); features=[]
        for r in test_rows: features.append(extract(a.dataset/r["v2_relative_path"]))
        X=np.stack(features); y=np.asarray([r["v2_binary_class"]=="damaged" for r in test_rows],dtype=np.uint8); scores=bundle["pipeline"].decision_function(X); pred=scores>=THRESHOLD
        tn=int(((y==0)&~pred).sum()); fp=int(((y==0)&pred).sum()); fn=int(((y==1)&~pred).sum()); tp=int(((y==1)&pred).sum()); safe=lambda n,d:float(n/d) if d else 0.0
        recall=safe(tp,tp+fn); specificity=safe(tn,tn+fp); precision=safe(tp,tp+fp); f1=safe(2*precision*recall,precision+recall); auc=float(roc_auc_score(y,scores)); pr_auc=float(average_precision_score(y,scores))
        metrics={"dataset_version":"parcel_binary_v2","test_images":len(y),"frozen_threshold":THRESHOLD,"target_recall":TARGET_RECALL,"target_achieved":recall>=TARGET_RECALL,"tp":tp,"tn":tn,"fp":fp,"fn":fn,"confusion_matrix_intact_damaged":[[tn,fp],[fn,tp]],"accuracy":safe(tp+tn,len(y)),"damaged_precision":precision,"damaged_recall":recall,"damaged_recall_95pct_wilson_ci":wilson(tp,tp+fn),"damaged_f1":f1,"specificity":specificity,"specificity_95pct_wilson_ci":wilson(tn,tn+fp),"false_positive_rate":safe(fp,fp+tn),"false_negative_rate":safe(fn,fn+tp),"balanced_accuracy":(recall+specificity)/2,"roc_auc":auc,"pr_auc_average_precision":pr_auc,"evaluation_seconds":time.perf_counter()-started,"system_modified_after_test":False}
        (a.output/"final_test_metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
        prediction_rows=[]
        for i,r in enumerate(test_rows):
            subgroup="open_box" if r["phase8b_final_label"]=="open_box" else "ordinary_damaged" if y[i] else "intact"
            prediction_rows.append({"image_id":r["image_id"],"relative_path":r["v2_relative_path"],"group_id":r["group_id"],"true_label":"damaged" if y[i] else "intact","decision_score":float(scores[i]),"frozen_threshold":THRESHOLD,"predicted_label":"damaged" if pred[i] else "intact","correct":bool(pred[i]==bool(y[i])),"subgroup":subgroup,"open_box":r["phase8b_final_label"]=="open_box"})
        with (a.output/"final_test_predictions.csv").open("w",newline="",encoding="utf-8") as h:
            w=csv.DictWriter(h,fieldnames=list(prediction_rows[0])); w.writeheader(); w.writerows(prediction_rows)
        subgroup_results={}
        for name in ("ordinary_damaged","open_box"):
            subset=[r for r in prediction_rows if r["subgroup"]==name]; hits=sum(r["predicted_label"]=="damaged" for r in subset); subgroup_results[name]={"support":len(subset),"true_positives":hits,"false_negatives":len(subset)-hits,"recall":safe(hits,len(subset)),"recall_95pct_wilson_ci":wilson(hits,len(subset))}
        (a.output/"subgroup_results.json").write_text(json.dumps(subgroup_results,indent=2),encoding="utf-8")
        cm=np.asarray([[tn,fp],[fn,tp]]); fig,ax=plt.subplots(figsize=(5,4)); im=ax.imshow(cm,cmap="Blues"); ax.set(xticks=[0,1],yticks=[0,1],xticklabels=["intact","damaged"],yticklabels=["intact","damaged"],xlabel="Predicted",ylabel="True",title="Phase 8F final test confusion matrix");
        for i in range(2):
            for j in range(2): ax.text(j,i,str(cm[i,j]),ha="center",va="center",color="white" if cm[i,j]>cm.max()/2 else "black")
        fig.colorbar(im,ax=ax); fig.tight_layout(); fig.savefig(a.output/"final_confusion_matrix.png",dpi=160); plt.close(fig)
        fpr,tpr,_=roc_curve(y,scores); fig,ax=plt.subplots(figsize=(5,4)); ax.plot(fpr,tpr,label=f"AUC={auc:.3f}"); ax.plot([0,1],[0,1],"--",color="gray"); ax.set(xlabel="False-positive rate",ylabel="True-positive rate",title="Phase 8F final test ROC"); ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(a.output/"final_roc_curve.png",dpi=160); plt.close(fig)
        pr,rc,_=precision_recall_curve(y,scores); fig,ax=plt.subplots(figsize=(5,4)); ax.plot(rc,pr,label=f"AP={pr_auc:.3f}"); ax.set(xlabel="Recall",ylabel="Precision",title="Phase 8F final test precision-recall curve"); ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(a.output/"final_pr_curve.png",dpi=160); plt.close(fig)
        fn_rows=sorted([r for r in prediction_rows if r["true_label"]=="damaged" and not r["correct"]],key=lambda r:r["decision_score"],reverse=True); fp_rows=sorted([r for r in prediction_rows if r["true_label"]=="intact" and not r["correct"]],key=lambda r:r["decision_score"])
        contact_sheet(fn_rows,a.dataset,a.output/"false_negative_contact_sheet.jpg","False negatives (closest to threshold first)"); contact_sheet(fp_rows,a.dataset,a.output/"false_positive_contact_sheet.jpg","False positives (closest to threshold first)")
        repro={"one_time_final_evaluation":True,"timestamp_utc":datetime.now(timezone.utc).isoformat(),"python":sys.version,"platform":platform.platform(),"numpy":np.__version__,"scikit_learn":sklearn.__version__,"scikit_image":skimage.__version__,"matplotlib":matplotlib.__version__,"pillow":PIL.__version__,"hog":{"image_size":IMAGE_SIZE,**HOG_PARAMS},"feature_shape":list(X.shape),"model_retrained":False,"scaler_refit":False,"threshold_recalibrated":False,"test_driven_modification":False}
        (a.output/"reproducibility_metadata.json").write_text(json.dumps(repro,indent=2),encoding="utf-8")
        print(json.dumps({"integrity":integrity["passed"],"metrics":metrics,"subgroups":subgroup_results},indent=2))
    except Exception:
        # Preserve a failed pre-evaluation integrity record, but never retain a partial scored evaluation.
        if (a.output/"final_test_metrics.json").exists(): raise
        for path in a.output.iterdir():
            if path.name!="integrity_check.json": path.unlink()
        raise


if __name__=="__main__": main()
