"""Phase 8D controlled LinearSVC selection using train/validation only."""

from __future__ import annotations

import argparse, csv, json, platform, sys, time
from pathlib import Path
import joblib, matplotlib, numpy as np, sklearn, skimage
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def evaluate(y, pred, score):
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", pos_label=1, zero_division=0)
    cm = confusion_matrix(y, pred, labels=[0, 1]); tn, fp, fn, tp = cm.ravel()
    return {"accuracy": float(accuracy_score(y, pred)), "damaged_precision": float(p), "damaged_recall": float(r),
            "damaged_f1": float(f), "intact_recall_specificity": float(tn/(tn+fp)), "roc_auc": float(roc_auc_score(y, score)),
            "confusion_matrix_intact_damaged": cm.tolist(), "true_negatives": int(tn), "false_positives": int(fp), "false_negatives": int(fn), "true_positives": int(tp)}


def main():
    p = argparse.ArgumentParser(); p.add_argument("--features", type=Path, default=Path("models/phase8d_artifacts/features")); p.add_argument("--output", type=Path, default=Path("models/phase8d_artifacts")); p.add_argument("--model", type=Path, default=Path("models/hog_svm_phase8d_v2.joblib")); a=p.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    tr=np.load(a.features/"hog_train.npz"); va=np.load(a.features/"hog_valid.npz"); X,y=tr["X"],tr["y"]; Xv,yv=va["X"],va["y"]
    candidates=[]
    for weight in (None,"balanced"):
        for C in (0.0001,0.001,0.01,0.1):
            pipe=Pipeline([("scale",StandardScaler(with_mean=False)),("svm",LinearSVC(C=C,class_weight=weight,dual="auto",max_iter=20000,random_state=42))])
            started=time.perf_counter(); pipe.fit(X,y); elapsed=time.perf_counter()-started
            pred=pipe.predict(Xv); score=pipe.decision_function(Xv)
            result={"C":C,"class_weight":weight,"training_seconds":elapsed,**evaluate(yv,pred,score)}; candidates.append((result,pipe,pred,score)); print(json.dumps(result))
    result,model,pred,score=max(candidates,key=lambda z:(z[0]["damaged_recall"],z[0]["damaged_f1"],z[0]["damaged_precision"],z[0]["roc_auc"],-z[0]["C"],z[0]["class_weight"] is None))
    subgroup={}
    adj=va["adjudications"]
    for name,mask in (("ordinary_damaged",(yv==1)&(adj!="open_box")),("open_box",(yv==1)&(adj=="open_box"))):
        n=int(mask.sum()); subgroup[name]={"support":n,"recall":float((pred[mask]==1).mean()) if n else None,"false_negatives":int((pred[mask]==0).sum())}
    payload={"selection_policy":"maximize damaged recall, then F1, precision, ROC-AUC, then lower C; train/validation only", "test_opened":False,"selected":result,"subgroup_validation":subgroup,"candidates":[z[0] for z in candidates]}
    (a.output/"validation_results.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    with (a.output/"model_selection.csv").open("w",newline="",encoding="utf-8") as h:
        fields=list(candidates[0][0]); w=csv.DictWriter(h,fieldnames=fields); w.writeheader(); w.writerows(z[0] for z in candidates)
    with (a.output/"validation_predictions.csv").open("w",newline="",encoding="utf-8") as h:
        w=csv.writer(h); w.writerow(["image_id","true_label","predicted_label","decision_score","final_adjudication"]); w.writerows(zip(va["image_ids"],yv,pred,score,adj))
    cm=np.asarray(result["confusion_matrix_intact_damaged"]); fig,ax=plt.subplots(figsize=(5,4)); im=ax.imshow(cm,cmap="Blues"); ax.set(xticks=[0,1],yticks=[0,1],xticklabels=["intact","damaged"],yticklabels=["intact","damaged"],xlabel="Predicted",ylabel="True",title="Phase 8D validation confusion matrix");
    for i in range(2):
        for j in range(2): ax.text(j,i,str(cm[i,j]),ha="center",va="center",color="white" if cm[i,j]>cm.max()/2 else "black")
    fig.colorbar(im,ax=ax); fig.tight_layout(); fig.savefig(a.output/"validation_confusion_matrix.png",dpi=160); plt.close(fig)
    metadata={"dataset":"parcel_binary_v2","frozen_split":"train/valid only; test unopened","positive_class":"damaged","label_mapping":{"intact":0,"damaged":1},"selected_hyperparameters":{"C":result["C"],"class_weight":result["class_weight"]},"scaling":"StandardScaler(with_mean=False), fit on train only","random_state":42,"python":sys.version,"platform":platform.platform(),"numpy":np.__version__,"scikit_learn":sklearn.__version__,"scikit_image":skimage.__version__,"matplotlib":matplotlib.__version__}
    bundle={"pipeline":model,"selected_config":metadata["selected_hyperparameters"],"validation_metrics":result,"hog_metadata":json.loads((a.features/"hog_metadata.json").read_text()),"reproducibility":metadata,"frozen_after_phase8d_validation":True,"threshold_calibrated":False}
    joblib.dump(bundle,a.model); (a.output/"reproducibility_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8"); print("SELECTED",json.dumps(result,indent=2)); print("SUBGROUP",json.dumps(subgroup,indent=2))


if __name__ == "__main__": main()
