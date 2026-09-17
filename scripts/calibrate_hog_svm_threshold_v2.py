"""Calibrate the frozen Phase 8D LinearSVC threshold on validation only."""

from __future__ import annotations

import argparse, csv, hashlib, json, platform, sys
from datetime import datetime, timezone
from pathlib import Path

import joblib, matplotlib, numpy as np, sklearn
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

TARGET_RECALL = 0.90
EXPECTED_CM = [[125, 75], [87, 305]]


def metric(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, object]:
    pred = score >= threshold
    tn = int(((y == 0) & ~pred).sum()); fp = int(((y == 0) & pred).sum())
    fn = int(((y == 1) & ~pred).sum()); tp = int(((y == 1) & pred).sum())
    safe = lambda n, d: float(n / d) if d else 0.0
    recall, specificity, precision = safe(tp, tp + fn), safe(tn, tn + fp), safe(tp, tp + fp)
    return {"threshold": float(threshold), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "accuracy": safe(tp + tn, len(y)), "damaged_precision": precision,
            "damaged_recall": recall, "damaged_f1": safe(2 * precision * recall, precision + recall),
            "specificity": specificity, "false_positive_rate": safe(fp, fp + tn),
            "false_negative_rate": safe(fn, fn + tp), "balanced_accuracy": (recall + specificity) / 2}


def subgroup(pred: np.ndarray, y: np.ndarray, adjudication: np.ndarray) -> dict[str, dict[str, object]]:
    result = {}
    for name, mask in (("ordinary_damaged", (y == 1) & (adjudication != "open_box")),
                       ("open_box", (y == 1) & (adjudication == "open_box"))):
        support = int(mask.sum()); fn = int((~pred[mask]).sum())
        result[name] = {"support": support, "recall": float((pred[mask]).mean()) if support else None, "false_negatives": fn}
    return result


def pct(value: float) -> str: return f"{100 * value:.2f}%"


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--model", type=Path, default=Path("models/hog_svm_phase8d_v2.joblib")); p.add_argument("--features", type=Path, default=Path("models/phase8d_artifacts/features/hog_valid.npz")); p.add_argument("--phase8d-predictions", type=Path, default=Path("models/phase8d_artifacts/validation_predictions.csv")); p.add_argument("--manifest", type=Path, default=Path("datasets/processed/parcel_binary_v2/manifests/dataset_manifest.csv")); p.add_argument("--output", type=Path, default=Path("models/phase8e_artifacts")); p.add_argument("--report", type=Path, default=Path("reports/PHASE-8E_threshold_calibration.md")); a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=True); a.report.parent.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(a.model)
    if bundle["selected_config"] != {"C": 0.0001, "class_weight": None}: raise RuntimeError("Unexpected Phase 8D model configuration")
    valid = np.load(a.features); y = valid["y"].astype(np.uint8); ids = valid["image_ids"].astype(str); adjudication = valid["adjudications"].astype(str)
    with a.phase8d_predictions.open(newline="", encoding="utf-8") as h: prior = {r["image_id"]: r for r in csv.DictReader(h)}
    if set(ids) != set(prior): raise RuntimeError("Phase 8D validation score IDs do not match validation features")
    scores = np.asarray([float(prior[i]["decision_score"]) for i in ids]); default = metric(y, scores, 0.0)
    if [[default["tn"], default["fp"]], [default["fn"], default["tp"]]] != EXPECTED_CM: raise RuntimeError(f"Threshold-0 reproduction failed: {default}")
    auc = float(roc_auc_score(y, scores))
    # Every observed score is an exact prediction transition; include zero for reproduction.
    thresholds = sorted(set(scores.tolist() + [0.0]))
    sweep = [metric(y, scores, t) for t in thresholds]
    feasible = [m for m in sweep if m["damaged_recall"] >= TARGET_RECALL]
    if not feasible: raise RuntimeError(f"No threshold satisfies recall target {TARGET_RECALL}")
    selected = max(feasible, key=lambda m: (m["specificity"], m["damaged_precision"], -abs(m["threshold"])))
    threshold = float(selected["threshold"]); pred0 = scores >= 0; pred = scores >= threshold
    subgroup0, subgroup_selected = subgroup(pred0, y, adjudication), subgroup(pred, y, adjudication)
    wanted = set(ids); metadata = {}
    with a.manifest.open(newline="", encoding="utf-8") as h:
        for row in csv.DictReader(h):
            iid = row["image_id"]
            if iid in wanted: metadata[iid] = {"path": row["v2_relative_path"], "group_id": row["group_id"], "final_adjudication": row["phase8b_final_label"]}
    if set(metadata) != wanted: raise RuntimeError("Missing validation metadata")
    with (a.output / "validation_predictions.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.writer(h); w.writerow(["image_id", "relative_path", "group_id", "true_binary_label", "decision_score", "final_adjudication", "is_open_box_mapped_to_damaged", "prediction_threshold_0", "prediction_calibrated"])
        for i, iid in enumerate(ids):
            m=metadata[iid]; w.writerow([iid,m["path"],m["group_id"],"damaged" if y[i] else "intact",scores[i],m["final_adjudication"],str(m["final_adjudication"]=="open_box").lower(),"damaged" if pred0[i] else "intact","damaged" if pred[i] else "intact"])
    fields = list(sweep[0]);
    with (a.output / "threshold_calibration.csv").open("w", newline="", encoding="utf-8") as h:
        w=csv.DictWriter(h,fieldnames=fields); w.writeheader(); w.writerows(sweep)
    selected_index = thresholds.index(threshold); lo=max(0,selected_index-5); hi=min(len(sweep),selected_index+6)
    with (a.output / "selected_threshold_neighborhood.csv").open("w",newline="",encoding="utf-8") as h:
        w=csv.DictWriter(h,fieldnames=fields); w.writeheader(); w.writerows(sweep[lo:hi])
    xs=np.asarray(thresholds); marks=(("Default",0.0,"#555555"),("Selected",threshold,"#d62728"))
    plots=(("recall_vs_threshold.png","Damaged recall","damaged_recall"),("specificity_vs_threshold.png","Specificity","specificity"),("precision_vs_threshold.png","Damaged precision","damaged_precision"))
    for filename,label,key in plots:
        fig,ax=plt.subplots(figsize=(7,4)); ax.plot(xs,[m[key] for m in sweep]);
        for name,x,c in marks: ax.axvline(x,color=c,linestyle="--",label=f"{name}: {x:.6g}")
        if key=="damaged_recall": ax.axhline(TARGET_RECALL,color="#2ca02c",linestyle=":",label=f"Target: {TARGET_RECALL:.0%}")
        ax.set(xlabel="Decision threshold",ylabel=label,title=f"Phase 8E validation: {label} vs threshold"); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(a.output/filename,dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4)); ax.plot(xs,[m["fp"] for m in sweep],label="False positives"); ax.plot(xs,[m["fn"] for m in sweep],label="False negatives");
    for name,x,c in marks: ax.axvline(x,color=c,linestyle="--",label=f"{name}: {x:.6g}")
    ax.set(xlabel="Decision threshold",ylabel="Count",title="Phase 8E validation error trade-off"); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(a.output/"fp_fn_vs_threshold.png",dpi=160); plt.close(fig)
    model_hash=hashlib.sha256(a.model.read_bytes()).hexdigest(); now=datetime.now(timezone.utc).isoformat()
    artifact={"phase":"8E","dataset_version":"parcel_binary_v2","selected_model_artifact":a.model.as_posix(),"selected_model_sha256":model_hash,"selected_threshold":threshold,"calibration_dataset":"validation only","test_accessed":False,"damaged_recall_target":TARGET_RECALL,"target_source":"Predefined in Phase 7/8C project guidance before Phase 8E","selection_criterion":"satisfy damaged recall target; maximize specificity; then precision; then threshold closest to zero","roc_auc":auc,"validation_metrics_threshold_0":default,"validation_metrics_calibrated":selected,"subgroup_threshold_0":subgroup0,"subgroup_calibrated":subgroup_selected,"timestamp_utc":now,"version":"phase8e-v1","threshold_frozen_before_test":True}
    (a.output/"selected_threshold.json").write_text(json.dumps(artifact,indent=2),encoding="utf-8")
    repro={"python":sys.version,"platform":platform.platform(),"numpy":np.__version__,"scikit_learn":sklearn.__version__,"matplotlib":matplotlib.__version__,"candidate_thresholds":len(sweep),"validation_samples":len(y),"validation_score_min":float(scores.min()),"validation_score_max":float(scores.max()),"test_accessed":False,"model_retrained":False}
    (a.output/"reproducibility_metadata.json").write_text(json.dumps(repro,indent=2),encoding="utf-8")
    cm0=f'[[{default["tn"]}, {default["fp"]}], [{default["fn"]}, {default["tp"]}]]'; cm=f'[[{selected["tn"]}, {selected["fp"]}], [{selected["fn"]}, {selected["tp"]}]]'
    report=f'''# Phase 8E — Validation-Only Operating Threshold Calibration\n\n## Outcome\n\nThe frozen Phase 8D model was calibrated on validation scores only. It was not retrained. The frozen test split was not loaded, featurized, scored, or evaluated.\n\nThe predefined damaged-recall target was **≥{TARGET_RECALL:.0%}**. No more specific criterion was encoded, so the documented fallback selected, among feasible thresholds, maximum specificity, then damaged precision, then proximity to zero. The frozen threshold is **`{threshold:.12g}`** and achieved the target.\n\n## Threshold-0 reproduction gate\n\n| Metric | Threshold 0 |\n|---|---:|\n| Accuracy | {pct(default['accuracy'])} |\n| Damaged precision | {pct(default['damaged_precision'])} |\n| Damaged recall | {pct(default['damaged_recall'])} |\n| Damaged F1 | {pct(default['damaged_f1'])} |\n| Specificity | {pct(default['specificity'])} |\n| ROC-AUC | {pct(auc)} |\n| FP / FN | {default['fp']} / {default['fn']} |\n| Confusion matrix | `{cm0}` |\n\nThis exactly reproduces the Phase 8D confusion matrix, so calibration proceeded. ROC-AUC remains {pct(auc)} because thresholding does not change score ranking.\n\n## Selected operating point\n\n| Metric | Threshold 0 | Calibrated `{threshold:.6g}` | Change |\n|---|---:|---:|---:|\n| Accuracy | {pct(default['accuracy'])} | {pct(selected['accuracy'])} | {100*(selected['accuracy']-default['accuracy']):+.2f} pp |\n| Damaged precision | {pct(default['damaged_precision'])} | {pct(selected['damaged_precision'])} | {100*(selected['damaged_precision']-default['damaged_precision']):+.2f} pp |\n| Damaged recall | {pct(default['damaged_recall'])} | **{pct(selected['damaged_recall'])}** | {100*(selected['damaged_recall']-default['damaged_recall']):+.2f} pp |\n| Damaged F1 | {pct(default['damaged_f1'])} | {pct(selected['damaged_f1'])} | {100*(selected['damaged_f1']-default['damaged_f1']):+.2f} pp |\n| Specificity | {pct(default['specificity'])} | **{pct(selected['specificity'])}** | {100*(selected['specificity']-default['specificity']):+.2f} pp |\n| False positives | {default['fp']} | **{selected['fp']}** | {selected['fp']-default['fp']:+d} |\n| False negatives | {default['fn']} | **{selected['fn']}** | {selected['fn']-default['fn']:+d} |\n| Confusion matrix | `{cm0}` | `{cm}` | — |\n\nThe recall requirement costs {selected['fp']-default['fp']} additional false positives while preventing {default['fn']-selected['fn']} damaged misses on validation.\n\n## Mandatory damaged-subgroup analysis\n\n| Subgroup | Support | Recall at 0 | Recall calibrated | Change | FN at 0 | FN calibrated |\n|---|---:|---:|---:|---:|---:|---:|\n| Ordinary damaged | {subgroup0['ordinary_damaged']['support']} | {pct(subgroup0['ordinary_damaged']['recall'])} | {pct(subgroup_selected['ordinary_damaged']['recall'])} | {100*(subgroup_selected['ordinary_damaged']['recall']-subgroup0['ordinary_damaged']['recall']):+.2f} pp | {subgroup0['ordinary_damaged']['false_negatives']} | {subgroup_selected['ordinary_damaged']['false_negatives']} |\n| Open box mapped to damaged | {subgroup0['open_box']['support']} | {pct(subgroup0['open_box']['recall'])} | {pct(subgroup_selected['open_box']['recall'])} | {100*(subgroup_selected['open_box']['recall']-subgroup0['open_box']['recall']):+.2f} pp | {subgroup0['open_box']['false_negatives']} | {subgroup_selected['open_box']['false_negatives']} |\n\nSubgroups were analyzed only after global threshold selection and did not tune the threshold.\n\n## Frozen artifacts\n\n- Full sweep: `models/phase8e_artifacts/threshold_calibration.csv`\n- Frozen threshold: `models/phase8e_artifacts/selected_threshold.json`\n- Enriched validation predictions: `models/phase8e_artifacts/validation_predictions.csv`\n- Selected-point neighborhood: `models/phase8e_artifacts/selected_threshold_neighborhood.csv`\n- Curves: recall, specificity, precision, and FP/FN under `models/phase8e_artifacts/`\n- Reproducibility metadata: `models/phase8e_artifacts/reproducibility_metadata.json`\n\nThe threshold is frozen before test evaluation and must not be changed after test results are seen. Phase 8E stops here.\n'''
    try:
        a.report.write_text(report, encoding="utf-8")
    except PermissionError:
        # Some managed workspaces allow generated model artifacts but reserve
        # report files for the workspace editor. Calibration remains complete.
        (a.output / "generated_report.md").write_text(report, encoding="utf-8")
        print(f"WARNING: report path not writable; generated copy saved under {a.output}")
    print(json.dumps(artifact,indent=2))


if __name__ == "__main__": main()
