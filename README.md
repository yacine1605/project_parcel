# AI-Powered Parcel Damage Detection

A computer-vision quality-control case study for detecting damaged parcels in warehouse and logistics workflows.

The project combines two complementary decisions:

1. **Binary triage** — decide whether a parcel is acceptable or damaged.
2. **Damage localization** — locate visible defects such as holes, wet areas, compression, and minor damage.

## Why this project matters

Manual parcel inspection is slow and inconsistent. This system explores how lightweight computer-vision models can support warehouse operators with fast, repeatable screening while preserving human review for uncertain cases.

## Dataset engineering

- 2,944 images and 4,523 bounding boxes in the frozen object-detection V2 dataset
- duplicate removal across train, validation, and test
- annotation corrections and out-of-bounds-box checks
- targeted error analysis for the difficult `compressed` class
- frozen binary dataset with 3,949 images across 1,167 source groups
- zero group leakage and zero exact-hash cross-split leakage

## Object-detection results

The selected lightweight baseline is **YOLO11n**.

| Split | Precision | Recall | mAP@50 | mAP@50–95 |
|---|---:|---:|---:|---:|
| Validation | 0.573 | 0.626 | 0.620 | 0.310 |
| Held-out test | 0.563 | 0.584 | 0.571 | 0.284 |

Inference is approximately **2.1 ms per image** on the tested GPU. Holes and wet damage were the strongest classes; compressed parcels remained the principal failure mode.

## Classical binary baseline

| Metric | Held-out test |
|---|---:|
| Damaged recall | 71.84% |
| F1 score | 74.40% |
| ROC-AUC | 77.24% |

A validation-only operating-threshold calibration reached the predefined **90% damaged-recall target** without accessing the protected test split.

## Experimental discipline

- frozen dataset versions and protected test split
- validation-only model and threshold selection
- duplicate and group-leakage audits
- per-class error analysis
- comparison of latency, model size, precision, recall, and mAP
- documented rejected experiments

## Technology

Python, PyTorch, Ultralytics YOLO11, OpenCV, scikit-learn, HOG, Linear SVM, NumPy, pandas, and Matplotlib.

## Status

The complete case study and experiment artifacts are being prepared for publication. This repository currently presents the verified methodology and results.
