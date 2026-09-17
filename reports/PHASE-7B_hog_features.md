# Phase 7B — HOG Features

Feature extraction completed for all **3,959 retained images** with **zero failures**.

## Configuration

| Parameter | Value |
|---|---|
| Fixed image size | 128×128 |
| Conversion | Pillow grayscale (`L`), scaled to [0,1] |
| Resize | Center-fit/crop with Lanczos |
| Orientations | 9 |
| Pixels per cell | 8×8 |
| Cells per block | 2×2 |
| Block normalization | L2-Hys |
| Feature vector length | 8,100 |

Features were extracted independently of model fitting. Any subsequent scaling is fitted only by the SVM pipeline on training data.

## Outputs and time

| Split | Shape | Processing time |
|---|---:|---:|
| train | 2,771 × 8,100 | 34.15 s |
| valid | 594 × 8,100 | 6.47 s |
| test | 594 × 8,100 | 6.47 s |

Compressed arrays and metadata are in `datasets/processed/parcel_binary_v1/features/`. The reproducible implementation is `scripts/extract_hog_features.py`. Test features were generated without calculating or inspecting test performance during configuration selection.
