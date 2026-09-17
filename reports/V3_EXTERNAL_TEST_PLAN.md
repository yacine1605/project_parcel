# External final-test plan for `transfer-mobile-v3-dev-1`

The next valid evidence must come from new external parcel captures. The preserved
V2/V3 test directory is development-contaminated because its earlier error
analysis motivated label correction.

## Predeclared primary metrics

- damaged recall with 95% confidence interval;
- specificity with 95% confidence interval;
- damaged precision and F1;
- confusion matrix;
- recall by ordinary damage, open box, and minor damage;
- performance by lighting, view angle, distance, session, and camera.

## Acceptance gate

The candidate passes the development objective only if external damaged recall is
at least 90%. Specificity, precision, confidence intervals, subgroup support, and
failure examples must still be reported even if recall passes. No post-test
threshold change is allowed.

## Model locked for that evaluation

- Version: `transfer-mobile-v3-dev-1`
- Checkpoint SHA-256: `fdba95ebfcb73545ab118bd55dbccbdb2ff769234945600b8c017365d157f7a5`
- Threshold: `0.6363311409950256`
- Input: RGB, shorter-side resize 256, center crop 224, ImageNet normalization

Collection template:
`datasets/external/parcel_binary_external_v1/manifest.csv`.
