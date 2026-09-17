# Parcel Damage V1 — Near-Duplicate Review

Review date: 2026-08-26

## Outcome

**23 candidates reviewed: 3 TRUE_DUPLICATE, 20 FALSE_POSITIVE.**

`parcel_damage_v1` is **NOT FROZEN**. The three true cross-split duplicate groups remain in place pending approval. No dataset image or annotation was modified.

The screening metric is a 64-bit difference hash (dHash). The displayed similarity is `1 - distance / 64`; it is not a calibrated probability. Verdicts were assigned by visual comparison of scene geometry, damage contours, surface marks, and package details.

## Candidate verdicts

| ID | A (split / filename) | B (split / filename) | Distance | Similarity | Verdict | Visual basis |
|---:|---|---|---:|---:|---|---|
| 01 | train / `-2023-01-13-11-11-56_png_jpg.rf.de1607867b535e87284440af47cf592c.jpg` | valid / `-2023-01-16-7-24-58_png_jpg.rf.f66c4060145db070895b7a2a22b54a4b.jpg` | 5 | 0.9219 | **TRUE_DUPLICATE** | Identical torn box, folds, holes, framing, and background; tonal/quality variation only. [Comparison](figures/near_duplicates/candidate_01.jpg) |
| 02 | train / `20230116_113038_jpg.rf.17b87a214f2be829c193d07a9e7c9574.jpg` | test / `R246593_Z_Major_Compressed_4852056_jpg.rf.703ec802ee6a39a236de68491f76a67b.jpg` | 5 | 0.9219 | FALSE_POSITIVE | Plain overhead box versus labeled shipment close-up. [Comparison](figures/near_duplicates/candidate_02.jpg) |
| 03 | train / `20230116_131553_jpg.rf.74cc6f4d71255fc075805a125e5ef634.jpg` | valid / `20230116_131446_jpg.rf.10f9493e1adda4807d1806b293c0cea5.jpg` | 3 | 0.9531 | FALSE_POSITIVE | Different damage contours and viewpoints. [Comparison](figures/near_duplicates/candidate_03.jpg) |
| 04 | train / `20230116_133934_jpg.rf.9a037a50deb59fe2f7785345ecb4ecb1.jpg` | valid / `20230116_162941_jpg.rf.2e6d7a2e019d7ad7622da2f3696b7aae.jpg` | 4 | 0.9375 | FALSE_POSITIVE | Different edge tears and package geometry. [Comparison](figures/near_duplicates/candidate_04.jpg) |
| 05 | train / `20230116_163216_jpg.rf.dd4bb685a132701fbbb91669739276db.jpg` | valid / `20230116_162941_jpg.rf.2e6d7a2e019d7ad7622da2f3696b7aae.jpg` | 5 | 0.9219 | FALSE_POSITIVE | Different corner and lower-edge damage patterns. [Comparison](figures/near_duplicates/candidate_05.jpg) |
| 06 | train / `20230116_163738_jpg.rf.89179e67517a1c8e4a086421eae6a75e.jpg` | valid / `20230116_144247_jpg.rf.df4e6409f2329b7f101dd92cd488c296.jpg` | 5 | 0.9219 | FALSE_POSITIVE | Distinct arrangements of holes and seams. [Comparison](figures/near_duplicates/candidate_06.jpg) |
| 07 | train / `20230116_172124_jpg.rf.e434653ae4c3864f9676e0f117048df9.jpg` | valid / `20230116_132026_jpg.rf.8009826b49f4ce9a302fec968eae438f.jpg` | 5 | 0.9219 | FALSE_POSITIVE | Different cutouts, edges, and camera views. [Comparison](figures/near_duplicates/candidate_07.jpg) |
| 08 | train / `3ED5BA0299_JPG.rf.806934371cee4528dad1db5c43b67da2.jpg` | test / `1F69888D87_JPG.rf.c87bcc7ea37c4b779d650fb654c28376.jpg` | 2 | 0.9688 | **TRUE_DUPLICATE** | Identical opening, plastic wrap, inner box, label, and framing; minor quality/tonal variation. [Comparison](figures/near_duplicates/candidate_08.jpg) |
| 09 | train / `6F8BBBB90B_JPG.rf.51e21c70814261b518b92a713bebf676.jpg` | test / `R246593_Z_Major_Compressed_4852056_jpg.rf.703ec802ee6a39a236de68491f76a67b.jpg` | 3 | 0.9531 | FALSE_POSITIVE | Surface crack versus shipment-label scene. [Comparison](figures/near_duplicates/candidate_09.jpg) |
| 10 | train / `7951f5c5004b4790d65d9f83371b7424_JPG.rf.d995e8a1856e35a0b2f9d1872d9bea72.jpg` | test / `R246593_Z_Major_Compressed_4852056_jpg.rf.703ec802ee6a39a236de68491f76a67b.jpg` | 4 | 0.9375 | FALSE_POSITIVE | Surface crack versus shipment-label scene. [Comparison](figures/near_duplicates/candidate_10.jpg) |
| 11 | train / `8HM0431_X_Minor_Creased_4850966_jpg.rf.8de65e69c12a07ad84f4177ca1341bc7.jpg` | valid / `R438604_Y_Moderate_Creased_4858192_jpg.rf.de01ca03fff1b7c21a8694b36ee684bb.jpg` | 5 | 0.9219 | FALSE_POSITIVE | Different labels, packages, dates, and scene structure. [Comparison](figures/near_duplicates/candidate_11.jpg) |
| 12 | train / `av_129024_jpg.rf.41c7b7a8359bb013465806885258008d.jpg` | test / `8FQ1678_Y_Moderate_Creased_4846632_jpg.rf.dcaebe716c4638dbc2c61f6cd5684dd8.jpg` | 3 | 0.9531 | FALSE_POSITIVE | Torn carton versus unrelated shipping-label photograph. [Comparison](figures/near_duplicates/candidate_12.jpg) |
| 13 | train / `IMG_2552_JPG_jpg.rf.67ee1ddc720a15a7da60044693f5a5bd.jpg` | valid / `IMG_2553_JPG_jpg.rf.1412d7a7e0d2ffb6ad5f94a10dd7a2cb.jpg` | 4 | 0.9375 | **TRUE_DUPLICATE** | Identical holes, stains, seam, carpet, framing, and shadows; slight exposure/quality variation. [Comparison](figures/near_duplicates/candidate_13.jpg) |
| 14 | train / `IMG_2772_JPG_jpg.rf.df642618eb29a5482defd6d60a69df13.jpg` | valid / `20230116_131446_jpg.rf.10f9493e1adda4807d1806b293c0cea5.jpg` | 4 | 0.9375 | FALSE_POSITIVE | Different packages, damage, and backgrounds. [Comparison](figures/near_duplicates/candidate_14.jpg) |
| 15 | train / `R433721_Y_Moderate_Compressed_4859716_jpg.rf.432886e45d35f70c1a5833c4c8f62538.jpg` | valid / `R433721_Y_Moderate_Compressed_4859717_jpg.rf.7b6309a2049eb37f58da558d825467e7.jpg` | 5 | 0.9219 | FALSE_POSITIVE | Different labels and package content despite related filename series. [Comparison](figures/near_duplicates/candidate_15.jpg) |
| 16 | train / `R433721_Y_Moderate_Compressed_4859716_jpg.rf.432886e45d35f70c1a5833c4c8f62538.jpg` | valid / `R438604_Y_Moderate_Creased_4858192_jpg.rf.de01ca03fff1b7c21a8694b36ee684bb.jpg` | 3 | 0.9531 | FALSE_POSITIVE | Different airway-bill identifiers and photographs. [Comparison](figures/near_duplicates/candidate_16.jpg) |
| 17 | train / `R438604_Y_Moderate_Creased_4858193_jpg.rf.cfaf95ef5d71e922ef4b645d4dfa2eef.jpg` | valid / `R438604_Y_Moderate_Creased_4858192_jpg.rf.de01ca03fff1b7c21a8694b36ee684bb.jpg` | 4 | 0.9375 | FALSE_POSITIVE | Different label types and distinct captures. [Comparison](figures/near_duplicates/candidate_17.jpg) |
| 18 | train / `R438604_Y_Moderate_Creased_4858193_jpg.rf.cfaf95ef5d71e922ef4b645d4dfa2eef.jpg` | test / `8FQ1678_Y_Moderate_Creased_4846632_jpg.rf.dcaebe716c4638dbc2c61f6cd5684dd8.jpg` | 3 | 0.9531 | FALSE_POSITIVE | Different label identifiers, surfaces, and dates. [Comparison](figures/near_duplicates/candidate_18.jpg) |
| 19 | train / `R438604_Y_Moderate_Creased_4858193_jpg.rf.cfaf95ef5d71e922ef4b645d4dfa2eef.jpg` | test / `8FQ1678_Y_Moderate_Creased_4846635_jpg.rf.451e14ed34936525ea1511b68fb55294.jpg` | 5 | 0.9219 | FALSE_POSITIVE | Similar label layout, but different SO numbers, dates, framing, and package surface. [Comparison](figures/near_duplicates/candidate_19.jpg) |
| 20 | train / `R438604_Y_Moderate_Creased_4858226_jpg.rf.9a1ab8943e46617c5688da2ad24f9b37.jpg` | valid / `R438604_Y_Moderate_Creased_4858192_jpg.rf.de01ca03fff1b7c21a8694b36ee684bb.jpg` | 4 | 0.9375 | FALSE_POSITIVE | Same airway bill family but different pieces (`2 of 2` versus `1 of 2`) and distinct photographs. [Comparison](figures/near_duplicates/candidate_20.jpg) |
| 21 | train / `R438604_Y_Moderate_Creased_4858226_jpg.rf.9a1ab8943e46617c5688da2ad24f9b37.jpg` | test / `8FQ1678_Y_Moderate_Creased_4846632_jpg.rf.dcaebe716c4638dbc2c61f6cd5684dd8.jpg` | 5 | 0.9219 | FALSE_POSITIVE | Different identifiers, labels, dates, and surfaces. [Comparison](figures/near_duplicates/candidate_21.jpg) |
| 22 | valid / `R433721_Y_Moderate_Compressed_4859717_jpg.rf.7b6309a2049eb37f58da558d825467e7.jpg` | test / `8FQ1678_Y_Moderate_Creased_4846635_jpg.rf.451e14ed34936525ea1511b68fb55294.jpg` | 5 | 0.9219 | FALSE_POSITIVE | Different label designs, identifiers, and packages. [Comparison](figures/near_duplicates/candidate_22.jpg) |
| 23 | valid / `R438604_Y_Moderate_Creased_4858192_jpg.rf.de01ca03fff1b7c21a8694b36ee684bb.jpg` | test / `8FQ1678_Y_Moderate_Creased_4846635_jpg.rf.451e14ed34936525ea1511b68fb55294.jpg` | 5 | 0.9219 | FALSE_POSITIVE | Different shipment identifiers, dates, label geometry, and package surface. [Comparison](figures/near_duplicates/candidate_23.jpg) |

## Proposed correction — approval required

Following the retention rule, keep the train copy and remove the corresponding valid/test copy. Each image and its matching YOLO label must be handled together. Proposed removals from a future corrected dataset version are:

1. Candidate 01, remove from `valid`:
   - `valid/images/-2023-01-16-7-24-58_png_jpg.rf.f66c4060145db070895b7a2a22b54a4b.jpg`
   - `valid/labels/-2023-01-16-7-24-58_png_jpg.rf.f66c4060145db070895b7a2a22b54a4b.txt`
2. Candidate 08, remove from `test`:
   - `test/images/1F69888D87_JPG.rf.c87bcc7ea37c4b779d650fb654c28376.jpg`
   - `test/labels/1F69888D87_JPG.rf.c87bcc7ea37c4b779d650fb654c28376.txt`
3. Candidate 13, remove from `valid`:
   - `valid/images/IMG_2553_JPG_jpg.rf.1412d7a7e0d2ffb6ad5f94a10dd7a2cb.jpg`
   - `valid/labels/IMG_2553_JPG_jpg.rf.1412d7a7e0d2ffb6ad5f94a10dd7a2cb.txt`

These six paths are relative to `datasets/processed/parcel_damage_v1/`. Per the project versioning rule, the recommended safe action is to create a corrected `parcel_damage_v2` rather than silently alter V1. No removal has been performed.

## Accepted Dataset V1 limitations

- Eleven marginally out-of-bounds bounding boxes are accepted and are not a freeze blocker.
- The observed class imbalance is intentional for EXP-001; no rebalancing or validation/test augmentation is proposed.
- The broad `minor_damage` ontology is accepted for V1 and remains a documented limitation; no relabeling is proposed.
