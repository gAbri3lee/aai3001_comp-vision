# Whole-scene mall development: completed online/software pass

A more rigorous pilot was executed, but robust Pillar recognition is **not solved**. No experimental model or threshold replaced the app defaults. Physical range/alert measurements were explicitly deferred by the user because a measured setup is unavailable. Cloud implementation remains with the teammate.

## Dataset and controls

A scoped large-structural-column dataset was annotated from whole Singapore mall images. Export: data/mall_column_pilot_v1/data.yaml. It has 10 training images, two validation images and four new test images (23 boxes total). Training venues, validation venues and test venues are disjoint. ION Orchard and 313@Somerset were new test venues; no model predictions were run on them before protocol/model completion. One additional candidate, F001, was excluded before predictions because its support/wall-pier annotation was ambiguous. Its draft is retained in the audit trail, not exported as ground truth.

These are AI first-pass whole-image annotations and an overlay review, **not independent human-certified labels**. The scoped class includes clear structural-column sections >=15% height and >=2% width; small/ambiguous supports, wall piers, trees, signs and sculptures are excluded. Original v4 class semantics and data remain unchanged. This is a starter pilot, not a sufficiently diverse, mature mall dataset or a complete five-class benchmark. Architectural/elevated views remain in development data; new ION views are camera-height-like, but no physical camera walk was performed. Public images may overlap pretraining.

Nine actual CPU runs compared standard full-scene training, matched dim/blur/JPEG augmentation, and transfer of the original model's Pillar classifier row. All use 30 epochs, 40 training draws per epoch, batch four, input 416, first ten layers frozen and seeds 0/1/2. The augmentation control repeats the same source images four times; the augmented arm changes their pixels at the same budget. Validation pixels/labels are identical. The warm-start arm was registered before any new-test prediction, after development-only diagnosis. Extraction parity passed on an old missed-column case and a nonempty detected-column case. Initial weights differ by design in the warm-start comparison.

Every run uses its fixed last checkpoint. The mean validation F1 selects an arm; seed0 is fixed for downstream profiling. **A favourable test seed is not selected for deployment.** Two validation images/three boxes are plainly insufficient for reliable model selection; this remains a material limitation.

## New test results

Four images from two new venues contain five unambiguous scoped column targets. Original model: 0 TP / 0 FP / 5 FN at confidence 0.35; AP50 0.208. New models use the predeclared 0.5 operating threshold and the same scoped size filters; AP uses low-confidence predictions.

| Arm | Seed | New-test TP / FP / FN | AP50 |
| --- | ---: | --- | ---: |
| standard | 0 | 2 / 0 / 3 | 0.679 |
| degradation_augmented | 0 | 2 / 0 / 3 | 0.967 |
| standard | 1 | 3 / 0 / 2 | 0.736 |
| degradation_augmented | 1 | 1 / 0 / 4 | 0.698 |
| standard | 2 | 1 / 2 / 4 | 0.409 |
| degradation_augmented | 2 | 1 / 0 / 4 | 0.642 |
| pillar_warmstart | 0 | 1 / 0 / 4 | 0.515 |
| pillar_warmstart | 1 | 1 / 0 / 4 | 0.547 |
| pillar_warmstart | 2 | 1 / 0 / 4 | 0.649 |

Full-scene training found 1–3 of five columns across runs. This is encouraging relative to the original on this small test, but misses remain and seed variability is substantial. Small-sample AP must not be advertised as general mall accuracy. Augmentation did not consistently beat standard training. The validation-selected arm was pillar_warmstart; its fixed seed0 found only one of five. Do not replace it with the best test seed after inspecting results.

## Broader validation and deployment decision

The selected auxiliary model was merged conservatively with the original detector: core detections retained; only new nonduplicate scoped Pillar boxes admitted at confidence >=0.5. On all 925 original validation images, Pillar TP/FP/FN changed from **47/13/114 to 50/35/111**. Precision fell from **78.3% to 58.8%**. The other four classes' counts and AP were verified identical. The predefined promotion gate failed. Original weights and configuration remain active.

## General obstacles

An explicit frozen YOLO-World vocabulary includes bins, seating and planters. On the older selected targets it matched 5/10 versus the current system's 2/10, including two previously missed bins; those bin confidences were below the current 0.5 alert cutoff. This does not establish better warning reliability.

A new type-scoped test has eight seating/planter targets across the five new venue photos. At confidence 0.35, COCO matched 2/8 with zero unmatched scoped predictions; World matched 4/8 with zero unmatched scoped predictions. At the current 0.5 alert cutoff, COCO remained 2/8 while World fell to 1/8. Therefore a lower-threshold coverage gain does **not** justify silently lowering warning thresholds. This small AI-labelled test has no bin positives and cannot establish general bin recall, arbitrary-obstacle recall or population precision. More complete calibration data is required before changing the general detector.

## Warning behaviour and runtime

The real 9.47-second VivoCity escalator clip was visually sampled at 20 timestamps; no clear large structural column was observed. Both current and research pipelines produced zero column-warning events over 142 sampled frames. This short, already used negative clip cannot establish a reliable false-warning-per-minute rate. Audio was disabled.

Actual whole-app latency includes inference, general detection, decisions, annotations, JPEG publication and localhost HTTP. Speech, browser paint and physical camera buffering are excluded. A 60-second repetition of that clip provides enough paced frames for profiling; repetition is a timing workload, not new independent footage. An ABBA-order repeat measured:

| Profile | FPS, two runs | Capture-to-HTTP median, two runs |
| --- | --- | --- |
| Current | 7.11, 10.06 | 171, 110 ms |
| Research supplement | 7.95, 4.94 | 141, 234 ms |

No speedup is established. Shared-laptop timing varies and the extra inference can impose a substantial cost. All timed runs reported zero pipeline errors. Software regressions: **197 tests passed plus 12 subtests**.

## What remains

The online/software experiment pass is complete. A strong deployable recognition result is not. The main remaining work is a substantially larger, independently reviewed full-scene corpus covering required classes and column styles, a representative validation set, additional unseen walking footage and calibration of object confidence versus false warnings. This pilot provides a reproducible method and evidence of a promising direction; it does not finish that data/model work. Original v4 annotations have not been exhaustively repaired. Physical camera distance, approach-to-audible-alert timing and representative-user usability remain unverified.

## Reproduction and evidence

- annotations/mall_whole_scene_columns.json and annotations/mall_fresh_obstacles.json: annotation scope, source hashes, boxes and venue membership.
- configs/mall_fresh_holdout_candidates.json: licensed acquisition candidates.
- scripts/mall_whole_scene_study.py and scripts/mall_warmstart_study.py: actual matched training.
- scripts/evaluate_mall_whole_scene.py: single first-test evaluation and validation-only selection.
- scripts/mall_fresh_obstacle_test.py: explicit type-scoped object check.
- scripts/mall_supplement_validation.py: unchanged original validation check.
- scripts/mall_video_warning_check.py and scripts/mall_runtime_profile.py: warning/timing research harnesses.
- docs/evidence/mall-whole-scene-study.json: compact results, limitations and unchanged active checkpoint hash.
- artifacts/mall-whole-scene-study/: frozen protocol, checkpoints, training plans, predictions, metrics and actual HTTP timing logs.

Image author/licence details are preserved in the source data folders' sources.json and ATTRIBUTION.md. Preserve those records when distributing derived images or labels. Model/data binaries remain ignored by Git. No Git push or merge was performed.
