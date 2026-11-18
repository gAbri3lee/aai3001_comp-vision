# Four-class migration — 17 November 2026

Active classes are **0 Escalator, 1 People, 2 Stair, 3 Wet Floor Sign**. Pillar is removed from the local classifier head, dataset taxonomy, GUI/library/summary and alert config. Dataset/model originals and historical results remain preserved. No commit, push, merge or hosted cloud deployment was performed.

## Dataset audit and cleanup

The audit used the actual reproduced notebook scene split: 10,092 images (8,247 train / 925 valid / 920 test). This is the prepared training split, rather than the entire raw Roboflow download. Its manifest matches the recorded fingerprint and supplied test membership.

| Check | Original prepared split | Cleaned derivative |
| --- | ---: | ---: |
| Images | 10,092 | 9,377 |
| Corrupt images / invalid boxes | 0 | 0 |
| Orphan labels / duplicate annotation rows | 0 | 0 |
| Exact byte or decoded-RGB duplicate groups | 0 | 0 |
| Exact duplicate groups crossing splits | 0 | 0 |
| Recorded scene/source-stem overlap across splits | 0 | 0 |
| Perceptual candidate pairs | 2,177 | 0 |
| Perceptual candidate pairs crossing splits | 0 | 0 |

Every image was decoded and every label structurally checked. The all-class perceptual screen uses 64-bit pHash distance <=4 and dHash distance <=6, with RGB-thumbnail difference recorded. A visual sample across the range shows many adjacent video frames/repeated views. Most pairs share a recorded scene; one pair has different recorded scenes within the same training split. Candidates are not automatically called confirmed duplicates.

The derivative conservatively **quarantines 715 candidate images** by excluding them from the new copy, never deleting originals. Selection prefers retained-target count, then resolution, then filename; it uses no predictions, confidence or test scores. Original split membership is preserved, with zero images moved between splits. No retained pair remains an edge in the candidate graph. A completely fresh scan of all copied images confirms zero exact duplicates and zero remaining perceptual candidates under this screen.

This does not certify all alternative crops, rotations, same-location views or unlabelled hazards. **Automated checks covered every image/label; independent visual reannotation of every image was not completed.** Exclusion reasons, original hashes/names and scene membership are retained. Short content-derived filenames prevent Windows path-length failures.

Pillar labels are removed and old IDs 0/1/3/4 become new 0/1/2/3. Useful images and background negatives remain; **pillars may still appear in background scenery**. Empty labels explicitly mark backgrounds. Source attribution/readmes are preserved as supplier provenance, separately from the new dataset specification.

| Split | Images | Escalator boxes | People boxes | Stair boxes | Sign boxes | Backgrounds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 7,749 | 919 | 8,109 | 2,677 | 1,415 | 1,135 |
| Valid | 859 | 121 | 816 | 316 | 68 | 140 |
| Test | 769 | 103 | 559 | 268 | 52 | 143 |

Zero pillar annotations remain. Fingerprint: `d4242de8c1d3776941c614a99dc432a6`. The actual Ultralytics resolver and YOLO dataset loader accepted all 9,377 images/labels with zero corrupt files. YAML omits `path: .` so Ultralytics resolves split paths against the YAML directory rather than the launch directory.

## Are the supplied weights invalid because of duplicates?

**The audit does not establish duplicate leakage as a flaw in those weights.** No exact or screened perceptual pair crosses the reproduced splits. The original notebook groups recorded source scenes before splitting and includes wet-floor-specific similarity filtering; the new audit broadens the screen to every class.

Similar frames within a split reduce effective diversity and make frame-weighted metrics depend on long recordings. They do not by themselves show test images were trained on. Grouping/hashes remain heuristic and cannot prove every scene's independence or a checkpoint's complete training history. The external mall failures establish a generalisation limitation; they do not prove duplicates caused it. Structural checks cannot certify annotation completeness.

The old figures remain descriptions of their recorded evaluation protocol, with these limitations. They are not guaranteed mall-field accuracy. Removing pillar does not fix escalator misses.

## Native four-output model and regression checks

`models/local/yolov8n_v4_four_class.pt` is an **inference-only projection**, not retraining. The pillar row was removed from each final independent classifier convolution. Backbone, box/DFL regression and retained class weights remain unchanged. Class metadata/output dimensions were updated; stale optimizer/training metrics were not presented as new four-class results.

On black/random 640x640 tensors, retained boxes/class scores for all 8,400 decoded anchors were exactly equal before NMS (maximum difference 0). Native predictions were then computed on **all 925 original validation and 920 original test images**, before using cleaned subsets:

- At confidence 0.35, all four classes have identical TP/FP/FN, precision and recall.
- All retained per-image predictions match within IoU >0.999 and confidence difference <0.00001.
- The strict AP-identity gate **failed**: maximum low-threshold AP50 loss is 0.00021723 absolute, or **0.0217 percentage points**. Removing a competing class changes low-confidence argmax/NMS. Separate engineering acceptance allows <=0.001 absolute AP loss with deployed-output parity; conversion passed that criterion. This is disclosed, not described as exact AP identity or an improvement.

Tested equivalence does not guarantee every future input is unchanged. Original checkpoint SHA256: `7d59f6ee9bd6c1ada26ca74436185ee7d9bb2cc4668520401bd390c6f025fabe`. Four-class checkpoint SHA256: `58626a2662586f1a9c55de6a8b5aa3843dbbcb29f9e8ddc4455fd0583ceb9a39`.

**207 tests + 12 subtests passed**, including migration/cloud mapping regressions. All **20 golden images passed**, with 20 retained detections matched and zero extras. Real HTTP checks used actual four-class and general weights: 11 external photos matched previous retained-class outputs, both real videos completed, and four-class UI config/library/summary, removed expected-class rejection, corrupt-file isolation, replay/return-to-camera and cleanup passed. No pillar outputs or swapped IDs appeared.

Camera acquisition alone was replaced by an idle source. Speech, physical distance and hosted cloud were not tested here. Previous external-media weaknesses remain, including escalator misses in VivoCity. Previous ONNX results refer to the five-class export; it is rejected by the active config until a matching four-class export is verified.

## Cleaned-subset benchmark

Same projected model, **769 retained test images**, confidence 0.35 and IoU >=0.5 for counts; low-threshold COCO AP:

| Class | AP50 | Precision | Recall |
| --- | ---: | ---: | ---: |
| Escalator | 94.39% | 92.55% | 84.47% |
| People | 69.41% | 75.35% | 57.96% |
| Stair | 76.34% | 80.36% | 67.16% |
| Wet Floor Sign | 69.60% | 92.11% | 67.31% |

Four-class mean AP50 is 77.44%. **This changed class set/subset is not an improvement over the old 65.77% five-class mean.** The model was trained on the original training data; cleanup did not retrain it, erase original training exposure or create an independent holdout. These inspected scenes remain historical development/evaluation evidence.

## Handover and reproduction

Restart the application to load the active `visionaid_config.json` and matching four-class local weights. Original `data/v4_scene_split`, `models/local/yolov8n_v4.pt` and `configs/legacy_v4_five_class.json` remain preserved. Old README/context/demo material is locally archived under `artifacts/four-class-cleanup/`, with committed history also available.

The teammate's native five-class RF-DETR remains supported by explicit `cloud.source_classes`. Adapters validate source IDs/names, filter pillar/background and map Stair 3->2 and sign 4->3. Mixed/forged contracts are rejected; the HTTP schema is unchanged. A newly trained native four-class cloud model needs a matching source-class update and hosted integration test. No cloud weights were pruned, uploaded or deployed here.

Reproduction tools: `scripts/audit_four_class_dataset.py`, `build_four_class_dataset.py`, `remove_pillar_head.py`, `compare_four_class_head.py`, `verify_four_class_runtime.py`. Builders refuse to overwrite outputs. Pre-conversion comparisons use preserved original cached predictions from this supplied checkpoint. Original five-class training notebooks/research are historical and should not be rerun under the active four-class schema without adaptation.

Raw evidence: `artifacts/four-class-cleanup/` contains original/cleaned audits, build/exclusion record, head conversion/comparison, cleaned metrics, structural validation, actual Ultralytics loading, real runtime uploads and golden results. Compact evidence: `docs/evidence/four-class-migration-2026-11-17.json`. Matching dataset and model/config handover ZIPs are local artifacts; weights/data are git-ignored and no remote publication was performed.

Background-negative use and split reasoning follow the [Ultralytics testing guide](https://docs.ultralytics.com/guides/model-testing) and [data-leakage guidance](https://www.ultralytics.com/glossary/data-leakage). Numerical head-conversion behaviour was established against the installed model and recorded tests, rather than assumed from generic pruning advice.
