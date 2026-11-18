# Rubric assessment and evidence**18 November follow-up:** [Four-class development and GUI checks](FOUR_CLASS_DEVELOPMENT_2026-11-18.md) records targeted review of 60 training images, 14 omitted boxes repaired, nine CPU control/curation/sampling runs, 212 passing tests plus 12 subtests, and actual browser workflows. All CPU candidates failed acceptance; full-data GPU comparisons are pending. Physical distance and representative-user evidence remain outstanding. Older five-class studies below are historical.

Assessed 17 November 2026 against the supplied AAI3001 Project Briefing, slides 2, 3 and 5. This is an evidence assessment, not a promised grade. The brief supplies Excellent/Satisfactory/Average descriptions, not a numerical A+ boundary or component weights.

**Active scope update (17 November):** Pillar is removed at the user's direction. See [four-class migration](FOUR_CLASS_MIGRATION_2026-11-17.md). A traceable four-class derivative and native four-output local head passed automated integrity checks, 207 tests + 12 subtests, 20 golden checks and real uploads. The pillar studies below and previous ONNX results are historical five-class evidence. Conversion is not retraining or an accuracy improvement. Retained-class annotation review/generalisation, hosted cloud, measured calibration and usability remain outstanding.

| Criterion | Excellent descriptor in the brief | Current evidence | Required before an Excellent claim |
| --- | --- | --- | --- |
| Dataset | Acquire own or curate existing data; annotate own or existing data | Active four-class derivative has 7,749 train / 859 validation / 769 test images. Integrity and exact/screened duplicate checks completed; no screened cross-split overlap found. Targeted AI review of 60 retained-class training images repaired 14 omitted boxes in seven frames, with original labels and hashes preserved. | Team review and sign-off of annotations, broader completeness checks and source/venue independence. AI review is not team annotation; online imagery is not team photography. |
| Model development | Architecture not previously used for a similar task, or novel training strategy | Original model and four-class projection are traceable; projection preserved deployed outputs on the original validation/test images within declared tolerances. Nine paired four-class CPU runs across seeds 0/1/2 completed and failed acceptance; full-data GPU comparisons are in progress. Earlier five-class/pillar studies are historical and produced no accepted model. | Finish/report the registered full-data comparisons, class AP/precision/recall and failure cases; confirm any qualifying candidate on independent scenes. Explain the task-specific strategy without claiming global architectural novelty. A standard YOLO fine-tune alone does not satisfy the strictest descriptor. |
| Deployment | Both local and cloud deployments with optimized models | Working four-class local app, 20 golden checks and real uploads; RF-DETR service/container code and explicit legacy-class mapping. Historical five-class ONNX preserved behavior but offered no speedup; it is not a matching active four-class export. Earlier whole-app camera and upload timings are recorded. | Teammate to demonstrate reachable hosted cloud integration and comparable local/cloud latency, concurrency and recovery. Keep PyTorch unless a matching export proves a useful whole-app gain. Historical ONNX timing (about 69 ms versus 63 ms PyTorch) does not establish a speedup. |
| User interface | Intuitive, well-designed, outstanding usability | Ten actual Chrome workflow checks passed with real detectors: images, corrupt inputs, video, keyboard, pause/replay/back-to-camera, views and mobile layout. Windows replay file ownership and upload calibration scoping bugs fixed; 212 tests plus 12 subtests pass. | Representative users must assess task success, time, recovery and speech clarity. Actual measured camera distance and audible warning timing remain untested. Automated browser checks do not establish usability for blind users. |
| Scalability | Clear avenues for enhancement and growth | Shared detection contract, local/cloud adapters, separate general-object model, bounded cloud work and latest-frame processing. Optional ONNX adapter keeps the same contract. | Measure concurrent service requests, CPU/GPU/RAM and stale-response behavior. Demonstrate adding a model through the existing contract. More classes require new labels, retraining, thresholds and regression tests; no unsupported open-world claim. |

## Priority order

1. Complete independent review of the retained four-class annotations. Automated structural/duplicate checks are done; external mall examples expose escalator/crowd misses and sign lookalikes. Pillar is outside active scope.
2. Controlled studies have completed without an accepted gain. Strengthen full-scene annotations and hard negatives, then validate a new candidate; retain the original checkpoint meanwhile.
3. Freeze an online external challenge separately from development images, check duplicate/source overlap, and annotate it before predictions. Online footage cannot establish physical distance accuracy or camera performance while walking.
4. Prove optimized local and hosted cloud operation with comparable workloads, measured p50/p95 latency and failure handling. Keep slow exports as experimental alternatives.
5. Finish usability evidence, A4 poster using the supplied template, a five-minute real-time demo with slides, backup recording and preparation for the ten-minute Q&A. Team identities/contributions and presentation claims must be factual.

## Current numerical evidence

Runtime reliability now has a separate [17 November validation report](RUNTIME_VALIDATION_2026-11-17.md): 195 passing tests plus 12 subtests, real combined-model uploads, actual laptop-camera timing and native Windows speech recovery. Physical distance accuracy, browser click-through and representative-user usability remain outstanding; this evidence does not improve the model accuracy figures below.

- Historical 920-image test, original YOLO: COCO mAP50 0.65770; Pillar recall 26/153 (17.0%) at 0.35. This set has repeatedly been inspected and is historical evaluation, not a fresh final holdout.
- Current 925-image validation, original YOLO: notebook-style COCO mAP50 0.709121, mAP50-95 0.443492; Pillar TP/FP/FN 47/13/114 (precision 78.3%, recall 29.2%).
- First GPU pilot, same validation and fixed threshold: Pillar 52/42/109 (precision 55.3%, recall 32.3%); overall Ultralytics mAP50 0.721746 to 0.715151. Candidate rejected. These Ultralytics AP values use a different evaluator from the notebook-style COCO values above and must remain separately labelled.
- New ONNX adapter, same 925 images: COCO mAP50 differs by less than 0.000001 from PyTorch; all class precision/recall counts unchanged. This preserves behavior rather than improving recognition.
- Distance estimates are optional known-height estimates for People and Wet Floor Sign. No measured maximum range or reliable metre accuracy has been established.

Fresh online challenge testing shows large-column misses and no trained detections in the processed VivoCity clip; see ONLINE_DATA_CURATION.md. This prevents an objective claim of strong mall recognition today. Application functionality and model generalization are separate.

## Reproduction

Use Python 3.12, install requirements.txt and requirements-test.txt. Optional ONNX versions are in requirements-onnx.txt. Copy supplied weights into the configured models/local and models/cloud paths for normal startup, or pass an explicit --model path for local verification. No Roboflow credential is needed to run the app.

```sh
python scripts/verify_detector.py --model artifacts/supplied_weights/local/yolov8n_v4.pt --images data/v4_scene_split/valid/images --evaluate --frames 0 --output artifacts/validation-baseline.json
python scripts/validation_study.py --dataset data/v4_scene_split --predictions artifacts/validation-baseline.json --model artifacts/supplied_weights/local/yolov8n_v4.pt --output artifacts/validation-study
python scripts/prepare_annotation_review.py --dataset data/v4_scene_split --output artifacts/training-annotation-review
python scripts/build_curated_training.py --dataset data/v4_scene_split --output data/v4_train_curation_1
python scripts/fetch_online_sources.py
python -m pytest -q
```

Artifacts/data/model binaries are ignored by Git. Share approved data and weights separately and preserve source attribution. The training review is incomplete, not a certification of every image. No improved model or new hosted service has been deployed by this strengthening pass.


## Latest Singapore-mall evidence

See [the detailed 17 November indoor-mall validation](SG_INDOOR_VALIDATION_2026-11-17.md). Actual TCP uploads passed for 21 photos from eight malls and a real VivoCity clip. Current combined detector matched 1/13 selected structural columns, 2/10 selected generic obstacles and 10/10 selected people; these are partial-primary-target checks, not class accuracy or mAP. The video produced no trained-class detections. Software reliability is stronger than mall generalization.

Context padding, two pretrained prompting approaches, Grounding DINO and two separately trained binary crop pilots were investigated. No candidate met the evidence needed for deployment. Grounding DINO improved selected-target coverage but was slow and produced false classifications; the crop pilots failed whole-scene validation. Dataset recipes, checkpoint hashes, failed results and source attribution are retained for reproducible analysis. No A+ model-development claim follows from these failures alone. Full-scene large-column coverage, consistent annotation and new independent field validation remain the main gaps. The user reports that a teammate owns cloud implementation; live cloud integration has not been revalidated in this pass.


## Follow-up whole-scene study

[The whole-scene study](MALL_WHOLE_SCENE_STUDY.md) adds a scoped 16-image/23-box export, venue-disjoint validation/test, nine actual matched runs across three seeds, explicit-object vocabulary testing, all-925-image validation, warning-event checks and ABBA-order whole-app timings. The four new test images have only five column targets: full-scene models find 1–3, original finds zero, but the validation-selected model increases original-validation Pillar false detections from 13 to 35. No candidate is accepted. The explicit-vocabulary detector increases low-threshold object coverage but loses that advantage at the current alert cutoff. Regression: 197 tests + 12 subtests. Labels remain AI first-pass annotations pending independent review; this is a pilot, not completed large-scale data repair or strong field accuracy. The user explicitly deferred physical testing because a measured setup is unavailable.
