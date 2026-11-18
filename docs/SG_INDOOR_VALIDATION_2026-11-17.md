# Singapore indoor mall validation â€” 17 November 2026

The upload/playback software passed its checks. Large-column and general-obstacle recognition remain weak. No experimental model was promoted; the original weights and app configuration are unchanged. This is online real-media validation, not an in-person walking or physical distance trial.

## Inputs and protocol

21 licensed photographs from eight Singapore malls: Funan, Suntec City, VivoCity, City Square Mall, Thomson Plaza, Nex, Bugis Junction and Bugis+ (Iluma). One outdoor Millenia Walk candidate was excluded before predictions. Sources include archival and newer photographs; many views are elevated architectural views, so they are not a representative head-height walking sample.

34 selected targets were visually boxed before first inference: 13 structural columns, 10 generic obstacles, 10 people and one escalator. These are AI-reviewed primary-target annotations with human/team review pending. They are incomplete image annotations: **the fractions below are selected-target matches, not overall class accuracy, mAP or a false-positive rate**. A prediction must have the right label and IoU >=0.5, with one-to-one matching. Source hashes and targets are frozen in annotations/sg_indoor_primary_targets.json. Unmatched predictions require individual review, not automatic FP labelling.

No byte-identical overlap was found against all 10,092 v4 images. This does not rule out resized/cropped overlap, shared venues or foundation-model pretraining overlap. The challenge has now been inspected during exploratory development; a future final model needs a fresh independent assessment.

## Selected-target results

| System | Pillar targets | Object targets | People targets | Median inference |
| --- | ---: | ---: | ---: | ---: |
| baseline | 1/13 | 2/10 | 10/10 | 81 ms |
| text specialist | 0/13 | 2/10 | 10/10 | 137 ms |
| visual specialist | 0/13 | Not applicable | Not applicable | 145 ms |
| Grounding DINO | 3/13 | 7/10 | 7/10 | 8482 ms |
| column pilot | 1/13 | Not applicable | Not applicable | 26 ms |
| hard-negative pilot | 0/13 | Not applicable | Not applicable | 24 ms |

The visual and binary specialists are column-only models; their other classes are not applicable. These are separately evaluated candidates, not full-system replacements. Baseline means the deployed five-class YOLO plus its actual COCO general detector. Of the ten matched people, seven were found by the trained model and three by the general supplement. Only two of ten selected generic obstacles were matched; all three Nex recycling bins were missed. No selected generic target qualified for a warning. The one still-image escalator target was missed. A false Stair box on M010’s ceiling (confidence 0.564) was rejected by the size gate, so it did not trigger a warning. No true stair or wet-floor-sign targets were available in this new collection, so no new Singapore-mall accuracy claim is made for those classes.

Grounding DINO matches more selected targets, including the three bins and three large columns. It also produces false classifications: the reviewed bin photo contains a false Pillar on a wall and a false Escalator on a vertical fixture. Its roughly eight-second CPU inference is unsuitable for this live app. These outputs are research proposals, not trusted automatic annotations or a drop-in replacement.

## Actual application checks

Actual TCP HTTP upload routes, actual trained and general weights, no simulated detector outputs: all 21 images completed and matched direct-model class/count/confidence outputs. Corrupt images were isolated and corrupt video rejected. The real 9.47-second VivoCity clip processed 84 frames, with pause/resume, MJPEG, replay, return-to-camera and temporary-file cleanup passing. **No trained class was detected in that clip**, despite clearly visible escalators. General People detections occurred but were rejected by the size gate. The software functioning is not evidence of successful recognition.

Camera capture was replaced by an idle source for upload isolation; speech, depth and cloud were disabled. Manual browser click-through and human audibility were not tested. A longer licensed Suntec walking clip could not be acquired because the media server returned HTTP 429; no result is claimed for it.

## Pillar improvement experiments

Context padding at scale 0.75 gives the existing model a correct detection of M010's large white column at confidence about 0.60. A new binary specialist trained on 40 reviewed v4 TRAIN-only crops (32 train/8 internal validation) detects that column at about 0.69. Its crop validation AP looks high, but it fails badly on whole validation scenes. Those eight crops are not a field benchmark.

The false predictions often cover people, walls, stairs or almost the entire portrait-image area. This suggests a framing/context shortcut, not proof of a single underlying cause. Training-only hard-negative mining was visually reviewed: ten confirmed non-column crops and one manually corrected structural-column example were added. Two ambiguous proposals were deferred. The second 40-epoch pilot uses 51 crops, the exact same internal held-out source groups and no new Singapore challenge pixels. It reduces false alarms but loses the new column detection. These are small exploratory studies, not a robust improvement claim.

All three supplements were checked on all 925 original validation images with unchanged labels, original class threshold 0.35, extra-column threshold 0.5 and pillar-only merging. Binary supplements reject boxes touching horizontal image edges; all other class detections/AP remain unchanged.

| Supplement to original model | Pillar TP / FP / FN | Precision | Added inference |
| --- | --- | ---: | ---: |
| None (original) | 47 / 13 / 114 | 78.3% | 0 ms |
| context | 47 / 14 / 114 | 77.0% | 59 ms |
| binary | 49 / 172 / 112 | 22.2% | 25 ms |
| binary-hard-negative | 48 / 67 / 113 | 41.7% | 29 ms |

All fail the established deployment gate. Context padding adds one FP and no TP on this validation; neither binary pilot is deployable. Validation labels may be incomplete, but visual review of 24 unmatched examples from different scenes confirms many genuine false alarms. The original model remains active.

## Robustness and remaining work

Thirty controlled inputs cover six photos under normal, dimmed, blurred, JPEG-quality-25 and 320px conditions. People targets remain 5/5 in this selected subset. Pillar matches change from 1/7 normally to 0/7 when dimmed and 2/7 when blurred; Object matches fall from 2/10 to 1/10 at low resolution. This instability supports the domain/data-quality diagnosis. These are controlled image transformations, not measured walking conditions or metre ranges.

The practical next model-development step is a reviewed, full-scene large-column training set with varied cladding, lighting, occlusion and camera angles, plus matched wall/person/stair/sign negatives. The crop pilots demonstrate why a handful of close-up column patches is insufficient. Keep source groups separate, review every visible target under a consistent scope, and evaluate on new mall scenes before replacing weights. The existing broad v4 Pillar taxonomy has not been silently changed.

Physical laptop-camera distance calibration, measured approach-to-audible-alert timing and representative-user usability still require an actual setup. Hosted cloud remains the teammate's responsibility.

## Evidence and reproduction

Full suite: **195 tests passed + 12 subtests**. Four new regression checks cover coordinate remapping and one-to-one primary-target matching. The application weights/config were not changed and no Git push/merge was performed.

- Summary: docs/evidence/sg-indoor-validation-2026-11-17.json
- Raw reports, annotated images and HTTP/video evidence: artifacts/sg-indoor-validation/
- Ready-to-upload media and attribution: artifacts/sg-indoor-validation/demo-inputs/
- Reproducible reviewed crop recipe: configs/column_specialist_crop_review.json

```sh
python scripts/validate_sg_indoor.py --arm baseline --output artifacts/sg-baseline-repeat
python scripts/verify_sg_uploads.py
python scripts/stress_sg_images.py
python scripts/prepare_column_crops.py --variant initial --output artifacts/reproduced-column-initial
python scripts/train_column_specialist.py --source artifacts/reproduced-column-initial
python scripts/prepare_column_crops.py --variant hard-negative --output artifacts/reproduced-column-hard-negative
python scripts/train_column_specialist.py --source artifacts/reproduced-column-hard-negative
python scripts/study_pillar_context.py --method context
python -m pytest -q
```

Baseline tools use project requirements. Research-only dependencies/weights were isolated from the app; text/visual prompting follows [Ultralytics YOLOE](https://docs.ultralytics.com/models/yoloe/) and the heavier detector follows [official Grounding DINO documentation](https://huggingface.co/docs/transformers/model_doc/grounding-dino). Model/dataset binaries are ignored by Git and must be shared separately with their provenance. Crop recipes reproduce the reviewed inputs byte-for-byte; the recipes require the recovered v4 source dataset. Raw artefact directories preserve checkpoints and training plans.
