# Controlled Pillar study

The first pilot added exposure and training epochs simultaneously, then lost precision. It did not establish that sampling improves Pillar. Keep the original model.

The next experiment uses scripts/controlled_pillar_study.py with the already prepared private Kaggle v4 input. It requires CUDA and excludes test images. All arms start from the supplied checkpoint SHA256 recorded in pilot_input.json and use seed 0, 10 epochs, batch 16, AdamW at 0.0001, no early stopping and identical augmentation settings (including close_mosaic=0). Every arm has 8,247 training draws per epoch and the same 925 validation images. This controls update count, unlike the earlier pilot. A full 10-epoch run may be longer than early stopping.

Before this study's first submission, fine-tuning settings were made explicit: mosaic 0.25, scale 0.2, translation 0.05, one warmup epoch with bias learning rate 0.0001, cosine decay and final learning-rate fraction 0.1. The pinned library's defaults use three warmup epochs and bias warmup LR 0.1; these are not silently inherited for this low-LR continuation. These choices are hypotheses, not proven optimal settings, and are identical in all arms.

Every saved epoch is evaluated at the fixed deployment confidence 0.35. Selection ranks gate pass first, then Pillar recall, precision and overall AP. If every checkpoint fails, none is accepted regardless of the selected rejected checkpoint's recall. Repeated validation selection is exploratory and does not replace a final independent field set.

| Arm | Training inputs | Question |
| --- | --- | --- |
| control | Original v4 train images, each once | What does further conservative fine-tuning alone do? |
| curated | Remove the two reviewed problem scenes, then refill 23 draws from remaining training images | Does the documented curation help at the same update budget? |
| curated-balanced | Same exclusions; 25% pillar draws, cycling scenes with at most four draws per pillar image | Does the task-specific sampling strategy add value beyond curation? |

Balanced sampling intentionally changes image exposure and does not retain every original image in every epoch. Its 25% target and repeat cap are hypotheses fixed before this experiment, not proven optimal settings or a claim of global research novelty.

The two quarantined scenes are IMG_8603_MOV (a Pillar box enclosing floor) and amel-1_mp4 (a Pillar box enclosing stair/landing). Evidence and reasons are in configs/v4_training_quarantine.json. The curated variant preserves originals, class IDs, validation and test. It is not a complete annotation repair. All three arms have now completed ten epochs.

Run each arm on the same type of GPU, one at a time, with both runner scripts available in the working directory:

```sh
python controlled_pillar_study.py --arm control --seed 0
python controlled_pillar_study.py --arm curated --seed 0
python controlled_pillar_study.py --arm curated-balanced --seed 0
```

A ready-to-import Kaggle notebook is notebooks/pillar_controlled_study.ipynb. Attach the existing private VisionAid Pillar Pilot v4 Input. It embeds both runner scripts and launches the three arms sequentially; the submitted standalone script uses the same runner source. Repeat all compared arms at matching seeds 1 and 2 before claiming a robust gain. Do not compare only a lucky candidate seed with one control seed.

On 17 November 2026, CLI authentication was completed by the user and version 1 was submitted at approximately 14:57 Singapore time to the private `evantyk/visionaid-pillar-controlled-v4` script. Retrieved metadata confirms private visibility, GPU enabled, T4 machine shape and only the existing private pilot input attached. The execution log confirms Tesla T4 and the control baseline validation. The study completed: all 30 saved checkpoints failed the predeclared gate. The original app checkpoint is unchanged. The standalone submission source and run logs are under ignored `artifacts/kaggle-controlled/`.

Acceptance requires Pillar recall +10 percentage points, Pillar precision at least max(75%, baseline precision minus 2 points), no other class losing more than 3 points recall, and no more than 1 point overall mAP50 loss. Compare within the same evaluator. Also inspect error images, class confusion and false positives; a gate pass alone does not establish user safety.

Save training_draws.txt, plan/result JSON, before/after validation and best.pt. Freeze the chosen configuration before the separate external challenge. The current 920-image test is historical development evidence and must not guide threshold/sampling choices. Online imagery can overlap model pretraining or source scenes and cannot replace real camera/range trials.


## Completed three-arm results (17 November 2026)

All 30 checkpoints failed the gate. The selected **rejected** checkpoint in each arm is shown below; selection does not mean acceptance.

| Arm | Pillar TP / FP / FN | Precision | Recall | Decision |
| --- | --- | --- | --- | --- |
| Original baseline | 47 / 13 / 114 | 78.3% | 29.2% | Retain |
| Control, epoch0.pt | 47 / 16 / 114 | 74.6% | 29.2% | Reject |
| Curated, epoch9.pt | 47 / 28 / 114 | 62.7% | 29.2% | Reject |
| Curated-balanced, epoch2.pt | 55 / 45 / 106 | 55.0% | 34.2% | Reject |

The downloaded review archive SHA256 is `0873728919a69ae8715332192f73f41d7828897633b3ed45922203954ee89013`. Selected checkpoint hashes were checked against the run's comparison JSON. Raw outputs are under `artifacts/kaggle-controlled/extracted/`. The original deployment weights remain unchanged.

## Repaired-label follow-up

Training-only detector proposals were visually adjudicated, not accepted automatically. `configs/pillar_label_repair_2.json` records source image/label hashes, explicit pixel boxes, reviewer attribution and exclusions. It adds 34 boxes in 22 images and removes two confirmed wrong Pillar boxes. These include other trained classes where clearly visible. This is a partial AI-assisted review; team/human review remains pending. Validation/test labels are unchanged.

183 incompletely reviewed CCTV training frames are excluded in both arms. Registration generated 34 additional frame proposals, but these were **not adopted**, because other visible trained targets were still unannotated. The same repaired-membership-derived schedule is used with original and repaired labels: 8,247 draws, 1,604 Pillar draws, at most 32 Pillar draws per recorded scene and four per Pillar image. Remaining draws are unique non-Pillar images. The two arms differ only in reviewed labels.

Both start from the original checkpoint, use seed 0 and 12 epochs, freeze the first ten layers and freeze BatchNorm running statistics after `model.train()` at each epoch's first batch. Other optimizer/augmentation settings match the conservative study. This is a different training strategy, so its matched control is necessary. Every saved epoch is evaluated against the original 925-image validation set and the same acceptance gate. It remains exploratory, with no new independent field test and no automatic deployment.

Submitted private jobs: `evantyk/visionaid-pillar-labels-control-v4` and `evantyk/visionaid-pillar-labels-repaired-v4`. Submission source is saved under `artifacts/kaggle-label-repair/`. Results are now complete and all candidates were rejected; see the latest result below.


The local verified variant is `data/v4_pillar_label_repair_2_final/data.yaml`: all 9,172 image copies are byte-identical and independent, exactly 22 training labels differ, and validation labels are byte-identical. No test files were read in this verification. Intermediate local copies were marked incomplete and their training YAML disabled after discovering a Windows long-path `is_file()` omission (892 rather than 925 validation images). The copy helper now checks native long paths and exact manifest membership; Kaggle's Linux runs already confirmed 925 images and were unaffected. Full regression: 191 tests and 12 subtests passed.

The 12-epoch matched control completed; all 12 checkpoints failed. Its selected rejected checkpoint (epoch4.pt) has Pillar 53 TP / 43 FP / 108 FN, precision 55.2%, recall 32.9%, and also loses more than three recall points for Wet Floor Sign. Original baseline is still 47 / 13 / 114. The repaired-label run has completed; no candidate passed. See the latest result below.


## Completed repaired-label follow-up (latest result)

Both matched arms completed 12 epochs and all 24 checkpoints failed the fixed acceptance gate. The repaired run used the identical draw schedule as the control (verified SHA256), all 925 unchanged validation images, and no test images. Selected checkpoint SHA256 was verified after download.

| Pillar validation, confidence 0.35 / IoU 0.5 | TP | FP | FN | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original deployed weights | 47 | 13 | 114 | 78.3% | 29.2% |
| Selected rejected control, epoch4.pt | 53 | 43 | 108 | 55.2% | 32.9% |
| Selected rejected repaired labels, epoch4.pt | 54 | 41 | 107 | 56.8% | 33.5% |

The repair gives a small exploratory improvement over its matched control, but not a defensible deployment improvement over the original: seven more correct Pillar detections accompany 28 more false positives, and Wet Floor Sign recall falls from 87.3% to 83.9%. Repaired overall Ultralytics mAP50 is 71.99% versus original 72.17% (same evaluator). No candidate is deployed. Local native re-evaluation and GUI promotion tests were not run for rejected candidates. This experiment is finished; the Pillar class issue is **not resolved**. The partial annotation repair is retained as reviewed development work; broader consistent full-image annotation and independent column scenes remain necessary. This result does not establish that annotation repair in general cannot help.

Downloaded evidence: `artifacts/kaggle-label-repair/repaired/results/repair_repaired/result.json`, `plan.json`, `draw_names.txt`, `selected.pt`. Original deployed checkpoint SHA256 is unchanged.
