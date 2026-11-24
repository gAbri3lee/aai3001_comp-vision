# Pillar recovery: diagnosis and acquisition status

## Verified inputs

The active model is the supplied v4 YOLOv8n checkpoint, with Pillar at class ID 2. The training notebook references Roboflow workspace `potatos-workspace-kequm`, project `version2-nvt5l`, version 4. It downloads the raw export and constructs its own scene-grouped split and split_manifest.csv. Repository data/ is ignored; the full train/validation images are not in the fetched Git tree. The supplied weights ZIP contains the 920-image evaluation split, not train/validation.

Downloaded Version2.v4i.yolov8.zip through the signed-in Roboflow export UI. Preserved the raw export in data/raw_v4_export. Reproduced the original notebook scene split in data/v4_scene_split: 8,247 training, 925 validation and 920 test images. Fingerprint 49601080aac1f838190341d331785578 and all 920 supplied test-image stems match exactly. Structural validation passed; this does not certify semantic annotation completeness. Reusing test images as training/validation would invalidate subsequent evaluation.

## Completed diagnosis

Reused the already recorded low-threshold predictions (confidence >=0.001) and matched against all 153 labelled pillar boxes at IoU >=0.5. No deployment threshold was changed.

| Outcome | Boxes |
| --- | ---: |
| Matched Pillar at deployment threshold 0.35 | 26 |
| Matching Pillar prediction exists but confidence is below 0.35 | 93 |
| No matching Pillar box at IoU 0.5 | 34 |

For the 93 below-threshold cases, confidence ranges from 0.00124 to 0.31817; median 0.02076. None is a duplicate-assignment case with a matching Pillar prediction already above 0.35. This is evidence of low confidence/localization failures, not proof of a particular training cause. The original 26/153 recall and historical metrics are unchanged.

Visually triaged all 153 annotated-box examples using seven contact sheets. Observations: repeated outdoor video views form much of this sample; labels include structural columns, bollards, slim posts/poles and a cabinet-like structure. Some frames show other potentially in-scope vertical obstacles without matching labels. These are scope/completeness questions, not an assertion that every such label is wrong. Original full-resolution training examples need review before adopting exclusions or corrections. Test labels were not changed to improve reported metrics.

Evidence: `artifacts/pillar-diagnosis/failures.csv`, `summary.json`, and `pillars_00.jpg` through `pillars_06.jpg`. Reproduce with `python scripts/diagnose_pillar.py` after the model verification outputs exist.

## GPU pilot (2026-11-16)

User approved transferring the 389,195,030-byte training/validation and supplied YOLO checkpoint bundle to a private Kaggle input. Archive CRC passed; it contains 8,247 train and 925 validation images with labels, manifest and runner, with no test images. Both input dataset and notebook were set to Private. Submitted a saved ten-epoch GPU pilot at https://www.kaggle.com/code/evantyk/visionaid-pillar-recovery-pilot-v4 using scripts/pillar_gpu_pilot.py. Results are pending; submission is not evidence of improvement.

The pilot preserves labels, class definitions and application thresholds. It retains every training image and adds two extra exposures for up to 50 pillar-containing images per training scene. Validation is measured before and after. Other-class recall and overall AP regression gates prevent automatic promotion; no model is deployed automatically. Extra training and sampling are confounded in this pilot, so a matched control is needed for causal claims.

## Remaining work

1. Completed exact dataset recovery and structural audit. Review pilot execution and validation results; retain the baseline checkpoint.
2. Establish one explicit Pillar annotation scope with the team (structural columns only versus the broader existing vertical-obstacle category). Review the whole training image, not just existing pillar boxes. Correct annotations in a new version and record changes.
3. Audit pillar coverage by independent scene, size, viewpoint, lighting and occlusion; audit missing objects and confusing wall/fixture negatives. Address training coverage using independent scenes, not additional adjacent frames.
4. Compare a corrected-data baseline against a scene-balanced weak-class training strategy, retaining the other four classes. Choose thresholds on validation data, check false positives and regressions, and consider resolution/model-capacity experiments only under a controlled budget.
5. Retrain and verify the actual checkpoint, then test the complete application including the optional general detector. Do not assume a missed pillar is automatically detected as Object.
6. Report per-class recall/AP, false alerts, latency and size-binned results. Use a newly held-out field set for the final claim; the 920-image set is now repeatedly inspected historical evaluation data.

No improved checkpoint, improved recall, field-distance capability, or completed pillar fix is claimed at this stage. This PC has no CUDA GPU; the pilot uses Kaggle GPU compute. No improved checkpoint is accepted until verified results are available.

## Queue recovery (2026-11-17)

Version 1 remained queued for roughly an hour with zero output. Cancelled that unstarted batch attempt and started the same private input/runner in an interactive GPU session. Observed CUDA available=True, Tesla T4 with 14,912 MiB, Ultralytics 8.4.183, and baseline validation label scanning (466/925, zero corrupt at that point). This verifies actual GPU execution, not completed training or improved metrics. The original version-1 log is cancelled; follow the notebook editor for the active draft run. Before/after results remain pending.

## GPU execution correction (2026-11-17)

The initial saved batch stayed queued for about an hour and was cancelled. Interactive allocation succeeded on Tesla T4. The first execution completed baseline AP but failed before training: passing 925 paths as a Python list caused Ultralytics to load one batch and exhaust GPU memory. Fixed evaluation to use a directory source with batch=16, restarted the kernel, and launched the corrected runner in a new timestamped output directory. The new run saved before_validation.json successfully; training and after-validation results are still pending. A local check of the real Ultralytics file loader produced batches [16, 5] for 21 images; syntax passed. The uploaded original bundle runner remains unchanged; the notebook applies the explicit correction to a working copy.

## Completed pilot result (user-supplied Kaggle output, 2026-11-17)

The corrected pilot completed successfully, stopping normally after six epochs (patience=3; best checkpoint epoch 3). Training reported 0.199 hours, excluding preparation/evaluation. Final JSON gate=false. At fixed confidence 0.35 and matching IoU 0.5 on 925 validation images, Pillar recall rose from 47/161 (29.19%) to 52/161 (32.30%), but precision fell from 47/60 (78.33%) to 52/94 (55.32%). Pillar false positives rose from 13 to 42. Overall mAP50 fell from 0.721746 to 0.715151; mAP50-95 fell from 0.454806 to 0.446198. No other class exceeded the predeclared 3-percentage-point recall-regression limit, but pillar recall gain and precision requirements failed. Reject this candidate for deployment; keep supplied original weights. Resampling alone has not fixed the pillar issue. Training-label/coverage review and a controlled follow-up remain necessary; no new final field-test claim.

Evidence parsed into artifacts/pillar-pilot-result-user-output.json from the supplied output. This verifies reported completion/metrics, not possession or independent evaluation of the candidate checkpoint. The verbose Python dictionary after final JSON is runpy returning its module namespace, not an error.

## Project strengthening and new online inputs (2026-11-17)

See docs/RUBRIC_EVIDENCE.md for the actual briefing criteria and honest completion status. A 216-image representative training triage identified two confirmed bad Pillar annotation examples. configs/v4_training_quarantine.json and scripts/build_curated_training.py build a separate 8,224-image training candidate, excluding 23 images across two scenes while preserving raw data and validation/test. This has not been retrained. A three-arm matched-budget GPU study and importable notebook are prepared but not executed; browser access is unavailable and no Kaggle API credential is configured.

Current baseline validation and threshold study, verified ONNX adapter/metrics, online attribution/overlap screening and actual image/video route failures are summarized in docs/evidence and docs/ONLINE_DATA_CURATION.md. No model accuracy improvement is claimed. The broad Pillar definition and annotation completeness remain unresolved.

## Executed CPU classifier experiment (2026-11-17)

The GPU job had not started at this stage. The experiment used a separate Kaggle CLI environment and was awaiting authentication.

Actually trained a CPU-feasible frozen-feature candidate using `scripts/pillar_head_recovery.py`: all eligible training pillar images plus at most four non-pillar images per training scene, 2,048 training images total. The two existing quarantine scenes were excluded. No validation/test images entered training. Only 243 parameters (Pillar's final classifier row/bias at three scales) were learned; backbone, localization and other output rows stayed frozen. Mined 214,927 proposals, including 18,810 positive proposals from 95 recorded pillar scene groups. These are proposals/scene groups, not independent photographed objects/sites. Four interpolation strengths were fixed before validation. The cached baseline exactly reproduced the original 925-image validation mAP50 and operating-point counts; three initial cached outputs also matched native inference within 0.00025 pixels.

| Update strength | Pillar TP / FP / FN | Precision | Recall | Decision |
| --- | --- | --- | --- | --- |
| Original | 47 / 13 / 114 | 78.3% | 29.2% | Retain |
| 0.10 | 47 / 24 / 114 | 66.2% | 29.2% | Reject |
| 0.25 | 50 / 41 / 111 | 54.9% | 31.1% | Reject |
| 0.50 | 60 / 105 / 101 | 36.4% | 37.3% | Reject |
| 1.00 | 103 / 21,392 / 58 | 0.5% | 64.0% | Reject |

Every candidate failed the predeclared gate. The full update grossly miscalibrated background scores; reporting its recall alone would be misleading. At fixed 0.35, the other four classes retained their TP/FP/FN counts, but low-threshold AP also deteriorated in the strongest candidate. No candidate was deployed, confidence thresholds were not changed, and original checkpoint SHA256 remains `7d59f6ee9bd6c1ada26ca74436185ee7d9bb2cc4668520401bd390c6f025fabe`.

This rules out this particular classifier-only shortcut, not all fine-tuning strategies. Broader feature adaptation, a consistent annotation scope and independent mall coverage remain necessary. Four rejected validation candidates do not constitute improved accuracy or completion of the pillar fix. Evidence: `docs/evidence/pillar-head-recovery-2026-11-17.json`; raw caches/checkpoints remain ignored under `artifacts/pillar-head-recovery-scene-capped/`.

## Authenticated controlled GPU study (2026-11-17, complete)

The user completed official Kaggle CLI authentication. Private script `evantyk/visionaid-pillar-controlled-v4`, version 1, was submitted around 14:57 Singapore time. Retrieved metadata verifies private visibility, T4 GPU and the existing private train/validation input only. Logs confirm actual Tesla T4 execution. All three arms finished ten epochs; all 30 saved checkpoints failed the acceptance gate. Curated-balanced increased Pillar TP from 47 to 55 but also FP from 13 to 45 (55% precision), so it was rejected. No candidate has been accepted or deployed.

The explicit conservative fine-tuning settings, saved-epoch selection rule and same-budget comparison are documented in CONTROLLED_MODEL_STUDY.md. Initial mounted-input scanning and the first epoch were slow; subsequent epochs are about 65–68 seconds. The run was retained rather than restarted once loading warmed up. This elapsed training time excludes checkpoint evaluation and does not predict final completion precisely.

Independent local verification is prepared in `scripts/compare_validation_models.py`: exact native counts/AP, paired recorded-scene bootstrap intervals, Pillar size bins and alert-gate eligibility. Its baseline-versus-itself check reproduces original validation metrics and zero paired differences. Size bins are not metre ranges, and alert eligibility is not proof of audible speech or safe navigation.


## Training annotation repair and matched follow-up

The audit confirmed missing structural-column labels as well as the two known wrong labels. The first reviewed repair adds 34 boxes in 22 training images, removes two wrong Pillar boxes, and excludes 183 incompletely reviewed CCTV frames equally from both arms. Other existing labels and validation/test are preserved. These are AI-reviewed annotations, with human/team review pending; this is not an exhaustive dataset certification.

The matched original-label/repaired-label follow-up is submitted to private Kaggle T4 runs. It uses 12 epochs, a frozen backbone/BN statistics and identical scene-capped draws in both arms. Actual improvement remains pending. See CONTROLLED_MODEL_STUDY.md for exact settings and evidence paths. No replacement weights have been deployed.


Latest follow-up state: the matched frozen-backbone/BN control completed all 12 epochs and all 12 checkpoint evaluations; none passed. The repaired-label T4 job remains queued despite 27+ hours of unused free GPU quota and no reserved GPU time at the quota check. This is pending external GPU allocation, not completed model training. Local repair-copy verification passed for all 9,172 train/validation images, exactly 22 changed training labels and unchanged validation labels. Use `data/v4_pillar_label_repair_2_final/data.yaml`; superseded intermediate copies are marked incomplete. Regression: 191 tests + 12 subtests passed. The original app weights remain active.


## Completed repaired-label follow-up (latest result)

Both matched arms completed 12 epochs and all 24 checkpoints failed the fixed acceptance gate. The repaired run used the identical draw schedule as the control (verified SHA256), all 925 unchanged validation images, and no test images. Selected checkpoint SHA256 was verified after download.

| Pillar validation, confidence 0.35 / IoU 0.5 | TP | FP | FN | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original deployed weights | 47 | 13 | 114 | 78.3% | 29.2% |
| Selected rejected control, epoch4.pt | 53 | 43 | 108 | 55.2% | 32.9% |
| Selected rejected repaired labels, epoch4.pt | 54 | 41 | 107 | 56.8% | 33.5% |

The repair gives a small exploratory improvement over its matched control, but not a defensible deployment improvement over the original: seven more correct Pillar detections accompany 28 more false positives, and Wet Floor Sign recall falls from 87.3% to 83.9%. Repaired overall Ultralytics mAP50 is 71.99% versus original 72.17% (same evaluator). No candidate is deployed. Local native re-evaluation and GUI promotion tests were not run for rejected candidates. This experiment is finished; the Pillar class issue is **not resolved**. The partial annotation repair is retained as reviewed development work; broader consistent full-image annotation and independent column scenes remain necessary. This result does not establish that annotation repair in general cannot help.

Downloaded evidence: `artifacts/kaggle-label-repair/repaired/results/repair_repaired/result.json`, `plan.json`, `draw_names.txt`, `selected.pt`. Original deployed checkpoint SHA256 is unchanged.
