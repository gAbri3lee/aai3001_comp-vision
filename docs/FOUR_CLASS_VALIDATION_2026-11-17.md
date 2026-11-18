# Remaining-class recognition validation — 17 November 2026

Actual trained weights and the application's actual upload endpoints were tested. Wet-floor signs were the strongest class in this small sample; escalators remained weak. People benefited substantially from the general detector. **These are provisional, small-sample external-media results, not certified mall-wide accuracy or physical distance validation.**

## Results on 11 additional photos

Correct class plus one-to-one bounding-box IoU >=0.5, deployed confidence >=0.35. Only Escalator, People, Stair and Wet Floor Sign are scored. Pillar remains ID 2 in the model/application; nothing was deleted or renumbered.

| Class | Labelled targets | Trained model TP / FP / FN | App including general TP / FP / FN | App recall |
| --- | ---: | ---: | ---: | ---: |
| Wet Floor Sign | 4 | 4 / 2 / 0 | 4 / 2 / 0 | 100% |
| Stair | 3 | 2 / 0 / 1 | 2 / 0 / 1 | 66.7% |
| People | 31 | 14 / 1 / 17 | 22 / 3 / 9 | 71.0% |
| Escalator | 6 individual flights | 1 / 0 / 5 | 1 / 0 / 5 | 16.7% |

TP = matched detection, FP = unmatched prediction, FN = unmatched target. These are localisation counts, not just image-level class presence. A recognisable person with a poorly sized box can count as FP plus FN. Zero FPs on a handful of cases does not establish a low population false-positive rate.

Review of the unmatched outputs:

- Wet Floor Sign: one duplicate box on another face of the same physical A-frame sign (R003), and one false classification of a hanging Don Don Donki lantern (R012). Four successful sign instances cover only three source groups: R002/R003 show the same sign/setting. This is promising, limited evidence, not proof of reliable sign recognition.
- Stair: the downward view at Parklane and the upward double staircase at South City Mall matched. The foreground Schiedam staircase did not match.
- People: the general detector adds eight net matches. Most misses concern smaller, crowded or occluded people. Two unmatched app boxes are genuine people with localisation below IoU 0.5: the child's box includes wheeled luggage (IoU about 0.48), and a partial person box has IoU about 0.37. The third is a mannequin wrongly labelled People in the Sarasota shop display. Thus all three unmatched predictions are not three unrelated hallucinations.
- Escalator: one VivoCity flight matched. The other VivoCity flight and the HarbourFront/Changi pairs did not. Adjacent flights are separately annotated; a bank-level impression is not credited as multiple detections.

The trained model's photo AP50 values are 19.1% Escalator, 73.2% People, 66.3% Stair and 91.7% Wet Floor Sign. These use low-threshold predictions and the same provisional annotations. No combined-system AP is reported: the general detector's outputs are already cut at 0.35. Do not compare a combined cutoff-only AP with full low-threshold model AP, or present this as a five-class benchmark.

## Media, annotation and independence

Three Singapore mall photographs: HarbourFront Centre, Parklane and VivoCity; one additional Changi T4 airport view; seven overseas indoor mall photographs. Overseas locations and the airport remain explicitly identified. The collection includes close signs, crowds, upward/downward stairs, reflective surfaces and a mannequin display. R010 has distant people below the annotation size limit; it is not claimed to be a completely empty mall. Outdoor entrance R011 was excluded before predictions because it is outside the indoor scope.

There are 44 target boxes. The declared scope is visually resolvable physical instances at least 10% of image height; predictions below that height are also excluded. Adjacent escalator flights are individual targets, adjacent fixed stairs form one staircase. Mirror reflection and ambiguous distant flight regions are explicitly ignored. This **does not measure small/distant-object recall**. Pixel height cannot establish metres.

Annotations were made through AI visual inspection of raw photos before first inference. Subsequent prediction review found loose/omitted people boxes; the corrections are documented in `annotations/four_class_external.json`. Initial annotations and initial results are retained in `artifacts/four-class-validation/annotations-before-review.json` and `report-before-annotation-review.json`. No model, threshold or confidence tuning followed the results. Independent human/team annotation review is pending, so all box/count results remain provisional.

All 10,092 original v4 images were screened: no byte-identical photo matches. One near dHash candidate, R004, was visually reviewed: its closest training image is a close-up person on an escalator, whereas R004 is a yellow sign and mirror. It is a hash collision, not a duplicate. This heuristic screen does not exclude all resized/cropped overlap, related venues or foundation-model pretraining exposure. The two Broen sign photos and all frames of a video are related samples, not independent trials.

## Real videos and actual application checks

1. **New 20-second outdoor pedestrian clip** from OpenCV's public `vtest.avi` sample, converted to MP4. The actual video upload/playback endpoint completed and processed 161 frames on the recorded rerun. The trained People class was present in the summary throughout; this presence count is not per-person recall. Four raw decoded frames at 0, 5, 10 and 15 seconds contain 22 labelled person-instances. Uploading those frames through the actual image-analysis endpoint gave **18 TP, 0 FP, 4 FN** with the combined detector (81.8% recall). Misses include people occluded by the pole/sign or other people. These are repeated people in one outdoor scene, not 22 independent people or Singapore mall evidence. The frames were annotated without overlays, after the separate initial playback; this is not a blind unseen-video assessment.
2. **Reused 9.47-second VivoCity indoor escalator clip** completed through the actual video upload/playback endpoint and processed 76 frames on this rerun. **No trained class was reported, despite visible escalators.** General people outputs occur, but this is not successful escalator recognition. This is a repeat check of a previously used clip, not a second new independent source.

All 11 photo uploads and four keyframe uploads completed using actual weights. A corrupt image in the batch was isolated; valid images still completed. Temporary upload storage cleanup passed. Local camera acquisition was replaced with an idle source solely to isolate file uploads. Detector outputs were not mocked. Speech, depth and cloud were disabled. The manual browser button-click flow was not repeated in this pass; these are real HTTP endpoint checks, not a claim of new human UI testing.

Two additional proposed escalator videos could not be downloaded because their media endpoints returned HTTP 429; no result is claimed for them. No new positive stair or wet-floor-sign video was acquired. Those classes have photo evidence only in this pass.

File input exercises decoding and the same recognition pipeline, but does not validate the laptop camera's exposure, motion blur, calibrated distance, or camera-to-audible-alert timing. Generic Object recall is outside this four-class score; the mannequin error is relevant evidence about the general detector, not a comprehensive obstacle benchmark.

## Reproduction and evidence

The original active model is `models/local/yolov8n_v4.pt`, SHA256 `7d59f6ee9bd6c1ada26ca74436185ee7d9bb2cc4668520401bd390c6f025fabe`. Original weights, runtime thresholds, application code and five-class IDs were not changed by this validation pass. There was no push or merge.

Run with the project's validation environment/dependencies and the existing local video inputs:

```powershell
python scripts/fetch_online_sources.py --candidates configs/four_class_validation_candidates.json --output data/four_class_external
python scripts/audit_external_overlap.py --sources data/four_class_external --output artifacts/four-class-validation/overlap-screen.json
python scripts/validate_remaining_classes.py
```

Video acquisition/conversion provenance is retained in `data/four_class_videos/opencv-source.json`; the VivoCity conversion comes from the preceding online-media validation. The OpenCV clip is a public software sample; a separate Creative Commons media licence was not established. Do not describe either source as team-recorded footage. Commons photo licences, authors, source pages and resize modifications are preserved in `data/four_class_external/ATTRIBUTION.md` and `sources.json`.

- Annotated visual gallery: `artifacts/four-class-validation/index.html` (photos: trained predictions; keyframes: combined predictions; green GT/orange prediction).
- Full results: `artifacts/four-class-validation/report.json`; photo counts: `photo-counts.csv` beside it.
- Compact evidence: `docs/evidence/four-class-validation-2026-11-17.json`.
- Photo inputs: `data/four_class_external/`; frozen annotations: `annotations/four_class_external.json` and `annotations/four_class_video_frames.json`.

Priority from this evidence: improve escalator coverage/localisation first, then stair variation and crowded/occluded people. Add sign lookalikes and duplicate-sign cases to training-only negatives. Keep these inspected sources as development evidence; assess any subsequently trained model on new source-separated media. Fresh positive stair/sign videos and independent human label review remain outstanding. Physical laptop-camera calibration and audible timing still need an actual measured setup.
