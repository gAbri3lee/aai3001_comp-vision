# Four-class development and final GUI checks — 18 November 2026

The active model remains the original native four-class projection. No experimental checkpoint has been deployed. Classes are Escalator, People, Stair and Wet Floor Sign. Cloud deployment remains the teammate's responsibility.

## Annotation work and controlled experiments

AI visually reviewed 60 training images: 12 positive examples per class and 12 backgrounds, covering different recorded scenes. Seven frames had 14 confirmed omitted boxes: 13 People and one Wet Floor Sign. Those boxes were added to the working training labels after verifying image and original-label hashes. Original labels are archived. Validation/test labels and every image are unchanged. This is targeted AI review; independent team review and exhaustive annotation completeness remain outstanding. Suspected ambiguous stair labels were retained rather than removed without sufficient evidence.

The cleaned dataset still contains 7,749 training, 859 validation and 769 test images. Revised training box counts are Escalator 919, People 8,122, Stair 2,677 and Wet Floor Sign 1,416; 1,132 training images are backgrounds. The image/scene manifest fingerprint remains `d4242de8c1d3776941c614a99dc432a6`; it does not fingerprint the revised labels. The annotation revision and per-label hashes are recorded separately in `annotations/four_class_training_review.json`, dataset metadata and `artifacts/four-class-study/applied-label-review.json`.

Control records were frozen before corrections. Three paired arms isolate unchanged labels, repaired labels, and repaired labels plus class/scene balancing. All start with the same checkpoint and use fixed final epochs, seeds 0/1/2, matched update counts and no test images. The preregistered acceptance requires a mean AP50 gain of at least two percentage points over both the control and starting model, bounded class precision/recall losses, then external and runtime checks. No automatic promotion occurs.

A separate small CPU pilot completed all nine runs: 512 fixed training images, two epochs, 320-pixel training, and the same full 859-image validation evaluated at 640 pixels with the notebook COCO evaluator. These short runs are an optimization check, not evidence of convergence or a replacement for full-data training.

| CPU arm | Mean validation AP50 across three seeds | Sample SD | Mean change versus starting model |
| --- | ---: | ---: | ---: |
| Starting model | 79.76% | — | — |
| Unchanged-label control | 71.95% | 2.50 percentage points | −7.80 percentage points |
| Repaired labels | 71.91% | 2.34 percentage points | −7.85 percentage points |
| Repaired labels + balanced sampling | 65.05% | 2.60 percentage points | −14.71 percentage points |

All CPU candidates failed acceptance. Label repair showed no consistent gain over the matched control; the balancing recipe was worse. These outcomes support retaining the starting model, not claiming an accuracy improvement. Full-data four-epoch GPU runs use a separate registered plan. The historical validation set has already informed development and is not a fresh final holdout.

Two full-data GPU seed comparisons have completed on Tesla T4. These are partial results, not the final three-seed aggregate:

| Seed | Starting AP50 | Control AP50 | Repaired-label AP50 | Repaired + balanced AP50 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 79.76% | 78.62% | 78.89% | 78.22% |
| 2 | 79.76% | 79.66% | 79.61% | 79.60% |

Neither completed seed shows the required gain. Seed 0 remains queued on Kaggle; the full GPU study is **incomplete** until it finishes. Seed 1's batch output and seed 2's interactive output/checkpoints were downloaded. The temporary interactive session was asked to stop after saving and CRC-checking its archive, with explicit user approval. Seed 2 executed the identical fixed script used by the corrected batch submissions; its attached plan differs only in orchestration revision/hash metadata. Data and training/evaluation parameters match. No notebooks or datasets were deleted.

## Program defects corrected

- The GUI footer and video summary retained five-class/Pillar wording. Both now reflect the active four-class scope.
- Camera focal calibration could be applied to unrelated uploaded images and videos. Metre estimates now require the live-camera context; uploaded/file media cannot inherit that calibration. This does not establish actual distance accuracy or automatically identify camera hardware/zoom.
- Actual browser testing reproduced a Windows `WinError 32` crash when replay opened a video while return-to-camera disposed its temporary file. Source operations now share a consistent lock, and cancelled replay handles cannot delete the active video's file. Actual replay/return and ownership regressions pass.

Verification: **212 tests plus 12 subtests pass**. The earlier 20-image golden check remains applicable because no detector weights or inference preprocessing changed in these fixes.

## Actual browser tests

An isolated headless Chrome instance drove the real frontend with the actual local and supplementary general detector weights. Ten workflow checks passed: four-class controls, keyboard pause/resume and focus guards, native file chooser with valid/corrupt image batches, recognition/alert views, demo thumbnail and analysis, corrupt-video rejection, actual video upload/pause/resume/summary, replay/back-to-camera, a 390×844 layout check and absence of uncaught JavaScript errors. Mobile horizontal overflow was zero. The temporary demo fixture was removed.

This is browser automation, with **zero human participants**. It demonstrates functioning software paths, not representative-user usability or strong recognition on arbitrary malls. Evidence and screenshots: `artifacts/four-class-study/browser/`. Speech was disabled with `--no-tts` following the user's request. The test app was stopped after verification.

## Distance, speech and remaining limits

The user could not provide tape-measured distances. No valid physical measurement can be recovered from arbitrary online photographs or unknown camera intrinsics. Software calibration maths, source scoping and warning logic are checked, but actual metre error, maximum useful detection range and audible end-to-end warning timing remain unmeasured. No camera calibration was fabricated or installed.

Remaining evidence needed for a strong submission: team review of annotations, completed GPU comparisons reported honestly, a new independent venue/scene challenge, representative-user tasks, and measured laptop-camera distance/alert tests. Existing online escalator/crowd misses remain disclosed. The supplementary COCO detector covers its own vocabulary; it cannot recognise every unknown obstruction. The teammate should provide hosted-cloud integration and latency evidence. None of these checks guarantees an A+ grade.

## Reproduce and inspect

- Annotation decisions: `annotations/four_class_training_review.json`; application: `scripts/apply_four_class_review.py` (hash guarded, idempotent).
- Team sign-off pack: `artifacts/four-class-study/team-review/index.html` shows original/revised boxes side by side; `signoff.csv` remains explicitly pending. Regenerate with `scripts/build_four_class_review_pack.py`.
- Paired training: `scripts/four_class_training_study.py`; frozen plans/records under `artifacts/four-class-study/`.
- Aggregate all three seeds with `scripts/summarize_four_class_study.py`; CPU output: `artifacts/four-class-study/cpu-comparison.json`.
- Browser workflow: `scripts/verify_four_class_browser.py`; pass report: `artifacts/four-class-study/browser/report.json`.
- Regression log: `artifacts/four-class-study/pytest-final.log`.
- Compact tracked evidence: `docs/evidence/four-class-development-2026-11-18.json`.
- Silent local launch: `run_demo.bat --no-tts`.

No commit, push, merge or hosted-cloud deployment was performed. Models, full datasets and bulk experimental artifacts remain ignored by Git and require their matching handoff files.
