# Main integration and actual-model verification — 14 November 2026

## Integration

Refreshed all remote heads. Main `209dda4`, Matthias_Claude `c6733bc`, and Gabriel `9e1fd6e` are unchanged. Both other branches are ancestors of Gabriel. Git merge-tree reports no conflicts; main can take Gabriel's integrated history without separately reapplying Matthias's commits. Work and fixes are on the local `integration/main-ready` branch. No remote main update or push was performed.

Keep the app's `visionaid_config.json`: the ZIP's config has matching class IDs and local thresholds but lacks `alert_gate` and `severity`. Replacing it with the ZIP config breaks startup. Weights and test data are under ignored `artifacts/supplied_weights/`, not Git. The bundle's YAML aliases train/validation/test to the test images for evaluation convenience: never use it to train or calibrate quantization.

## Bugs fixed

Previously fixed: Windows stale-upload cleanup/process checking; mixed-resolution distance-calibration maths and corrupt calibration handling; disconnected-camera preview crash; capture pause/stop recheck after timed waits. Test fixes address Windows symlink fixture privileges and asynchronous video-disposal timing.

Actual model loading exposed a further upload crash: Ultralytics wraps Pillow's Image.open and may attempt an optional HEIF codec on malformed input. A missing codec caused HTTP 500 for a batch containing garbage. Uploads now reject non-JPEG/PNG signatures before decoding and handle missing optional codec errors; the cloud JPEG endpoint also handles that exception. Added regression tests. A malformed file now produces its own error without aborting the valid image batch.

Browser playback exposed a misleading UI label: a selected hazard was tagged 'spoken' even with speech off. It now says 'selected'; actual announcements remain in the spoken log. No model thresholds were tuned using the test results.

## Executed real-model tests

Used the actual supplied `local/yolov8n_v4.pt`, with canonical classes Escalator, People, Pillar, Stair, Wet Floor Sign. Camera, speech, optional depth and general COCO detection were disabled for isolated local-model verification. Only the absent camera was replaced in the verification harness; detector predictions were real.

- **Golden parity:** all 20 supplied golden images pass against their expected detections.
- **Full supplied test:** 920 labelled images, inference at 640 pixels on CPU, confidence 0.35 for alert metrics, confidence 0.001 for AP. mAP50 **0.65770**, mAP50–95 **0.41932**, precision **0.79791**, recall **0.59346**, macro safety-class recall **0.71553**. This reproduces the supplied model's performance; it does not prove independent field generalization or audit the original training split.
- **Warmed inference:** 50 frames after 10 warmups, median **39.1 ms**, p95 **49.6 ms**. These are detector timings on this PC, not whole-app FPS.

| Class | Labelled objects | Recall at confidence 0.35 / IoU 0.5 |
| --- | ---: | ---: |
| Escalator | 158 | 89.9% |
| People | 569 | 58.2% |
| Pillar | 153 | 17.0% |
| Stair | 288 | 68.8% |
| Wet Floor Sign | 116 | 56.0% |

The actual upload-route test used one selected large-labelled-object example for each class, plus a damaged file. Four expected classes were seen; the pillar example was missed. This deliberately selected five-image demonstration is not an accuracy estimate. The generated MP4 contains these five real images as a ten-second slideshow; all 100 frames were processed, replay accepted, and temporary-file cleanup passed. Pillar was never identified, and Stair appeared during its segment: the video result retains this false classification rather than claiming five-class success.

Also exercised the browser GUI at localhost:5055 using the real trained model: selected an image through the file chooser (Escalator 0.93 displayed), selected the generated MP4 through the video file chooser, observed playback and the finished summary (four classes seen, Pillar not seen). The video is a slideshow, not walking footage. Camera and speech remained disabled; cloud RF-DETR execution and full cloud/local comparison were not run.

## Distance results and limits

Verified the optional distance display with an explicitly assumed calibration and actual model boxes: a fully visible person gets the pinhole estimate and a truncated person gets no estimate. This verifies the software calculation, not physical distance accuracy. Only People and Wet Floor Sign have known-height estimates; stairs/escalators/pillars do not have metric distance estimates. The calibration used for this check was not saved as a genuine camera calibration.

Fresh size-binned recall was computed from all supplied test predictions. Examples: People at 5–10% of frame height: **28/149 (18.8%)** matched; People at 10–15%: **47/85 (55.3%)**. Wet-floor signs below 5%: **0/5**, 5–10%: **21/67 (31.3%)**, 10–15%: **34/34**. These are image-size results with limited sample counts, not maximum ranges in metres.

Recognition and alerts are separate. The sample sign was recognized at approximately 15.7% frame height, below the configured 18% alert threshold: no spoken alert there is expected by design. Recognition view should be used to distinguish a model miss from alert suppression.

No maximum reliable detection distance in metres can be verified from this archive: it contains no measured-distance camera trials. Those need the intended camera, fixed zoom/calibration, labelled recordings at tape-measured distances, and varied sites/light. Do not present assumed-calibration estimates or digitally scaled images as measured physical-range evidence.

## Evidence and reproduction

- `artifacts/real-model-golden.report.json`: golden parity.
- `artifacts/real-model-test.report.json`: full test metrics and timings.
- `artifacts/real-model-test.json`: actual prediction outputs.
- `artifacts/real-model-size-recall.json`: per-class size bins.
- `artifacts/real-uploads/report.json`: real upload detections, assumed-calibration check, video summary, replay/cleanup.
- `../output/main-integration-final.xml`: final regression results.

Run the app upload verification with `python scripts/verify_real_uploads.py`; `--serve` opens the same real-model test UI on localhost:5055, with camera/cloud/speech/general model disabled. Install project and test requirements first. Evidence was produced using the already available Windows Python 3.12 CPU environment, not the separate pinned Linux RF-DETR serving environment.
