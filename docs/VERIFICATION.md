# Verification: 10 November 2026

> Historical record of the 10 November integration. Since then OCR, voice input and
> item identification have been removed, the alert gate and audio changes were
> added, and the suite has grown. For current results see `docs/GUI_TEST_REPORT.md`.

The implementation uses the supplied checkpoints; no model was retrained.
The original handover bundle and all saved notebook training outputs are intact.

## Automated checks

**The integration's 20 tests and 7 subtests pass**, including in the original checkout's newly
installed local environment. The original-checkout YOLO golden check also
matched 20/20 reference boxes with no extras. Native-library cold startup on this
Intel Mac made that first fresh-environment suite take 6 minutes 21 seconds.

The regression suite covers the five-class mapping (including Pillar), wrong
checkpoint labels, invalid payloads, default/overridden configuration, no-depth
notices, relative-depth ranking, cooldown, background-image evaluation, golden
image coverage/exit status, RGB JPEG decoding and original coordinates, real HTTP
round trips, confidence overrides, total deadlines with one request in flight,
stale result expiry (including suppression of stale local speech), optional-query failures and unavailable microphone startup.
A fresh whole-branch review identified the stale-display and auxiliary-error
issues; regression tests reproduced both before the fixes.

## Model accuracy

| Runtime | Images | mAP50 | mAP50–95 | Safety recall |
|---|---:|---:|---:|---:|
| YOLOv8n PyTorch CPU | 920 | 0.6576959453 | 0.4193178136 | 0.7155263349 |
| YOLOv8n FP32 ONNX CPU | 920 | 0.6576955999 | 0.4193177963 | 0.7155263349 |
| RF-DETR Nano Linux CPU | 920 | 0.7394540704 | 0.4471151032 | 0.7671352878 |

All three runs use the notebook COCO AP / greedy IoU >=0.5 alert evaluator, the supplied
YOLO text labels, low-threshold AP predictions >=0.001 and the fixed alert
threshold 0.35 for YOLOv8n and 0.40 for RF-DETR. All 920 images are scored, including 92 missing-label background
images. These are not Ultralytics `.val()` metrics. Differences from saved
notebook metrics are small; this run uses written labels and a different CPU
runtime rather than the original notebook's in-memory labels/T4 environment.
The export mAP50 loss is approximately 0.0000003454, well below the 0.01 limit.

## Golden parity

- Local PyTorch: **20/20** reference boxes, **0 extras**; exact 20-image coverage.
- Local ONNX: **20/20** reference boxes, **0 extras**; exact 20-image coverage.
- RF-DETR via actual Linux HTTP service: **24/25** reference boxes, **0 extras**;
  exact 20-image coverage. This is **96%**, passing the specified 95% criterion;
  it is not identical detection output. Largest matched confidence difference:
  approximately 0.0157 across the different runtime/hardware.
- The local golden references contain no Pillar detections; dedicated class and
  RF-DETR adapter tests exercise Pillar instead.
- RF-DETR's final background logit can appear at threshold 0.001. The adapter
  discards only this known background slot (ID 5), preserves foreground IDs 0–4,
  and rejects any other unknown ID.

## Runtime and benchmark context

Local test machine: Intel Core i7-9750H, macOS 26.2, Python 3.12, PyTorch 2.2.2,
Ultralytics 8.4.183, ONNX Runtime 1.23.2. RF-DETR runs in a Linux Docker container
with RF-DETR 1.11.2, CPU PyTorch 2.9 and Transformers 5.1. The cloud endpoint was
bound only to localhost for verification; no paid hosting was provisioned. The
final rebuilt image was tested without source mounts: `/health` passed, and
`POST /detect?confidence=0.001` returned only foreground IDs 0–4.

Each detector timing uses 10 warm-up frames followed by 50 measured frames.
CPU timings varied significantly under concurrent verification/build load:
PyTorch medians 103–423 ms; ONNX medians 215–300 ms. The measured cloud HTTP median
was 1202 ms (p95 1968 ms) on this CPU/container host, including the local round trip.
These are provisional shared-machine observations, not controlled claims that
one format is faster or guarantees a particular frame rate. `.pt` therefore
remains the default; ONNX is available and accuracy-verified for a controlled
benchmark on the actual deployment device before adoption.

The recorded-image continuous safety-loop benchmark (detection, optional MiDaS
on detected frames, ranking/cooldown/message, annotation and JPEG encoding) used
10 warm-up +50 measured frames with depth enabled: median 2587 ms, p95 17448 ms,
mean throughput 0.193 FPS under shared load. It excludes physical webcam capture,
audible speech, on-demand queries and browser rendering. The full CPU pipeline
must not be described as real-time based on the old README's 10–15 FPS claim.
`--no-depth` remains available; it still announces class and position without
inventing distance.

## Auxiliary features and remaining external checks

- Actual item-ID, EasyOCR and MiDaS smoke tests succeeded on a supplied image.
  The installed TTS engine initialized and enumerated 180 voices without playing
  audio.
  The sample OCR scene returned no text; this verifies runtime execution, not
  text-recognition accuracy. MiDaS returned a 640×640 relative 0–1 map.
- Flask route/status and replayed-frame processing tests passed. Browser visual
  automation could not open the preview due browser-session/time-out errors;
  no visual verification is claimed.
- Physical camera, microphone recognition and audible TTS still need a user
  hardware smoke test. None were activated during verification.
- The exposed Roboflow API key still needs rotation in its account. Literals are
  removed from notebook source, but source cleanup cannot revoke the key or
  remove it from previous Git/Kaggle history.

Machine-readable reports are recorded in `verification_results.json`; per-image
outputs and additional benchmark reports are under the ignored `artifacts/` folder.

## Training CLI removal — current full-suite status

The user subsequently confirmed removal of both training CLI files, including
concurrently updated versions, and requested keeping training in the notebooks.
Both CLI files and the unused legacy dependency manifest are now deleted.
Export, benchmarking and inference-verification scripts remain available.

The standalone prepared-dataset validator was preserved. Its tests now call the
validator directly; only the test for the removed CLI's training options was
retired. Class order, labels, backgrounds, split overlap, manifest membership,
scene leakage, validated paths and fingerprint checks remain covered.

**The complete repository suite passes: 35 tests and 7 subtests.**
Earlier concurrent training-test failures recorded at the initial handoff are
historical and no longer describe the current suite. Notebook contents, trained
weights and measured model accuracy were not changed by this removal.
