# Runtime validation — 17 November 2026

Software reliability improved; physical distance accuracy and representative-user usability remain unverified. Changes are local, uncommitted and not merged/pushed. Trained weights, class IDs, confidence thresholds and inference resolution were not changed.

## Fixes

- In-flight results are invalidated by mode, source, view and pause changes. An obsolete result cannot publish or queue a new alert after a switch.
- Camera frames are checked again after waiting for the inference lock; expired frames are discarded before inference.
- Unmute resets alert cooldown. Shutdown stops speech and closes its idle worker.
- Windows consecutive speech and mute recovery use a persistent external event loop on the COM-owning worker. An engine purged by mute is retired so its old completion cannot truncate the next phrase. A dequeued phrase cancelled before playback cannot start later.
- MJPEG sends each newly published preview once. Live previews are bounded to 1280 pixels and quality 85; inference retains its original input. Recognition view reuses its identical preview encoding.
- Status polling allows one request at a time, times out after five seconds, and recovers after failure. Live status now distinguishes inference from capture-to-preview processing latency.
- Invalid/nonphysical calibration values are rejected; invalid existing calibration files are preserved. An explicit zero real-height override no longer silently substitutes the default.
- General-model output parsing failures preserve trained detections and recover on the next successful prediction. Failed optional inference time is included in latency. A wrong general checkpoint is rejected rather than interpreted as COCO.

The first five runtime failures were reproduced in failing regression tests before the fixes. Native Windows testing additionally reproduced the second utterance ending prematurely and a stale completion after mute.

## Measured performance

Both actual YOLO checkpoints enabled, CPU, 640 inference input, depth and speech disabled for timing isolation. Ten processed warmup frames, then 60 fresh publications per run. HTTP requests use a real localhost server. These are brief observations on a shared laptop, not sustained thermal or battery benchmarks.

| Workload | Processed FPS | Capture to HTTP observation, median / p95 | Median detector time |
| --- | ---: | ---: | ---: |
| Before, 1080p paced recording | 7.92 | 156 / 188 ms | 107 ms |
| After, first 1080p run | 5.22 | 234 / 267 ms | 159 ms |
| After, repeated 1080p run | 7.71 | 156 / 172 ms | 102 ms |
| Actual laptop camera, 640×480 | 7.49 | 156 / 187 ms | 127 ms |
| Experimental four CPU threads, recording | 6.84 | 164 / 219 ms | 109 ms |

No reliable FPS improvement is established. Four threads were not adopted. Median JPEG payload fell from 167,302 to 67,756 bytes on the recorded workload, approximately 59.5%. Duplicate transmission and stale queued work were removed. The repeated recorded run measured 125 ms median processing and 16 ms median preview encoding. Windows monotonic timestamps in this environment have roughly 16 ms granularity; do not interpret zero/small stage durations as exact.

Capture timestamps start after frame acquisition. Exposure/driver buffering, browser paint, remote network delay, speech playback, experimental MiDaS depth and an actual remote RF-DETR service are excluded. The recording is a 60-second slideshow workload built from real images, not a walking field trial. Camera frames were processed in memory and not saved.

## Functional evidence

- Final full suite: **179 tests passed, 12 subtests passed**. Includes deterministic delayed inference, cloud fallback, alert thresholds/cooldowns, approach re-alerts, pause/mute, upload validation, cleanup, keyboard logic and delayed status recovery. Controlled detector outputs are used for reproducible faults, separately from the real-model checks below.
- Actual trained plus COCO checkpoints: bus/person examples, additional general people, an obstacle labelled Object, duplicate suppression, and black/white blank negatives. The bus Object passed the central-obstacle alert gate. These selected examples do not establish general obstacle recall; COCO has no pillar class and cannot detect every unknown obstacle.
- Real upload routes: one selected labelled example per class plus a damaged image; **4/5 expected classes seen, Pillar missed**. This selected demonstration is not a class accuracy metric. A 10-second slideshow processed 87 frames; replay and temporary-file cleanup passed.
- Twelve online Singapore images and the real 9.47-second VivoCity clip decoded and completed with both detectors enabled. The recognition summary contains trained classes only; general detections are separately saved for the images. The pillar misses and lack of trained detections in the clip remain failures, not successful recognition claims.
- On the three primary column examples, SG001 produced a generic Object at confidence 0.392, below the 0.5 alert threshold, so no warning was issued; SG002 and SG013 produced no boxes. Generic fallback therefore did not supply a reliable column warning on these examples. Thresholds were not lowered to hide this failure.
- Real TCP HTTP multipart test: batch image plus corrupt image isolation; general Object alert rows; pause/resume, mute/unmute and rate; invalid control/video rejection; unavailable-cloud rejection; actual mall-video playback, pause, MJPEG stream, end, restart and return-to-camera. Camera capture was replaced with an idle source for these isolated upload tests; model inference was real.
- Native Windows voice: two consecutive utterances completed; a long utterance was deliberately interrupted; a subsequent recovery phrase completed; worker closed. Completion outcomes `[true, true, false, true]`. Native completion callbacks do not confirm audibility or intelligibility to a human. Interruption callbacks varied approximately 0.27–0.67 seconds across trials; there is no instant-stop guarantee.
- Distance: formula round trips over synthetic 1, 2, 3 and 5 metre geometries at 480/720/1080 heights, mixed-resolution normalization, invalid values, truncation and missing calibration. These test arithmetic, not measured camera accuracy. No fabricated calibration was installed.

## Remaining verification

1. Confirm the demo's actual laptop camera, fix lid angle/zoom/resolution, and calibrate using the measured height of a fully visible person/sign at a tape-measured distance. Test separate captures at 1, 2, 3 and 5 metres; report detection misses as well as metre error. Do not estimate range for Pillar/Stair or promise a maximum detection range without measurements. Current alerts primarily use relative box size; metric estimates apply only to known-height classes and are not spoken.
2. Measure object appearance/approach to audible alert using that setup. Existing cooldown tests and separate processing/voice timings do not measure this physical end-to-end delay.
3. Complete actual browser click-through and representative-user tests of keyboard, uploads, speech clarity, recovery and task completion. Windows Computer Use stopped because it could not determine Chrome's URL with enough confidence for its policy checks; no successful browser interaction is claimed in this run. Keyboard JavaScript and HTTP behavior were tested independently.
4. Reassess model generalization and pillar annotations/training separately. No improved weights or stronger detection accuracy resulted from this runtime pass.

## Reproduce

Install the repository requirements and test dependencies in a normal virtual environment; the sandbox validation used a workspace dependency target, with its `.pth` files processed for native Windows speech. Both real checkpoint files must be present under `models/local/` and `models/general/`.

```sh
python -m pytest -q
python scripts/benchmark_app.py --source 0 --general on --frames 60 --output artifacts/system-validation/laptop-camera.json
python scripts/benchmark_app.py --source path/to/video.mp4 --general on --frames 60 --output artifacts/system-validation/recorded.json
python scripts/probe_runtime_devices.py
python scripts/verify_general_runtime.py
python scripts/verify_real_uploads.py --general --output artifacts/system-validation/real-uploads
python scripts/verify_online_inputs.py --general --output artifacts/system-validation/sg-mall-uploads
# Start the server below, then run verify_http_runtime.py in another shell.
python scripts/verify_real_uploads.py --general --serve
python scripts/verify_http_runtime.py
```

The upload scripts require the previously acquired public media/supplied test bundle. Public example sources: [Ultralytics bus](https://raw.githubusercontent.com/ultralytics/assets/main/im/bus.jpg), [Ultralytics Zidane](https://raw.githubusercontent.com/ultralytics/assets/main/im/zidane.jpg). Singapore media attribution is in `docs/evidence/online-source-attribution.json`. The Windows implementation follows the documented [external event-loop API](https://pyttsx3.readthedocs.io/en/latest/engine.html).

Compact measurements and functional reports: [runtime evidence](evidence/runtime-validation-2026-11-17.json). Raw reports and recordings remain under ignored `artifacts/system-validation/`.
