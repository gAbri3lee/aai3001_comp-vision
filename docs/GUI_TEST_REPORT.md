# VisionAid GUI test report

This report covers the GUI hand-over work on Gabriel's branch: the baseline test,
the alert gate, audio alignment, distance bands, the Pillar comparison and the
local versus cloud consistency checks. Each section gives exact commands and outputs.

## Phase A: baseline (before any behaviour change)

### Starting point

- Branch `gabriel`. `git pull --ff-only` brought in one new commit from Gabriel,
  `0ccfe52 Update README` (README only, 23 insertions, 24 deletions).
- Baseline commit: `0ccfe52cabeea3b175e76ecf5fedfbf32582f6bc`.
- Machine: Apple M4, 10 cores, 24 GB RAM, macOS 15.7.9 (arm64).
- Python 3.12.13 in `.venv` (created with `uv venv --python 3.12`; no `python3.12`
  was on PATH). Installed with
  `uv pip install --python .venv/bin/python -r requirements.txt -r requirements-test.txt`.
  On Apple Silicon the non Intel lines apply: Ultralytics 8.4.183, PyTorch 2.15.0,
  OpenCV 5.0.0, NumPy 2.5.4, Flask 3.1.3, ONNX Runtime 1.31.1.
- Docker 29.9.2 (linux/arm64) for the RF-DETR service.
- Weights copied from `~/Downloads/visionaid_pt` to `models/local/yolov8n_v4.pt` and
  `models/cloud/rfdetr-nano_v4.pth` (both git-ignored).

### Roboflow key handling

- The notebooks read `ROBOFLOW_API_KEY` from the environment, then fall back to a
  Kaggle secret of the same name (`kaggle_secrets.UserSecretsClient`). This holds
  for `rf-detr.ipynb`, `yolo11n.ipynb`, `yolo26n.ipynb`, `yolo8s.ipynb` and
  `yolov8n2.ipynb`. `comparecorrect(1).ipynb` does not use Roboflow.
- A search of every tracked file, notebook outputs included, found no key-shaped
  literal in the current tree.
- **The old key is still in git history.** The parent of commit `af3f87c` contains
  `os.environ["ROBOFLOW_API_KEY"] = "<20 characters>"` in the notebooks. Removing
  it from source does not revoke it. A human must revoke it in Roboflow. The value
  was not printed or stored during this work.
- The web app does not need the key. Only dataset download in the notebooks does.

### Automated tests

```
python -m pytest -q
44 passed, 12 subtests passed in 3.69s
```

Gabriel's report said 35 tests and 7 subtests. The suite has grown since.

### Local golden parity

```
python scripts/verify_detector.py --images golden/images \
  --reference golden/expected/local_yolov8n.json --frames 50 \
  --output artifacts/local_golden.json
```

PASS: 20 of 20 reference boxes matched, 0 extras, largest confidence difference
0.00005. Detector benchmark: median 40.4 ms, p95 41.7 ms, 24.7 FPS (CPU).

### Cloud golden parity

Port 9000 on this Mac is taken by a local php-fpm service, so the container was
mapped to host port 9100. Nothing in the app hardcodes 9000.

```
docker build -f cloud/Dockerfile -t visionaid-rfdetr .
docker run -d --name visionaid-cloud -p 127.0.0.1:9100:9000 \
  -v "$PWD/models/cloud:/app/models/cloud:ro" visionaid-rfdetr
curl http://127.0.0.1:9100/health
python scripts/verify_detector.py --runtime cloud --endpoint http://127.0.0.1:9100 \
  --timeout 60 --max-frame-age 60 --images golden/images \
  --reference golden/expected/cloud_rfdetr-nano.json --frames 50 \
  --output artifacts/cloud_golden.json
```

`/health` returned `status: ready`, model `rfdetr-nano`, five classes, input 384,
thresholds 0.40. PASS: 24 of 25 reference boxes, 0 extras (96 percent, the same
as Gabriel's result). Round trip from the client: median 139.5 ms, p95 153.5 ms.

### App smoke test (no camera)

A 40 second, 640x640, 10 FPS clip was made from the 20 golden images (2 seconds
per image) at `artifacts/golden_clip.mp4`. `/status` was polled once a second
for 30 seconds.

```
python src/webapp.py --port 5001 --no-depth --no-tts --source artifacts/golden_clip.mp4
python src/webapp.py --port 5001 --backend cloud --endpoint http://127.0.0.1:9100 \
  --no-depth --no-tts --source artifacts/golden_clip.mp4
```

**Bug found and fixed (blocking):** on the first run every poll returned
`"error": "Camera/source ended"`. `LatestFrameCapture` decoded the video file as
fast as possible, so the whole clip was gone before inference saw it. A camera
delivers frames in real time, but a file does not. The README advertises
`--source /path/to/video.mp4`, so this affected users too. Fix in `src/capture.py`:
file sources are paced at their own frame rate. Camera indices are unchanged.
New test: `tests/test_capture.py`. The fake capture in `tests/test_backends.py`
gained a `get()` method (every real `cv2.VideoCapture` has one).

Results after the fix (logs: `artifacts/status_local.jsonl`, `artifacts/status_cloud.jsonl`):

| | Local | Cloud |
|---|---|---|
| Loop rate | about 10 FPS | about 5 FPS |
| Detector timing | 73 to 120 ms (trained plus general model) | 138 to 145 ms round trip, plus about 55 ms local general model |
| Errors | none | none |

The two runs were not frame-aligned (start-up timing differs), so only a
qualitative comparison is possible here. Phase F does a frame-aligned comparison.
Observed:

- Cloud reported a `Pillar` (0.81) on the first image. Local reported nothing there.
- Cloud found `People` (0.80) on an image where local found nothing.
- Both produced `Object detected on your left` for the same image.
- Every People alert was `Note, People ...` whatever its size, and every trained
  hazard ended with `; distance unavailable` (problem 1 in the hand-over). Small
  background people at 0.39 to 0.40 confidence were announced like near ones
  (problems 2 and 5).

### Old-class and removed-feature search

Searched source, tests, scripts, docs, requirements, templates and notebooks.

| Term | Result |
|---|---|
| bicycle, motorcycle, traffic light, stop sign, fire hydrant, potted plant, couch, suitcase, backpack, houseplant, Open Images, fiftyone, hazard_yolov8n, HAZARD_CLASSES, 16-class lists | No hits. |
| `Chair` | Legitimate: `tests/test_training_dataset.py` uses it as a wrong label that must be rejected; a comment in `tests/test_scene_detector.py` describes a COCO chair. |
| `ocr`, `easyocr` | **Leftovers:** `CLAUDE.md` (lines 26, 55), `README.md` (154, 168, 173, 188), `requirements.txt` (`easyocr`), `scripts/benchmark_pipeline.py` (excludes list). Historical record only: `docs/VERIFICATION.md`, `docs/verification_results.json`. |
| `voice`, `SpeechRecognition`, `pyaudio`, `speech_recognition` | **Leftovers:** `requirements.txt` (`SpeechRecognition` and a comment), `requirements-voice.txt`, README voice sections, `CLAUDE.md` line 55. No code imports them. |
| `--no-ocr`, `--no-voice`, `--no-product-id`, `--mic-index` | **Leftovers:** dead flags in `src/webapp.py`, `src/live_pipeline_demo.py` and README commands; `tests/test_pipeline.py` sets `args.mic_index`. |
| `ImageNet`, item ID | **Leftovers:** `CLAUDE.md` lines 26 and 56, `scripts/benchmark_pipeline.py`. |
| `identify_item`, `scan_text` | Legitimate: `tests/test_pipeline.py` asserts those routes return 404. |
| `test.py` | Unrelated sensor plot at the repository root. |

Phase H handles all of these leftovers.

### Not testable here (manual)

Live camera view, audible speech, how alerts feel while walking and the tape-measure
distance test. See `docs/MANUAL_TEST_SHEET.md`.

## Phase B: alert gate

The gate lives in `src/decision_engine.py` as the pure function `assess()`. The web
app, the OpenCV demos (through `PipelineState`) and `scripts/benchmark_pipeline.py`
all call it, and so does the replay tool added in Phase F. Local and cloud
detections go through the same function. The settings are in
`visionaid_config.json` under `alert_gate` and are validated at start-up.
`--no-gate` restores the old announce-everything behaviour for comparison.

Rules, applied to every detection in this order:

1. A general-model detection (`class_id` -1) below `object_min_confidence` (0.5)
   is ignored as `low_confidence`.
2. `size` = box height / frame height. Below the class `min_size` the hazard is
   `far` and is never spoken.
3. Left or right third: Objects are always ignored (`side`). Trained classes need
   `size >= min_size x 1.5`.
4. Band from `size / near_size`: 1.0 or more is near ("Warning"), 0.6 or more is
   mid-close ("Caution"), otherwise mid ("Note"). With depth enabled, the band
   comes from depth only (decision D8: never mixed with box size).
5. Ranking: hazards that pass come first, any trained class before any Object,
   then `severity x (0.5 + 0.5 x proximity)`, where proximity is `min(1, size / near_size)`
   without depth.

A COCO person that no trained box overlaps (IoU < 0.5) becomes a `People`
detection with `class_id` -1 and `source: general` (D5). The cooldown key is now
`(label, position, band)`, 4 s for trained hazards and 8 s for general detections.
A hazard whose proximity rises by more than 0.15 re-alerts early.

Effect on the Phase A clip (local, 30 polls): background people at 0.39 to 0.40
confidence and side objects are no longer spoken. The stair image gives
`Warning, Stair ahead` instead of `Note, Stair ahead; distance unavailable`.

### Where the hand-over and the code disagreed

- The B3 table expects a centred Wet Floor Sign of height 0.20 to say "Note".
  The B1 rule gives 0.20 / 0.28 = 0.71, which is mid-close ("Caution"). The tests
  follow the rule and check both sides of the boundary (0.16 is "Note", 0.20 is
  "Caution").
- B2 says `scene_detector` should apply the Object confidence floor. It is applied
  in the gate instead, so low-confidence objects still reach `/status` and the
  "ignored" list with the reason `low_confidence` (Phase E), and so the rule is
  backend-independent. The floor also applies to general-model People.
- With the placeholder sizes, a side People box that passes the side rule
  (0.40 x 1.5 = 0.60) is already near, so a side person is only ever "Warning".
  Phase D revisits the sizes.

## Phase C: audio and banner alignment

- `AlertPolicy.process()` in `src/decision_engine.py` makes one decision per frame
  (decision D6). The banner is the top gated hazard. The speaker gets the
  highest-priority gated hazard whose cooldown has expired. The spoken hazard
  stays marked as `selected` while the same class and position stay in view.
  Banner, list, highlight and speech all come from the same frame's gated hazards.
- `src/tts.py` `Speaker`: at most one phrase waits. A newer phrase replaces it, a
  playing phrase is never interrupted, and a waiting phrase older than
  `max_pending_age` (default 1.0 s, a constructor argument) is dropped when its
  turn comes. The old queue dropped the newest phrase instead.
- `/status` adds `selected` (class, position, band, source, phrase, box),
  `spoken_at` (Unix time the phrase was handed to the speaker), `gate_enabled`,
  `gate_calibrated`, `depth_enabled` and `tts_enabled`. Each detection adds
  `position`, `size`, `band`, `source`, `gated` (true means it passed the gate),
  `reason` (far, side or low_confidence), `proximity`, `priority` and `phrase`.
  Detections stay in detector order. The original four fields are unchanged.
- `/video_feed` draws gated boxes coloured by band (red near, amber mid-close,
  green mid) and the spoken box in thick magenta with a `SPOKEN` tag.
  `/video_feed?debug=1` also draws ignored boxes in thin grey with their reason
  and size.

Known trade-off: a phrase dropped as stale has already used its cooldown, so that
hazard waits up to 4 s (8 s for Objects) before it is offered again. This only
happens when the speaker is still busy more than 1 s after the phrase was queued.

## Phase D: distance (estimates, not measurements)

### D1. Recall by object size

```
python scripts/recall_by_size.py --images "$BUNDLE/test_set/test/images"
python scripts/recall_by_size.py --backend cloud --endpoint http://127.0.0.1:9100 \
  --timeout 60 --max-frame-age 60 --images "$BUNDLE/test_set/test/images"
```

Outputs: `artifacts/recall_by_size/{local,cloud}/recall_by_size.{csv,md,json}`,
People example sheets `people_{small,large}_{hits,misses}.jpg`, and the prediction
caches `artifacts/{local,cloud}_test_predictions.json`. Overall recall per class
matches the saved notebook values (local People 58%, Pillar 17%, Stair 69%,
Wet Floor Sign 56%, Escalator 90%), which confirms the matching rule.

**How to read size.** A detector sees pixels, not fractions. The test images are
portrait phone frames letterboxed into 640x640, so a test box of height fraction
`s` is `s x 640` model-input pixels. A 1280x720 webcam frame becomes 640x360
inside the model input, so the same pixel height is `640/360 = 1.78` times larger
as a fraction of the webcam frame. The "webcam fraction" column is that
conversion, and it is the unit of `alert_gate.min_size`. Distances use the pinhole
formula `distance = focal_px x real_height_m / box_height_px` with an **assumed**
focal length of 914 px (720p webcam, about 70 degrees horizontal field of view)
and a 1.7 m person. They are estimates. A real camera must be measured
(`docs/DISTANCE_TEST_SHEET.md`).

### Recall by box height, local (yolov8n, thresholds 0.35)

Test box height fraction, model-input pixels, the same pixels as a fraction of a 1280x720 webcam frame, and the estimated distance of a 1.7 m person at that size (focal 914 px). Cells: recall (hits/boxes).

| Test fraction | Input px | Webcam fraction | Person distance (estimate) | Escalator | People | Pillar | Stair | Wet Floor Sign |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| 0.00-0.05 | 0-32 | 0.00-0.09 | 24.3 m to far | - | 0% (0/15) | 0% (0/6) | - | 0% (0/5) |
| 0.05-0.10 | 32-64 | 0.09-0.18 | 12.1 m to 24.3 m | 0% (0/1) | 19% (28/149) | 0% (0/21) | 0% (0/10) | 31% (21/67) |
| 0.10-0.15 | 64-96 | 0.18-0.27 | 8.1 m to 12.1 m | - | 55% (47/85) | 8% (1/12) | 33% (4/12) | 100% (34/34) |
| 0.15-0.20 | 96-128 | 0.27-0.36 | 6.1 m to 8.1 m | 0% (0/1) | 69% (37/54) | 0% (0/9) | 33% (7/21) | 100% (9/9) |
| 0.20-0.25 | 128-160 | 0.36-0.44 | 4.9 m to 6.1 m | 20% (1/5) | 82% (49/60) | 0% (0/15) | 47% (9/19) | - |
| 0.25-0.30 | 160-192 | 0.44-0.53 | 4.0 m to 4.9 m | 25% (1/4) | 88% (29/33) | 9% (1/11) | 71% (12/17) | 100% (1/1) |
| 0.30-0.40 | 192-256 | 0.53-0.71 | 3.0 m to 4.0 m | 100% (15/15) | 80% (41/51) | 50% (6/12) | 69% (22/32) | - |
| 0.40-0.50 | 256-320 | 0.71-0.89 | 2.4 m to 3.0 m | 75% (3/4) | 83% (38/46) | 42% (8/19) | 71% (17/24) | - |
| 0.50-0.60 | 320-384 | 0.89-1.07 | 2.0 m to 2.4 m | 100% (7/7) | 74% (23/31) | 43% (6/14) | 84% (27/32) | - |
| 0.60-0.80 | 384-512 | 1.07-1.42 | cut off by frame | 95% (58/61) | 91% (20/22) | 15% (4/27) | 74% (37/50) | - |
| 0.80-1.00 | 512-640 | 1.42-1.78 | cut off by frame | 95% (57/60) | 83% (19/23) | 0% (0/7) | 89% (63/71) | - |
| all | | | | 90% (142/158) | 58% (331/569) | 17% (26/153) | 69% (198/288) | 56% (65/116) |

Recall floor per class: the smallest size bucket where recall reaches 50 percent, for that bucket and for all larger boxes, with at least 5 boxes. Shown as a test fraction and as the equivalent webcam fraction (the unit of `alert_gate.min_size`):

- Escalator: test 0.30, webcam 0.53 (config min_size now 0.25)
- People: test 0.10, webcam 0.18 (config min_size now 0.4)
- Pillar: no bucket reaches 50 percent (config min_size now 0.35)
- Stair: test 0.25, webcam 0.44 (config min_size now 0.2)
- Wet Floor Sign: test 0.10, webcam 0.18 (config min_size now 0.15)

### Recall by box height, cloud (rfdetr-nano, thresholds 0.40)

Test box height fraction, model-input pixels, the same pixels as a fraction of a 1280x720 webcam frame, and the estimated distance of a 1.7 m person at that size (focal 914 px). Cells: recall (hits/boxes).

| Test fraction | Input px | Webcam fraction | Person distance (estimate) | Escalator | People | Pillar | Stair | Wet Floor Sign |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| 0.00-0.05 | 0-32 | 0.00-0.09 | 24.3 m to far | - | 0% (0/15) | 0% (0/6) | - | 0% (0/5) |
| 0.05-0.10 | 32-64 | 0.09-0.18 | 12.1 m to 24.3 m | 0% (0/1) | 32% (48/149) | 0% (0/21) | 0% (0/10) | 61% (41/67) |
| 0.10-0.15 | 64-96 | 0.18-0.27 | 8.1 m to 12.1 m | - | 55% (47/85) | 17% (2/12) | 25% (3/12) | 97% (33/34) |
| 0.15-0.20 | 96-128 | 0.27-0.36 | 6.1 m to 8.1 m | 0% (0/1) | 87% (47/54) | 0% (0/9) | 33% (7/21) | 100% (9/9) |
| 0.20-0.25 | 128-160 | 0.36-0.44 | 4.9 m to 6.1 m | 40% (2/5) | 95% (57/60) | 20% (3/15) | 58% (11/19) | - |
| 0.25-0.30 | 160-192 | 0.44-0.53 | 4.0 m to 4.9 m | 100% (4/4) | 91% (30/33) | 45% (5/11) | 53% (9/17) | 100% (1/1) |
| 0.30-0.40 | 192-256 | 0.53-0.71 | 3.0 m to 4.0 m | 80% (12/15) | 92% (47/51) | 50% (6/12) | 78% (25/32) | - |
| 0.40-0.50 | 256-320 | 0.71-0.89 | 2.4 m to 3.0 m | 50% (2/4) | 100% (46/46) | 63% (12/19) | 83% (20/24) | - |
| 0.50-0.60 | 320-384 | 0.89-1.07 | 2.0 m to 2.4 m | 100% (7/7) | 97% (30/31) | 50% (7/14) | 88% (28/32) | - |
| 0.60-0.80 | 384-512 | 1.07-1.42 | cut off by frame | 84% (51/61) | 100% (22/22) | 26% (7/27) | 78% (39/50) | - |
| 0.80-1.00 | 512-640 | 1.42-1.78 | cut off by frame | 95% (57/60) | 96% (22/23) | 29% (2/7) | 96% (68/71) | - |
| all | | | | 85% (135/158) | 70% (396/569) | 29% (44/153) | 73% (210/288) | 72% (84/116) |

Recall floor per class: the smallest size bucket where recall reaches 50 percent, for that bucket and for all larger boxes, with at least 5 boxes. Shown as a test fraction and as the equivalent webcam fraction (the unit of `alert_gate.min_size`):

- Escalator: test 0.30, webcam 0.53 (config min_size now 0.25)
- People: test 0.10, webcam 0.18 (config min_size now 0.4)
- Pillar: no bucket reaches 50 percent (config min_size now 0.35)
- Stair: test 0.20, webcam 0.36 (config min_size now 0.2)
- Wet Floor Sign: test 0.05, webcam 0.09 (config min_size now 0.15)

### Operating range: how close must an object be to be recognised?

"Usable" means recall of at least 50 percent from that size upwards. Webcam
distances are estimates for the assumed camera.

| Class | Local YOLOv8n usable from | Cloud RF-DETR usable from | Estimated distance (webcam) |
|---|---|---|---|
| People (1.7 m) | test 0.10, webcam 0.18 (55 to 88 percent above) | test 0.10, webcam 0.18 (87 to 100 percent from 0.15) | about 12 m local, 80 percent or more within about 6 m; cloud 87 percent or more within about 8 m |
| Wet Floor Sign (0.65 m) | test 0.10, webcam 0.18 (100 percent above, few large examples) | test 0.05, webcam 0.09 | about 4.6 m local, about 9 m cloud |
| Stair | test 0.25, webcam 0.44 | test 0.20, webcam 0.36 | no fixed size, so no distance |
| Escalator | test 0.30, webcam 0.53 (only 11 smaller boxes) | test 0.30, webcam 0.53 | no fixed size, so no distance |
| Pillar | never steadily 50 percent; best 42 to 50 percent at test 0.30 to 0.60 | 45 to 63 percent at test 0.25 to 0.60 | no fixed size, so no distance |

### Is a person 1 to 2 m away in the background detected, or only one right in front?

From the data, the model is **not** limited to the person right in front.

- With a 720p webcam, a 1.7 m person needs to be about 2.2 m away to fit in the
  frame at all (`914 x 1.7 / 720`). A person 1 to 2 m away is therefore never "in
  the background": they fill the frame and their box is cut off at the edges. Test
  boxes that tall (test fraction 0.60 or more) are found 83 to 91 percent of the
  time locally and 96 to 100 percent on the cloud model. Most of the misses there
  are crowds, where one predicted box covers two labelled people
  (`artifacts/recall_by_size/local/people_large_misses.jpg`).
- People further back are detected too: locally 80 to 88 percent at about 3 to 6 m,
  55 percent at about 8 to 12 m and 19 percent at about 12 to 24 m. The cloud
  model finds 87 to 95 percent out to about 8 m. Examples:
  `artifacts/recall_by_size/local/people_small_hits.jpg` (small background people
  at confidence 0.36 to 0.65) and `people_small_misses.jpg`.
- This is exactly why the gate is needed. Before Phase B every one of those
  background people was spoken as "Note, People ...". Now a person below 0.40 of
  the frame height (about 5.4 m) is listed as far and stays silent.

### Proposed min_size (not applied)

The recall floor answers "below what size are detections unreliable". It does not
answer "how far away should a hazard be before we speak". The gate needs the
larger of the two. The floors in webcam units compared with the current
placeholders:

| Class | Local floor | Cloud floor | Current `min_size` | Suggestion |
|---|---:|---:|---:|---|
| People | 0.18 | 0.18 | 0.40 (about 5.4 m) | Keep 0.40. It is a speaking-distance choice, well above the floor. |
| Wet Floor Sign | 0.18 | 0.09 | 0.15 (about 5.5 m) | 0.18 (about 4.6 m), the local floor, so both backends speak at the same size. |
| Stair | 0.44 | 0.36 | 0.20 | Keep 0.20. Stairs are the most severe class, and a detected distant stair is still worth an early "Note". Raising to 0.44 would silence stairs until they are close. |
| Escalator | 0.53 | 0.53 | 0.25 | Keep 0.25 for the same reason. The floor rests on only 11 small boxes. |
| Pillar | none | none | 0.35 | Keep 0.35. |

The config is unchanged and keeps `"calibrated": false` until Nigel decides and the
tape-measure sheet is filled in.

Caveats: the test images come from phone cameras held in portrait, not the
deployment webcam. The pinhole conversion assumes one focal length. Box heights of
objects cut off by the frame understate their size.

### D2. Calibration and the debug distance estimate

- `scripts/calibrate_distance.py --camera 0 --class People --distance 3` (or
  `--image`) measures `focal_px = box_height_px x distance_m / real_height_m` and
  appends a sample to the git-ignored `camera_calibration.json`. The app uses the
  median sample. Boxes cut off by the frame are refused.
- When that file exists, `/status` adds `distance_est_m` and `distance_note` to
  People and Wet Floor Sign detections (`src/distance.py`). The estimate is shown
  only in the debug panel, labelled "estimate", and never spoken. Boxes cut off by
  the top or bottom edge get no estimate.
- `docs/DISTANCE_TEST_SHEET.md` is the tape-measure log (1, 2, 3, 5 and 8 m per class).

### D3. Pillar: local versus cloud

```
python scripts/compare_pillar.py --images "$BUNDLE/test_set/test/images" \
  --endpoint http://127.0.0.1:9100
```

All 105 test images with a Pillar label (153 boxes) and all 815 images without
one. Outputs: `artifacts/pillar/pillar_comparison.{md,json}` and contact sheets
`artifacts/pillar/pillar_{local,cloud}_{hits,misses}.jpg`.

| | local (yolov8n, 0.35) | cloud (rfdetr-nano, 0.40) |
|---|---:|---:|
| Pillar boxes (images) | 153 (105) | 153 (105) |
| Recall | 17.0% (26 hits, 127 misses) | 28.8% (44 hits, 109 misses) |
| Precision | 54.2% (22 false of 48) | 51.8% (41 false of 85) |
| Images without a pillar that got a Pillar box | 14 of 815 | 29 of 815 |
| Mean confidence of hits | 0.56 | 0.61 |
| Missed pillars covered by a general Object box (conf >= 0.5, IoU >= 0.5) | 3 of 127 | 3 of 109 |
| Recall, height 0.00-0.05 | 0% (0/6) | 0% (0/6) |
| Recall, height 0.05-0.10 | 0% (0/21) | 0% (0/21) |
| Recall, height 0.10-0.15 | 8% (1/12) | 17% (2/12) |
| Recall, height 0.15-0.20 | 0% (0/9) | 0% (0/9) |
| Recall, height 0.20-0.25 | 0% (0/15) | 20% (3/15) |
| Recall, height 0.25-0.30 | 9% (1/11) | 45% (5/11) |
| Recall, height 0.30-0.40 | 50% (6/12) | 50% (6/12) |
| Recall, height 0.40-0.50 | 42% (8/19) | 63% (12/19) |
| Recall, height 0.50-0.60 | 43% (6/14) | 50% (7/14) |
| Recall, height 0.60-0.80 | 15% (4/27) | 26% (7/27) |
| Recall, height 0.80-1.00 | 0% (0/7) | 29% (2/7) |

Local recall (17.0 percent) matches the saved result exactly. The cloud figure
over HTTP (28.8 percent) is slightly below the saved 29.4 percent because frames
are JPEG re-encoded on the way to the service. The miss sheets show three kinds of
miss: thin bollards and poles labelled Pillar where the model boxed the
neighbouring pole (IoU below 0.5), close-up pillar bases cut off by the frame,
and plain misses. Some "false" Pillar boxes are unlabelled poles.

**What the app does when a pillar is not detected:** it says nothing about it.
The general COCO model has no pillar class. It covered only 3 of 127 local misses
(3 of 109 cloud) with an Object box at confidence 0.5 or more, and that box is
spoken as "Object detected ahead" only if it also passes the gate (centre third,
large enough). No Pillar box does not mean there is no obstacle.

## Phase E: page

`src/templates/index.html`:

- Banner: the top gated hazard, or "No hazards", coloured by that hazard's band.
- Hazards list: one row per gated detection in priority order, with class (marked
  "general model" for COCO people), confidence, position, band, and a magenta
  "spoken" marker on the hazard being announced.
- "Ignored detections" is a collapsed section listing far, side and low-confidence
  detections with their reason.
- `/?debug=1` switches the video to `/video_feed?debug=1` (ignored boxes drawn
  grey), opens the ignored section and adds box size and, once calibrated, the
  distance estimate to each row.
- Status line: backend, model, inference or round-trip time, alert gate on or off
  (and "uncalibrated sizes" while `calibrated` is false), depth and speech state.
- Footer adds "Distance is an estimate from box size, not measured metres." and
  "No hazards does not mean the scene is safe". The em dash in the tagline was removed.

Checked with headless Chrome against the Phase A clip: banner `Warning, Stair ahead`,
the stair box drawn magenta with a `SPOKEN` tag, and the list row marked "spoken".
