# VisionAid: current project context

Software-only AAI3001 hazard-awareness prototype. The current training work is
in the five trained notebooks. The dataset-builder and training CLI files have
been removed; use the notebooks for training. Deployment utilities and the
prepared-dataset validator remain available.

## Final dataset and classes

Roboflow `potatos-workspace-kequm/version2-nvt5l` v4, scene-based split fingerprint
`49601080aac1f838190341d331785578`. Train/valid/test: 8247/925/920 images.
IDs must never be reordered: 0 Escalator, 1 People, 2 Pillar, 3 Stair,
4 Wet Floor Sign. `visionaid_config.json` owns class names, thresholds, severity,
model paths, input sizes, the alert gate and recorded benchmark metrics.

## Runtime architecture

The default is local YOLOv8n (`models/local/yolov8n_v4.pt`), Ultralytics 8.4.183,
640 px, confidence 0.35, NMS IoU 0.7. The optional remote detector is RF-DETR Nano
(`models/cloud/rfdetr-nano_v4.pth`), RF-DETR 1.11.2, 384 px, confidence 0.40.
RF-DETR has a separate Linux runtime; do not pass its `.pth` to Ultralytics YOLO.
The `POST /detect` request and response contract must not change.

All demos consume shared Detection/DetectionBatch records. Boxes are xyxy pixels
in the original frame. Checkpoint labels must match the canonical IDs at startup.
A second, local COCO YOLOv8n (`src/scene_detector.py`) adds general detections with
`class_id` -1: COCO people the trained model missed become `People`, everything
else is `Object`. It is not open-world.

The pipeline is capture, detector plus general model, optional relative depth
(experimental, off in first-test commands), then `AlertPolicy` in
`src/decision_engine.py`: alert gate, ranking, cooldown and one spoken-hazard
choice per frame, then audio. Local and cloud detections go through the same gate.

## Alert gate and audio policy

- Size is box height / frame height. Below the class `min_size` a hazard is far
  and never spoken. Left and right thirds need `min_size x side_multiplier`. Generic
  Objects are spoken only in the centre third. General-model detections need
  confidence `object_min_confidence` (0.5).
- Bands from `size / near_size`: near ("Warning"), mid-close ("Caution"), mid
  ("Note"). With depth on, bands come from depth only. Never speak metres.
- Any trained hazard outranks any Object. Cooldown key is (label, position, band):
  4 s trained, 8 s general. A proximity jump of 0.15 re-alerts early.
- Banner is the top gated hazard. Speech is the highest gated hazard off cooldown.
  The spoken hazard is marked in `/status` (`selected`) and highlighted. The
  speaker keeps one pending phrase, replaces it with the newest and drops it after
  1 s.
- Gate sizes in config are placeholders (`"calibrated": false`), except that Wet
  Floor Sign `min_size` was raised from 0.15 to 0.18 from test data (decision U1);
  with that, a Wet Floor Sign is never in the "Note" band. Data for them is
  in `docs/GUI_TEST_REPORT.md` (Phase D); a tape-measure log is in
  `docs/DISTANCE_TEST_SHEET.md`. `--no-gate` restores the old ungated behaviour.
- An optional distance estimate (`src/distance.py`, git-ignored
  `camera_calibration.json`) appears only in the debug panel for People and
  Wet Floor Sign, labelled as an estimate. It is never spoken.

The local UI remains the camera/audio host in cloud mode. Camera frames leave the
device only when cloud is explicitly selected; in local mode the cloud client is
never called (tested). With `--endpoint` both detectors are built and share one
general model; the app starts local unless `--backend cloud`. A cloud error or the
3 s deadline falls back to local (last good result held at most 1 s) and stays
local until a successful Retry; it never switches to cloud by itself, and the two
detectors are never fused. Switching mode clears the cooldown (U3). One request is
allowed at a time; newest frames replace buffered frames and obsolete cloud
responses are discarded. Recorded video sources are paced at their own frame rate.

## Web app endpoints (`src/webapp.py`)

- `GET /status`: live result plus `mode`, `mode_label`, `cloud_available`,
  `fallback_active`, `last_error`, `health` (list of `{code, severity, message}`),
  `paused`, `muted`, `speech_rate`, `speech_enabled`, `last_spoken`,
  `spoken_log` (last 10, newest first, memory only), `recognition` (every trained
  detection this frame), `view`, `source` (`type` camera or video, `origin`
  camera/file/upload/library, display `name` that is never a path, `duration`,
  `position`, `fps`, `ended`, `replay_available`), `video_summary` (per class:
  first/last seen, frames, best confidence), `frames_processed` and `accuracy_note`.
- `POST /mode` `{"backend": "local" | "cloud"}`: cloud only after its `/health`
  succeeds (this is also Retry cloud). 400 for anything else, 503 if unavailable.
- `POST /control`: any of `paused`, `muted` (booleans), `rate` (clamped to 120 to
  260 words per minute, default 180) and `action` (`test_voice`, `replay_last`,
  `clear_log`). Validated before anything changes; 409 when muted or speech is off.
- `POST /analyze`: up to 20 of multipart `image` (JPEG or PNG, 10 MB each), form
  `sample` (ids from `GET /samples`, images from `golden/images`), `demo` (ids from
  `GET /demo`) or `demo_class`. Optional `view` (`recognition` or `alert`, API
  default alert, page default recognition), `expected` (a class), `compare=1`,
  `speak=1`. Every response has `results` (one per image; a bad file is a row with
  `error`), `summary`, `accuracy_note`; one image also keeps the old top-level
  fields and HTTP error codes. Same detector, `decide()` and `annotate()` as the
  live loop with a fresh `AlertPolicy`; images are never written to disk. Images
  reach the cloud only in cloud mode or with `compare`.
- `POST /source`: multipart `video` (MP4, MOV or WebM, 150 MB, 120 s) or form
  `demo` (a library video id). The upload is saved under a random name in a
  private temp folder (`media.TempStore`) and handed to the live loop, which plays
  it like the camera. `POST /source/control` JSON `action` (`restart`, `camera`)
  and/or `view`. Recognition view skips the gate and never speaks; the camera
  always uses Alert view. The copy is deleted on replace, Back to camera, error,
  app stop or 10 minutes after the end.
- `GET /demo` lists `demo/<class>/` images and `demo/videos/` by id and name only,
  from an allow-list built per request; `GET /demo/thumb/<id>` is a 240 px JPEG.
- Shared-code rule: uploads, library files and videos go through the same
  detector, `decide()`, cooldown, speaker and annotation code as the camera. Add
  shared helpers (`media.py`, `capture.py`, `detection.py`) rather than copies.
- `GET /config`: priority order from the config severities (then Object), band
  labels and control ranges for the page.
- Health codes: `camera_missing` (camera retried every 2 s), `no_video` (3 s),
  `cloud_unreachable`, `general_unavailable`, `speech_unavailable`,
  `source_ended`, `pipeline_stopped`, `video_ended`, `video_deleted`. Missing weights or a class-order mismatch
  still stop the app at startup, as does a missing video file.
- Keyboard shortcuts live in `src/static/shortcuts.js` (tested with node).
  `scripts/check_contrast.py` and `scripts/check_layout.mjs` check the page.

## Training and evaluation

Do not retrain to integrate these checkpoints or change the finalized class set.
Keep notebook outputs and the recorded train/valid/test split. API credentials
come from the `ROBOFLOW_API_KEY` environment variable or Kaggle secret. The
previously exposed Roboflow key is still in git history and must be revoked in
its Roboflow account; source cleanup does not revoke it. The app itself does not
need a key. The five notebooks preserve their training/evaluation outputs.

`src/evaluation.py` reproduces the notebook's COCO AP and greedy IoU-0.5 alert
metrics. `scripts/verify_detector.py` verifies golden parity, evaluates a supplied
fixed test split and benchmarks batch-one inference. AP uses detections >=0.001;
alert metrics use the existing validation-selected threshold, never retuned on
test. Keep Ultralytics `.val()` results separately labelled if measured.
Score captions use `split_images.test` and `expected.test_map50` from the config.
`scripts/recall_by_size.py`, `scripts/compare_pillar.py` and
`scripts/replay_alerts.py` analyse size, Pillar and local versus cloud alerts.

## Boundaries and limitations

Depth is frame-relative, not metres. Distance bands come from box size and are
estimates. Pillar recall is weak (about 17 percent local, 29 percent cloud); a
missed pillar is a training limitation, not a bug. A Wet Floor Sign is a warning
sign, not proof of a spill. A lack of detections is not a declaration that a scene
is safe. This is supplementary hazard awareness. There is no measured distance,
stair direction classifier, floor-wetness sensor, OCR, voice input, item
identification, face recognition, route navigation, physical wearable build or
currency model.

See README.md for setup, commands, cloud container and export verification.
