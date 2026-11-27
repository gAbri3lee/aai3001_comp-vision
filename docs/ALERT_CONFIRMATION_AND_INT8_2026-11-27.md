# Alert confirmation, per-class thresholds and INT8 export (2026-11-27)

Three no-retraining improvements. They were first added as opt-in flags (sections 1–3 record that
evaluation). **Later the same day they became the defaults:**

- the INT8 model is the default local detector (`local.file`), committed to Git;
- 2-of-3 confirmation is on (`alert_gate.confirm_frames`) for camera and video. Uploaded photos are a single frame
  and skip it, otherwise a photo of a stair would be answered "No hazards present";
- `local.class_thresholds` is Stair 0.25, Wet Floor Sign 0.20, Escalator and People 0.35. Cloud RF-DETR keeps 0.40.

`--confirm 1/1 --conf 0.35 --model deploy/vercel/model/yolov8n_v4_four_class.onnx` restores the earlier behaviour.
None of this has been scored against labels yet; see the last paragraph of section 3.

## 1. K-of-N frame confirmation (`--confirm 2/3`)

A hazard is **spoken** only after the same class has been detected at about the same place
(same label, box IoU >= 0.2) in K of the last N detector frames. The banner still shows every gated
hazard immediately; only speech waits. `1/1` (the default) is the old behaviour.

```powershell
python src/webapp.py --no-depth --confirm 2/3
python scripts/replay_alerts.py --source "demo/videos/Walk to the subway stairs.mp4" --confirm 2/3 --output artifacts/replay.jsonl
```

Config equivalent (not set in `visionaid_config.json`):
`"alert_gate": { ..., "confirm_frames": {"required": 2, "window": 3} }`

**Replay of the 8 bundled demo videos** (124 s, every 3rd frame, FP32 ONNX, general model on):

| Setting | Spoken alerts | Per minute | Alerts started by a one-frame detection | First-alert delay |
|---|---|---|---|---|
| 1 of 1 (current) | 52 | 25.2 | 4 | – |
| **2 of 3** | **45** | **21.8** | **0** | median 125 ms (one detector frame) |

- 2-of-3 removed every alert that came from a detection seen in only one frame, and 13% of spoken alerts overall.
- Cost: each hazard is announced one detector frame later (about 125 ms here; about 70 ms at 15 detector FPS).
- Three hazards that appeared in just **one** frame of a whole clip were never spoken: two general "Object"
  boxes and one side stair in *Walk to the subway stairs*. A hazard the detector catches only once is not
  announced. This is the trade-off; `2/4` is the gentler option if that matters.
- Evidence: `docs/evidence/alert-confirmation-replay-2026-11-27.txt`; reproduce with
  `scripts/summarize_confirmation_replays.py` on replays named `<config>__<video>.jsonl`.

## 2. INT8 ONNX (`scripts/quantize_detector.py`)

Static INT8 (QDQ, per-channel weights) of the verified FP32 export. The box/score decoding, the last conv of
each box/class branch and the DFL conv stay FP32: quantising those removed every detection (0 of 20) in testing.

```powershell
python scripts/quantize_detector.py                                   # demo/ frames for calibration
python scripts/quantize_detector.py --calib data/v4_four_class_clean/train/images --calib-count 300   # preferred
python src/webapp.py --no-depth --model models/local/yolov8n_v4_four_class_int8.onnx
```

| | FP32 ONNX | INT8 ONNX |
|---|---|---|
| File size | 12.2 MB | 3.5 MB |
| Median time per frame (2 CPU threads, incl. pre/post-processing) | 56.8 ms | 38.2 ms (**1.49x faster**) |
| Golden parity against FP32 (20 held-out golden images) | – | **pass**: 19/20 strict matches, 20/20 at IoU 0.5 |
| Largest confidence change | – | 0.13 |
| Demo replay with 2-of-3 confirmation | 45 spoken alerts | 48 spoken alerts, same hazards announced |

Evidence: `docs/evidence/int8-quantization-2026-11-27.json`. The INT8 file is gitignored (`*.onnx`), so
regenerate it with the script.

**Before adopting INT8:** calibrate on the cleaned train split (above), then check validation mAP and safety
recall against FP32, e.g. `yolo val model=models/local/yolov8n_v4_four_class_int8.onnx data=data/v4_four_class_clean/data.yaml split=val imgsz=640 batch=1`.
These results compare INT8 with FP32 on 20 images, not with labels.

## 3. Per-class thresholds (`--class-conf`)

Lower cut-offs for single classes, **opt-in**: `visionaid_config.json` keeps 0.35 for every class.

```powershell
python src/webapp.py --no-depth --confirm 2/3 --class-conf "Stair=0.25" --class-conf "Wet Floor Sign=0.20"
```

`--class-conf CLASS=VALUE` is repeatable and applied after `--conf`. The detector's own pre-filter (NMS
confidence locally, the request's `confidence` for the cloud service) drops to the lowest class threshold, so
a lowered class actually reaches the per-class filter; every other class keeps its threshold. The permanent
equivalent is editing `local.class_thresholds` / `cloud.class_thresholds` in the config.

**Why revisit the threshold study.** `docs/evidence/validation-threshold-study.json` found no class where a
lower threshold qualified, but it was run on the earlier five-class model and judged single frames. With 2-of-3
confirmation, a low-confidence box seen in one frame is no longer spoken, so a lower threshold costs less.

**Replay of the 8 demo videos** (124 s, every 3rd frame, general model on). The clips are not frame-labelled, so
boxes in clips that contain the class count as hits, and boxes in clips that do not contain it as likely false alarms:

| Setting (FP32 ONNX) | Wet-floor clip: frames with a sign box | Stair clips: frames with a Stair box | Stair boxes in clips without stairs | Spoken alerts |
|---|---|---|---|---|
| 0.35, 1 of 1 (shipped) | 30/80 (38%) | 183/291 | 0 | 53 |
| 0.35, 2 of 3 | 30/80 | 183/291 | 0 | 46 |
| 0.25 both, 2 of 3 | 45/80 (56%) | 186/291 | 0 | 48 |
| 0.20 both, 2 of 3 | 54/80 (68%) | 187/291 | 1 (conf 0.21, gated) | 49 |
| 0.15 both, 2 of 3 | 61/80 (76%) | 190/291 | 2 | 51 |
| **Stair 0.25, sign 0.20, 2 of 3** | **54/80 (68%)** | **186/291** | **0** | **49** |

- No Wet Floor Sign box appeared in any other clip, even at 0.15. Stair's first false box appears at 0.20
  (a gated "Stair" in *Pedestrian - field sequence*), which only 2-of-3 confirmation keeps from being spoken.
  Hence the suggested **Stair 0.25, Wet Floor Sign 0.20**.
- Stair gains little: two clips already show stairs in nearly every frame, and *Walk to the subway stairs* stays
  at 2/104 frames. That clip is a model recall gap, not a threshold problem.
- Most extra sign boxes are still too small for the alert gate (`min_size` 0.18 of frame height): 5 → 9 gated
  frames. The box and banner appear sooner; speech still waits for the sign to be near.
- Evidence: `docs/evidence/class-thresholds-replay-2026-11-27.txt`; reproduce with
  `scripts/summarize_class_threshold_replays.py` on replays named `<config>__<video>.jsonl`.

**With INT8.** INT8 lowers some confidences, mostly Stair: at 0.35 it finds stairs in 100/120 frames of
*Passengers descending the central staircase* (FP32 119/120) and the sign in 37/80 frames. The suggested cut-offs
recover most of it (115/120 and 49/80). Re-checked on a second laptop: INT8 passes golden parity (19/20 strict,
20/20 at IoU 0.5) and is **1.86x** faster (27.1 vs 50.2 ms median, 2 threads).

**Before making these the default:** this is 124 s of unlabelled demo footage. On a machine with
`data/v4_four_class_clean`, cache low-confidence predictions with `scripts/verify_detector.py --evaluate` for the
four-class model, run `scripts/validation_study.py` on them, and read off validation precision and recall for
Stair at 0.25 and Wet Floor Sign at 0.20 against 0.35. Frame confirmation cannot be checked on still images.
