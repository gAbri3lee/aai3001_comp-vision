# VisionAid

AAI3001 hazard-detection prototype. The live GUI reports **Escalator, People,
Stairs and Wet Floor Sign** using local YOLOv8n or a separate RF-DETR Nano
service. Local mode runs an **INT8 ONNX** export of YOLOv8n by default (3.5 MB, about 1.5–1.9× faster on CPU
than full precision). The Vercel photo demo in this repository uses the full-precision four-class YOLOv8n ONNX export.

| What to try | Where to run it | Input |
| --- | --- | --- |
| **Local YOLOv8n** | Start the browser GUI on your Mac or Windows PC | Webcam, images or video |
| **RF-DETR Nano** | Connect the same GUI to an RF-DETR service and click **Cloud** | Webcam, images or video |
| **Vercel Upload Image** | Open the Vercel link below in your browser | One photo at a time |

## Local setup and launch

Run the terminal commands from the repository root, the folder containing this
README and `src/`. Use **Terminal** on macOS and **PowerShell** on Windows.

### 1. Prepare the Python environment

Install **Python 3.12**. Reuse an existing `.venv` if its dependencies are already
installed.

**macOS**

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

If `uv` is used to manage Python on macOS, run
`uv venv --python 3.12 .venv` and
`uv pip install --python .venv/bin/python -r requirements.txt` instead. Use the
`uv pip` command for an existing `.venv` that reports `No module named pip`.

**Windows**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. Set Model Weights

**Fresh clone, local demo:** nothing to copy. The repository includes the default local detector,
`models/local/yolov8n_v4_four_class_int8.onnx` (INT8), and `requirements.txt` installs ONNX Runtime for it.
Go straight to step 3. You can run the photo demo without a webcam: open **Upload → Demo library** and click
**Escalator landing**. Recognition view should display **Escalator** and “Check the direction before stepping on.”

**Intel Mac:** the pinned ONNX Runtime has no Intel macOS build, so use the PyTorch checkpoint below and add
`--model models/local/yolov8n_v4_four_class.pt` to the step 3 command.

**Full-precision PyTorch model or self-hosted RF-DETR:** Use the following handoff weights.

Copy the weights from the project handoff to the paths below. Git ignores these
files.

| File | Used by |
| --- | --- |
| `models/local/yolov8n_v4_four_class.pt` | Local YOLOv8n in full precision with `--model models/local/yolov8n_v4_four_class.pt`; needed on Intel Macs |
| `models/cloud/rfdetr-nano_v4_four_class.pth` | RF-DETR Docker service, needed when you run the service yourself |

The full-precision ONNX export used by Vercel, `deploy/vercel/model/yolov8n_v4_four_class.onnx`, also ships
with Git and works with `--model`.

If you only have the original `yolov8n_v4.pt`, place it in `models/local/` and
create the four-class checkpoint once. The script projects the existing model
to four classes for inference and keeps the original weights file. Run it only
if the four-class checkpoint is missing. The `artifacts` folder is for the
conversion report.

**macOS**

```bash
mkdir -p artifacts
.venv/bin/python scripts/remove_pillar_head.py
```

**Windows**

```powershell
New-Item -ItemType Directory -Force artifacts | Out-Null
.\.venv\Scripts\python.exe scripts/remove_pillar_head.py
```

The GUI also runs a supplementary COCO YOLOv8n for extra People and Object boxes. On CPU it uses the INT8 export
committed at `models/general/yolov8n_int8.onnx`. On other devices, or without ONNX Runtime, it downloads
`models/general/yolov8n.pt` on first use; `--general-model models/general/yolov8n.pt` forces that file. If the general
model cannot load, the trained hazard detector still runs.

### 3. Start the browser GUI

**macOS**

```bash
.venv/bin/python src/webapp.py --port 5001 --no-depth
```

**Windows**

```powershell
.\.venv\Scripts\python.exe src/webapp.py --port 5001 --no-depth
```

Open [the local GUI](http://127.0.0.1:5001). Look for **LOCAL, YOLOv8n** and
the live feed. If the camera is unavailable, check camera permissions for
Terminal or Python and try **Upload** while you troubleshoot. Use **Test voice**
to check speech. Add `--no-tts` for a silent run. These commands skip the
experimental MiDaS depth stage to keep the demo responsive. Press **Ctrl+C** to
stop the GUI before starting it again on the same port.

### Alert settings (defaults since 2026-11-28)

| Setting | Default | Effect |
| --- | --- | --- |
| Frame confirmation | 2 of 3 | Camera and video: a hazard is spoken only after it is seen at about the same place in 2 of the last 3 detector frames. Removes alerts caused by a one-frame detection; each alert comes about one detector frame later. The banner is not delayed. Uploaded photos are a single frame and are not confirmed. |
| Confidence cut-offs | 0.35 for every class | A box is used only at or above this confidence. Cloud RF-DETR keeps 0.40. Lower Stair and Wet Floor Sign cut-offs were tried and reverted; see below. |

Both are in `visionaid_config.json` (`alert_gate.confirm_frames`, `local.class_thresholds`). Override them per run,
also in `scripts/replay_alerts.py` and `scripts/verify_detector.py`:

| Flag | Example |
| --- | --- |
| `--confirm K/N` | `--confirm 1/1` turns confirmation off |
| `--class-conf CLASS=VALUE` (repeatable) | `--class-conf "Stair=0.25"` |
| `--conf VALUE` | `--conf 0.5` sets every class |

`--confirm 1/1` restores the behaviour before 2026-11-27. Stair 0.25 and Wet Floor Sign 0.20 were the defaults from
27 to 28 November 2026. A sweep on the labelled validation split found they did not meet the team's rule for lowering
a cut-off, so every class is back at 0.35 ([results](#int8-default-and-alert-defaults-2026-11-28)).

## Cloud detector with RF-DETR

RF-DETR is a separate service. Start Docker Desktop on your device
and run the commands below from the repo root. The build command is the same
on Mac and Windows.

```sh
docker build -f cloud/Dockerfile -t visionaid-rfdetr .
```

**macOS**

```bash
docker run --rm -p 127.0.0.1:9100:9000 --mount "type=bind,source=$PWD/models/cloud,target=/app/models/cloud,readonly" visionaid-rfdetr
```

**Windows**

```powershell
$cloudModels = Join-Path $PWD.Path 'models\cloud'
docker run --rm -p 127.0.0.1:9100:9000 --mount "type=bind,source=$cloudModels,target=/app/models/cloud,readonly" visionaid-rfdetr
```

Leave the service running. In a second terminal, wait for `/health` to return
`"status": "ready"` and `"model": "rfdetr-nano"`, then launch the GUI with its
endpoint. Stop any earlier GUI instance on port 5001 first.

**macOS**

```bash
curl http://127.0.0.1:9100/health
.venv/bin/python src/webapp.py --port 5001 --no-depth --endpoint http://127.0.0.1:9100
```

**Windows**

```powershell
curl.exe http://127.0.0.1:9100/health
.\.venv\Scripts\python.exe src/webapp.py --port 5001 --no-depth --endpoint http://127.0.0.1:9100
```

Open [the local GUI](http://127.0.0.1:5001), click **Cloud**, and check for
**CLOUD, RF-DETR Nano**. Click **Local** to return to YOLO. Without
`--endpoint`, **Cloud** is disabled. If the service drops out, the GUI falls
back to YOLO. Use **Retry cloud** once the service is ready again.

If a teammate already hosts RF-DETR, use their service's base URL for
`--endpoint` and skip Docker. The GUI sends JPEG images or video frames to
that service only in Cloud mode or when you ask it to compare both models.
The service loads the native four-class RF-DETR checkpoint. To build it from the original project weights, run `python scripts/project_four_class_rfdetr.py` in the project environment. The GUI connects to the service's `/health` and `/detect` routes.

## Vercel Upload Image

1. Open [VisionAid Photo Check on Vercel](https://aai3001-gwup54owc-computer-vision2.vercel.app/).
   If prompted, sign in to Vercel with an account that has access to the deployment.
2. Click **Choose a photo** and select a JPEG from [the bundled golden images](golden/images)
   or from the test folder.
3. Look for the annotated photo, class labels and confidence, the highest-priority
   alert or **No hazards detected**, and the model and request timings. **Read the
   result aloud** uses your browser's audio feature.

The browser resizes each photo and sends it as a JPEG to `/api/detect`. A Python
function runs YOLOv8n ONNX. The four-class deployment build uses the four-class export evaluated below. To check which build a hosted URL serves,
open [the Vercel health endpoint](https://aai3001-gwup54owc-computer-vision2.vercel.app/api/health)
while signed in and check the returned class list as well as the model name. This demo accepts
still images only. To check a frame from a video on Vercel, save a screenshot
from the clip as a photo. Use the live GUI for video playback and timed alerts.
[Cloud branch deployment guide](https://github.com/gAbri3lee/aai3001_comp-vision/blob/cloud-deployment/deploy/vercel/README.md)

## Test sample images and videos in the live GUI

- **Images.** Under **Upload → Images**, click **Choose images** or select a
  bundled photo from the sample menu and click **Analyse sample**. Try the
  escalator image above in **Recognition** view and check its label and box. The file
  picker accepts up to 20 JPEG or PNG files, each up to 10 MB. Use **Alert** view
  to check which detections would trigger a live alert. A small or off-centre
  detection can appear in Recognition and be filtered in Alert.
- **Compare detectors.** With RF-DETR connected, tick **Compare local and cloud
  side by side** before uploading the same image batch. Check each model's
  detections and results. Use **Back to live** after the image check.
- **Video.** Under **Upload → Video**, click **Choose video** and pick a short
  landscape MP4, MOV or WebM (up to 150 MB and 120 seconds). The clip plays
  through the selected detector. Check **Pause**, **Recognition** and **Alert**
  views, the end summary, **Replay** and **Back to camera**. Repeat with the
  other detector to compare. See the [video demo guide](docs/DEMO_VIDEO_GUIDE.md)
  for recording a clip.

The [20 golden JPEGs](golden/images) ship with Git. The cleaned held-out split
is available from the separate `visionaid-four-class-dataset.zip` handoff.
Arrange it under `data/v4_four_class_clean/` so the photos are in
`data/v4_four_class_clean/test/images/` and labels are in the sibling `labels/`
folder. Use the same photos when comparing results. A few selected uploads
are demonstration checks. Use the labelled split for a model evaluation.

### Play sample videos from the Demo library

1. In the local GUI, open **Upload → Demo library** and click **Reload**.
2. Under **Videos**, click **Play** beside the clip you want to test. Choose **Following a commuter down the escalator** or **Passengers descending the central staircase**.
3. Use **Alert** to check the banner and spoken alert, or **Recognition** to
   check detection boxes and final classification of the hazard type.
   When the clip ends, click on **Replay** or **Back to camera** for live view.

The library reads videos from `demo/videos/`. Bundled clips are included in Git. [Demo library guide](demo/README.md)

## Verification and Model Testing Results

### Detection benchmarks (by Model and Class)

The three pipelines in the original-split study were scored on the **same 920 original test images** with the
**four active classes**. The Pillar boxes in those images are excluded and
Pillar predictions are dropped. That study used the original five-class RF-DETR checkpoint; its configured
class mapping removes Pillar. No model was retrained for this.

| Model | Test split | mAP50 | mAP50–95 |
| --- | --- | ---: | ---: |
| Local YOLOv8n (PyTorch) | Original split, 920 images, four classes | 76.4% | 49.8% |
| Local YOLOv8n ONNX (four-class export) | Original split, 920 images, four classes | 76.4% | 49.8% |
| RF-DETR Nano (cloud) | Original split, 920 images, four classes | 83.6% | 51.4% |

| Class | YOLOv8n AP50 | YOLOv8n AP50–95 | RF-DETR AP50 | RF-DETR AP50–95 |
| --- | ---: | ---: | ---: | ---: |
| Escalator | 96.3% | 60.8% | 93.1% | 55.0% |
| People | 69.3% | 37.5% | 80.7% | 39.9% |
| Stairs | 77.5% | 56.3% | 79.7% | 58.6% |
| Wet Floor Sign | 62.4% | 44.5% | 81.1% | 52.0% |

The ONNX export matches PyTorch to four decimal places in every class and passes
all 20 golden checks, so it shares the YOLOv8n columns. Evidence, commands and
weight checksums: [four-class test results](docs/evidence/four-class-test-2026-11-19.json).

A local CPU rerun of the three original pipelines reproduced every headline figure above. The native four-class RF-DETR checkpoint used by this branch also scored 51.40% mAP50–95 on the same 920 images, against 51.37% for the legacy checkpoint. Both round to 51.4%. The dataset includes 151 images with no retained hazard boxes. The three runtimes were evaluated on those backgrounds too. [Independent rerun and weight hashes](docs/evidence/four-class-920-reproduction.json)

The metric definition changed from the earlier five-class benchmark. The archived five-class means
(YOLOv8n 65.8% / 41.9%, RF-DETR 73.9% / 44.7% mAP50 / mAP50–95) include Pillar,
which scored only 23.4% and 35.1% AP50. These are historical
development test images, not an unseen final evaluation.

AP50 measures average precision for one class at 0.50 intersection over union
(IoU). mAP50 averages that score across classes. mAP50–95 averages across classes
and IoU thresholds from 0.50 to 0.95.

### Four-class demo benchmark

All three runs used the same 769 cleaned test images, ground-truth boxes and COCO evaluator. Precision and recall count box matches at IoU 0.50 using each model’s confidence threshold at the time (0.35 for YOLO, the current default). The two YOLO rows run the same learned detector in different formats.

| Runtime | Confidence | Precision | Recall | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|---:|
| YOLOv8n PyTorch | 0.35 | 79.6% | 63.7% | 77.4% | 50.9% |
| YOLOv8n ONNX | 0.35 | 79.6% | 63.7% | 77.4% | 50.9% |
| RF-DETR Nano | 0.40 | 85.4% | 71.7% | 84.5% | 52.3% |

#### Per-class results

We excluded Pillar from the final scope because the original local model detected only 17.0% of labelled pillars at its alert threshold on the earlier five-class test. A reliable pillar warning would need stronger evidence.

| Tested class | Ground-truth boxes | Local YOLO precision | Local YOLO recall | Local YOLO AP50 | RF-DETR AP50 |
|---|---:|---:|---:|---:|---:|
| Escalator | 103 | 92.6% | 84.5% | 94.4% | 91.8% |
| People | 559 | 75.3% | 58.0% | 69.4% | 80.9% |
| Stair | 268 | 80.4% | 67.2% | 76.3% | 79.4% |
| Wet Floor Sign | 52 | 92.1% | 67.3% | 69.6% | 85.7% |

#### Model and deployment notes

The old README quoted historical tests on 920 images and five classes. Removing Pillar from the dataset left those measurements and the original classifier weights intact. The active YOLO PyTorch and ONNX heads now emit four classes. For RF-DETR we retained the four foreground rows and the background slot across its encoder and decoder classifiers. All three use the same four class IDs. This conversion involved no retraining. Older hosted services can still return five classes through the explicitly validated compatibility mapping. Check the hosted service’s version before quoting local results for it. The Vercel function code was also evaluated locally on all 769 images and matched the displayed ONNX metrics to one decimal place.

[Full metrics, model hashes and test-set fingerprint](docs/evidence/four-class-demo-benchmark.json) · [Historical five-class evidence](docs/verification_results.json)

### INT8 default and alert defaults (2026-11-28)

The default local detector is a static INT8 quantisation of the four-class ONNX export, made with
`scripts/quantize_detector.py`. The detection head's box decoding stays in full precision.

| | Full-precision ONNX | INT8 ONNX (default) |
| --- | ---: | ---: |
| File size | 12.2 MB | 3.5 MB |
| Median time per frame, 2 CPU threads | 50–57 ms | 27–38 ms (1.5–1.9× faster) |
| Agreement with full precision, 20 golden images | – | 20/20 detections at IoU 0.5; largest confidence change 0.13 |

**Accuracy on the 769-image test split** (the same split as the benchmark above, all classes at 0.35):

| Model | Precision | Recall | mAP50 | mAP50-95 | Safety-class recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full-precision ONNX | 79.6% | 63.7% | 77.4% | 50.9% | 73.0% |
| **INT8 ONNX (default)** | **79.0%** | **63.7%** | **77.5%** | **50.8%** | **74.2%** |

INT8 is as accurate as full precision. Per class, its recall is equal or higher for Escalator (87.4% vs 84.5%),
Stair (67.9% vs 67.2%) and Wet Floor Sign (67.3%), and 0.9 points lower for People (57.1% vs 58.0%).
[Per-class table](docs/evidence/int8-test-evaluation-2026-11-28.txt)

**Lower cut-offs for Stair and Wet Floor Sign (tried, then reverted).** Stair 0.25 and Wet Floor Sign 0.20 were
tried because they found more stairs and signs in the demo videos. They were then checked on labelled images
with the INT8 model.

The validation split (859 images) was used to choose. The team's rule from `scripts/validation_study.py` was applied:
recall must gain at least 5 points and precision must stay at or above max(75%, current − 2 points).
No class met it:

| Class (validation) | 0.35 | Lower cut-off | Result |
| --- | --- | --- | --- |
| Stair | P 76.0%, R 67.1% | 0.25: P 69.0%, R 70.6% | +3.5 recall for −7.0 precision; fails |
| Wet Floor Sign | P 89.2%, R 85.3% | 0.20: P 81.7%, R 85.3% | No recall gain at any cut-off from 0.10 to 0.45; fails |
| Escalator, People | – | – | No cut-off meets the rule |

The test split showed the same thing (INT8). Stair at 0.25 gave recall 70.5% and precision 73.5%, against 67.9% and
79.8% at 0.35. Wet Floor Sign at 0.20 gave the same recall with 10 false alarms instead of 3. Every class is
therefore back at 0.35. [Validation sweep](docs/evidence/class-thresholds-validation-sweep-2026-11-28.json)

**General object model, also INT8 (2026-11-28).** Every frame also runs a supplementary COCO YOLOv8n for extra
People and Object boxes. It was the largest remaining cost (56 ms in PyTorch), so it is now quantised the same way
(`scripts/quantize_general_model.py`): 34.7 ms, 1.6× faster.

| Whole app on a demo video, depth off | Frames per second |
| --- | ---: |
| Full-precision hazard model + PyTorch general model (before) | 10 |
| INT8 hazard model + PyTorch general model | 12–15 |
| **INT8 hazard model + INT8 general model (default)** | **20** |

**Depth on every 3rd frame (2026-11-28).** Depth (MiDaS) costs about 100 ms per frame, so with depth on the app
reached only 5–7 FPS. In live video it now runs on every 3rd frame (`--depth-every 3`), and the frames in between
rank the current boxes against the last depth map. Relative depth changes little in 2 frames (about 150 ms).
The map is recomputed early when it is over 1 s old, when the frame size changes or when the source changes.
Uploaded photos always get fresh depth.

| Depth on, measured as frames per second over 150 video frames | FPS |
| --- | ---: |
| Every frame (`--depth-every 1`, the earlier behaviour) | 7.5 |
| Every 2nd frame | 11.2 |
| **Every 3rd frame (default)** | **13.0** |
| Depth off, for comparison | 17.7 |
[Measurements](docs/evidence/app-fps-2026-11-28.txt)

The general model's person boxes supply most of the app's People detections. On the test split, INT8 changes their
People recall from 85.3% to 83.0% (13 fewer of 559 people) and precision from 60.8% to 63.3%. Its other borderline
changes are boxes crossing the 0.35 cut-off (0.27–0.43 confidence). Hazard classes are unaffected. Object boxes
cannot be scored, because the labels cover only the four classes.
`--general-model models/general/yolov8n.pt` restores the full-precision model.
[Evidence](docs/evidence/general-model-int8-2026-11-28.json)

These are single images, so 2-of-3 confirmation could not be scored on them. It was chosen by replaying the 8 bundled
demo videos (124 s):

| Setting (full-precision ONNX) | Spoken alerts | Alerts from a one-frame detection | Wet-floor clip: frames with a sign box | Stair boxes in clips without stairs |
| --- | ---: | ---: | ---: | ---: |
| Earlier defaults (0.35, no confirmation) | 53 | 4 | 30/80 | 0 |
| **Current defaults:** 2 of 3, all 0.35 | 46 | 0 | 30/80 | 0 |
| Trial (reverted): 2 of 3, Stair 0.25, sign 0.20 | 49 | 0 | 54/80 | 0 |

The lower cut-offs found the sign in more video frames, but the labelled images did not confirm a gain. The clips are not frame-labelled, so these are indicative only. [Write-up and evidence](docs/ALERT_CONFIRMATION_AND_INT8_2026-11-27.md)

#### Demo steps

1. In **Upload → Demo library**, analyse one named photo in each class folder. Check the label and the action beside it.
2. Play **Stairs and pedestrians**. Switch to **Recognition** to see the separate step and pedestrian actions.
3. Switch to **Alert** to see which detection the app prioritises.

The four photos are selected examples from the 769-image test set. See the four-class demo benchmark above for precision and mAP. [Photo and clip guide](demo/README.md)

### Run the checks

Run these from the repo root with the project environment. Replace `python`
with the project interpreter path shown above if your environment is inactive.

```sh
python -m pip install -r requirements-test.txt
python -m pytest -q --basetemp artifacts/pytest-current
python scripts/verify_detector.py --images golden/images --reference golden/expected/local_yolov8n_four_class.json --frames 0 --output artifacts/golden-four.json
python scripts/verify_detector.py --images data/v4_four_class_clean/test/images --evaluate --frames 0 --output artifacts/four-class-int8.json
python scripts/verify_detector.py --images data/v4_four_class_clean/valid/images --evaluate --frames 0 --output artifacts/four-class-int8-valid.json
python scripts/sweep_class_thresholds.py --split data/v4_four_class_clean/valid --predictions artifacts/four-class-int8-valid.json --output artifacts/class-threshold-sweep.json
python scripts/quantize_general_model.py --skip-quantize --people-split data/v4_four_class_clean/test
python scripts/verify_detector.py --model models/local/yolov8n_v4_four_class.pt --images data/v4_four_class_clean/test/images --evaluate --frames 0 --output artifacts/four-class-test.json
python scripts/verify_detector.py --model deploy/vercel/model/yolov8n_v4_four_class.onnx --images data/v4_four_class_clean/test/images --evaluate --frames 0 --output artifacts/four-class-onnx.json
python scripts/verify_detector.py --runtime rfdetr --images data/v4_four_class_clean/test/images --evaluate --frames 0 --output artifacts/four-class-rfdetr.json
```

For a Mac environment managed by `uv` without pip, run
`uv pip install --python .venv/bin/python -r requirements-test.txt` for the
first step. The final command requires the labelled dataset handoff.

The four-class YOLO checkpoint is an inference-only projection of the original
trained model. The cleaned dataset has 7,749 training, 859 validation and
769 test images. See the [migration report](docs/FOUR_CLASS_MIGRATION_2026-11-17.md)
for model provenance, the [development checks](docs/FOUR_CLASS_DEVELOPMENT_2026-11-18.md)
for browser and runtime evidence, and the [external validation report](docs/FOUR_CLASS_VALIDATION_2026-11-17.md)
for known detection misses. The supplementary COCO model covers some common
objects. Alert bands use uncalibrated proximity cues, and a clear result can
still miss a hazard.
