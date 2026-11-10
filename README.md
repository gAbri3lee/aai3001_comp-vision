# VisionAid

AAI3001 software prototype for hazard detection. The trained Roboflow v4 models
recognise five classes: **Escalator, People, Pillar, Stair and Wet Floor Sign**, in
that class-ID order. The supplied weights are already trained; running the app
or its verification tools does not retrain them.

## Local setup and launch

Run the commands below in Terminal from the repository folder: the folder
containing this README, `src/` and `visionaid_config.json`. Press **Ctrl+C** to stop
the application from running.

### 1. Prepare the Python environment

If this checkout already has a working `.venv` with the dependencies installed,
skip the first-time setup block. For a fresh checkout, install Python 3.12 first.
The following commands create a project environment, activate it, and install the
local app's Python packages:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Expect package-installation output and a return to the Terminal prompt when it
finishes. This does not start the camera, open the GUI, or train a model. If
`python3.12` is not found, install or select Python 3.12 before continuing.

Each time you open a new Terminal, activate the existing environment:

```sh
source .venv/bin/activate
```

This creates a virtual environment `(.venv)`. Subsequent `python` commands then use the
project's environment. An environment created by `uv` may not include pip; if a
package-install command reports `No module named pip`, use
`uv pip install --python .venv/bin/python -r requirements.txt` instead, replacing
`<requirements-file>` with the filename in that command.

The model files must be present at these locations:

- `models/local/yolov8n_v4.pt`: default local YOLOv8n, confidence 0.35, input 640.
- `models/cloud/rfdetr-nano_v4.pth`: RF-DETR service weights, confidence 0.40, input 384.

They have been copied from the supplied handover bundle in this checkout. In a
fresh clone, copy them from that bundle into the same locations; Git deliberately
ignores model binaries. They use different loaders and are not interchangeable.
Configuration/model paths resolve from the project; run the examples from the
repository root so the relative script paths also work.

### 2. Open the browser web app (GUI)

For a first interactive test, start the web app with depth disabled. Camera
object detection and spoken output remain enabled. This avoids the slow,
experimental depth stage on a laptop CPU:

```sh
python src/webapp.py --port 5001 --no-depth
```

Wait for Terminal to report that the server is running, then open
**http://127.0.0.1:5001** in your browser. The browser does not open automatically.
Allow camera access if macOS asks. The general-object model downloads once to
`models/general/yolov8n.pt` on first use and needs internet for that download.
Subsequent launches reuse it. It runs locally in both local and cloud modes.

You should see a live camera feed, FPS, the active detector and processing time,
the alert gate, depth and speech state, an alert banner, a list of current hazards
and a collapsed list of ignored detections. Open **http://127.0.0.1:5001/?debug=1**
to also see ignored boxes, box sizes and (after calibration) distance estimates. The camera is
connected to the Python app on this computer. Moving your laptop camera or holding something in
front of it changes what the app sees.

**Automatic hazard detection needs no click.** When sufficiently confident,
it draws boxes with a class name and confidence for the five trained classes,
coloured by distance band. Detections that pass the alert gate (below) can produce
a visual/spoken alert such as `Warning, Stair ahead` or `Caution, People on your left`.
The hazard being spoken is drawn in magenta and marked "spoken" in the list.
`No hazards` means nothing passed the gate; it does not establish that the scene
is safe. Stale frames or inference errors clear expired results.

### 3. Automatic object and location alerts

Keep the live feed running; no click is required. Recognised trained detections
retain their names: **Escalator, People, Pillar, Stair and Wet Floor Sign**.
A second, general COCO model adds boxes for 80 common object types, suppressing
boxes that overlap a trained detection by IoU >=0.5 to avoid duplicate notices.
A COCO person the trained model missed is treated as **People** (marked "general
model"); everything else is **Object**. It is not an open-world detector. An empty
frame does not generate an invented object alert. In the app's `/status` response,
general detections use `class_id: -1` and `source: "general"`; the cloud `/detect`
contract and trained IDs 0 to 4 remain unchanged.

A generic Object is spoken as **Object detected ahead** only when it is in the
centre third of the image, at confidence 0.5 or more and large enough (see the
alert gate). Objects at the sides or far away are listed as ignored, not spoken.

On macOS, spoken alerts use the built-in `say` command and wait for each full
phrase to finish. Other platforms use `pyttsx3`. Speech runs in a background
worker, so camera processing continues during playback. At most one phrase waits
while another plays: a newer phrase replaces it, and a waiting phrase more than
1 second old is dropped because the hazard may have gone.

The general detector adds processing time and can miss objects or produce false
positives. A missed trained hazard may receive only a generic Object box if the
general detector finds it. Neither model detecting anything establishes safety.

### Alert gating: what is spoken and why

Both models see far more than a walking user needs to hear: the trained model finds
about 80 percent of people as far as 6 m away, which used to make every background
person an alert. The **alert gate** decides which detections are spoken. The same
rules apply to local and cloud mode. Settings are in `visionaid_config.json` under
`alert_gate`.

| Rule | Effect |
|---|---|
| Size = box height / frame height | Stands in for distance. No metres are spoken. |
| Below the class `min_size` (People 0.40, Stair 0.20, Escalator 0.25, Wet Floor Sign 0.18, Pillar 0.35, Object 0.35) | "far": listed as ignored, never spoken |
| Left or right third | Spoken only if clearly near (1.5 times `min_size`). Objects at the sides are never spoken. |
| General-model detection below confidence 0.5 | Ignored as low confidence |
| Size against the class `near_size` | **Warning** (near), **Caution** (mid-close) or **Note** (mid) |
| Ranking | Any trained hazard comes before any Object, then by severity and nearness |
| Cooldown | Per class, position and band: 4 s (8 s for general detections). A hazard that gets clearly closer re-alerts early. A far person never mutes a near one. |

The banner shows the top gated hazard. The speaker says the highest gated hazard
whose cooldown has expired; that hazard is highlighted on the video and marked
"spoken". The page's **Ignored detections** list shows everything else with the
reason (far, at the side, low confidence).

The sizes are **uncalibrated placeholders** (Wet Floor Sign was raised from 0.15
to 0.18 from the test-set data, decision U1): the page shows "uncalibrated sizes"
until they are confirmed with the tape-measure test. Test-set data on how recall changes with size, and a
proposal, are in `docs/GUI_TEST_REPORT.md` (Phase D). Fill in
`docs/DISTANCE_TEST_SHEET.md` with a tape measure to check them on your camera.
Add `--no-gate` to any app command to compare with the old behaviour, which
announced every detection.

**Distance estimate (debug only).** Optionally measure your camera's focal length
with a person or Wet Floor Sign at a tape-measured distance:

```sh
python scripts/calibrate_distance.py --camera 0 --class People --distance 3
```

This saves the git-ignored `camera_calibration.json`. The debug view then shows an
approximate distance for People and Wet Floor Sign, labelled "estimate". It is
never spoken, and boxes cut off by the frame edge get no estimate.

### Page controls and keyboard shortcuts

| Control | What it does |
|---|---|
| **Local / Cloud** switch and mode badge | Chooses the trained detector. The badge always shows the active one, for example `LOCAL, YOLOv8n` or `CLOUD, RF-DETR Nano`. Cloud is greyed out with the reason when the app was started without `--endpoint`. Switching clears the alert cooldown, so the first hazard afterwards is spoken. |
| **Retry cloud** | Appears after a cloud failure. It checks the service's `/health` (no frame is sent) and switches back only if that succeeds. |
| **Pause** | Stops detection and speech. The camera stays open and the last frame stays on screen with a "Paused" label. |
| **Mute** | Stops the phrase being spoken at once and silences new ones. Detection, the banner and the lists keep working. |
| **Speech rate** | 120 to 260 words per minute (default 180). Applies to the next phrase. There is no volume control because macOS `say` has none. |
| **Test voice**, **Replay last alert** | Speak a fixed test phrase, or repeat the last alert, through the same speaker. Both respect Mute. |
| **Spoken log** | The last 10 phrases sent to the speaker with time and backend, newest first. Kept in memory only; **Clear** empties it. |
| **Priority legend** | The class order from the config severities, highest first, then Object. The class of the current top hazard is outlined and underlined. |
| **Analyse an image** | Upload a JPEG or PNG (up to 10 MB) or pick a sample from `golden/images`. It runs through the same detector, alert gate and phrase code as the live view and shows the result in place of the video until **Back to live**. Uploads are not stored. **Speak result** is off by default. |

Keyboard shortcuts: **Space** pauses or resumes, **M** mutes or unmutes, **C**
switches Local or Cloud. They are ignored while a text box, slider, list or
checkbox has focus. The banner is a screen-reader live region, and every level is
written out (Warning, Caution, Note) as well as coloured.

**Cloud privacy notice.** While cloud mode is active the page shows "Cloud mode:
camera frames are sent to the server." In local mode no frame ever leaves the
computer; the cloud client is not called at all.

Problems while running appear as plain messages at the top of the page instead of
a blank screen: camera not found (retried every 2 seconds), no video for 3 seconds
(the banner is cleared and nothing is spoken), cloud unreachable (the app falls
back to local), extra object detection off, and speech unavailable. Missing weights
or a wrong class order still stop the app at startup.

### 4. Try the full pipeline, including depth

Depth is **experimental**. MiDaS_small on a laptop CPU made the full pipeline run
at about 0.19 FPS in testing, too slow for live use. Stop the previous web app with
Ctrl+C, then use this command to enable all stages, including relative depth:

```sh
python src/webapp.py --port 5001
```

Open or refresh **http://127.0.0.1:5001**. The alert gate still decides what is
spoken, but with depth on, relative depth (not box size) sets whether alerts start
with `Note`, `Caution` or `Warning`. The two are never combined. Depth does not
provide metres or distinguish stairs going up/down. Return to `--no-depth` if it
is too slow. MiDaS downloads its weights on first use. Add `--no-tts` if you do
not want spoken output.

The app's default port is 5000 when `--port` is omitted. These examples use 5001
to avoid common macOS AirPlay conflicts. The web app runs locally; this command
does not publish a website on the internet.

### 5. Optional desktop camera windows

This command opens the **detection-only OpenCV window**, rather than a browser:

```sh
python src/webcam_hazard_demo.py
```

Expect the live feed, boxes, class labels, confidence and FPS. Depth and speech
are disabled in this demo. Press **q** in its window to
close it. Use this when you want to check camera/model detection alone.

This command opens the **full-pipeline OpenCV window**:

```sh
python src/live_pipeline_demo.py
```

Expect camera boxes, relative-depth-based hazard ranking and spoken alerts.
Press **q** to close it. This uses the same detection, alert gate and audio
pipeline as the web app.

Camera index 0 is the default. Use `--source 1` to try another camera or
`--source /path/to/video.mp4` for recorded input. `--model`, `--conf`, `--imgsz`
and `--device` override local detector settings in all demos. Shared defaults are
in `visionaid_config.json`.

### 6. Minimal web test and troubleshooting

This command starts camera-based hazard detection in the browser without depth or
speech:

```sh
python src/webapp.py --port 5001 --no-depth --no-tts
```

At **http://127.0.0.1:5001**, expect live detection boxes and gated hazard notices,
without speech. The general-object model still runs and may download its weights
on first use. OCR, voice input and item identification have been removed.

Another package can install `opencv-python-headless` alongside the GUI build. If a
desktop OpenCV window reports that GUI support is unavailable, run these repair
commands inside the activated environment:

```sh
python -m pip uninstall -y opencv-python-headless opencv-python
python -m pip install --no-deps 'opencv-python>=4.7,<4.12'
```

The first command removes conflicting OpenCV packages; the second reinstalls the
GUI-enabled build. Expect installation output, then retry the desktop demo. These
are troubleshooting commands, not a step to repeat on every launch.

## Cloud detector

Cloud mode changes the trained object detector to RF-DETR. The camera, browser,
general-object detector, depth and speech remain local; JPEG camera frames go to
the configured service. If a cloud request fails or times out (3 seconds), the app
falls back to local on its own and stays there until you press **Retry cloud**.
It never switches to cloud by itself, and the two detectors are never fused.
This repository prepares the service but does not provision public hosting.

### Build and start the service

With Docker installed and running, build the service image from the repository
root. This packages RF-DETR and its separate Linux dependencies:

```sh
docker build -f cloud/Dockerfile -t visionaid-rfdetr .
```

Expect dependency downloads and Docker build logs, ending in a successful image
build. This can take several minutes. It does not start the service or the GUI.
The image uses RF-DETR 1.11.2, CPU PyTorch 2.9 and Transformers 5.1; the available
Intel macOS PyTorch 2.2 runtime cannot run that RF-DETR environment directly.

Start the service and make the supplied cloud checkpoint available read-only.
If port 9000 is taken (php-fpm uses it on some Macs), use port 9100 for the cloud
service: `-p 127.0.0.1:9100:9000`, then `http://127.0.0.1:9100` in the commands below.

```sh
docker run --rm -p 127.0.0.1:9000:9000 \
  -v "$PWD/models/cloud:/app/models/cloud:ro" visionaid-rfdetr
```

Expect server/model-loading logs in this Terminal. Leave it running. It serves
inference on **http://127.0.0.1:9000**, but has no browser camera GUI of its own.
The first model load can take a while. Ctrl+C stops this container.

In another Terminal, check whether the model has finished loading:

```sh
curl http://127.0.0.1:9000/health
```

A ready service returns JSON with `status: "ready"`, model `rfdetr-nano`, the five
class names, input size 384 and class thresholds. If it is still loading, wait
and retry; an open port alone does not establish that the model is ready.

### Connect the web app to the service

Activate the local Python environment in another Terminal, stop any previous
web app on port 5001, and start the GUI with the cloud service available:

```sh
python src/webapp.py --port 5001 --endpoint http://127.0.0.1:9000 --no-depth
```

Open **http://127.0.0.1:5001**. The app starts in local mode; press **Cloud** (or
**C**) to switch. Add `--backend cloud` to start in cloud mode instead. Both
detectors are loaded, so the local weights are needed even when starting in cloud
mode (they are the fallback). In cloud mode the badge reads `CLOUD, RF-DETR Nano`,
the status shows the round-trip time, and the privacy notice is shown. If the
service stops, the page says "Cloud unreachable, using local." and keeps working
on local. Both the service Terminal and GUI Terminal must remain running. This example tests the cloud-service architecture
on one computer; a real remote deployment needs its own host and endpoint.

The image defaults to CPU. On a Linux NVIDIA host, build with
`--build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128` and run with
`--gpus all -e VISIONAID_DEVICE=cuda`, with container GPU support installed.
For a public detector endpoint, configure the host's HTTPS/access control.
Do not expose the local Flask camera GUI as a multiuser cloud application: it
opens the camera of the computer it runs on. A publicly hosted version would need
browser camera capture (`getUserMedia`) posting frames to the server instead;
that is not built.

### Detection response for developers

`POST /detect` accepts a JPEG body with `Content-Type: image/jpeg`. The following
is an illustrative response for a 1280×720 frame; the numbers are examples, not
promised confidence or speed:

```json
{
  "detections": [
    {
      "class_id": 3,
      "class": "Stair",
      "confidence": 0.82,
      "box_xyxy": [120, 340, 610, 720]
    }
  ],
  "latency_ms": 22.8,
  "model": "rfdetr-nano"
}
```

This describes one detected stair region. Class ID 3 maps to Stair. The four box
coordinates are left, top, right and bottom in original-frame pixels.
`latency_ms` measures server inference; the local client adds `round_trip_ms`,
including JPEG encoding, upload and download. An empty `detections` list means no
objects passed the selected threshold.

The default request deadline and maximum frame age are three seconds, controlled
by `--timeout` and `--max-frame-age`. Only one request is in flight; capture keeps
the newest frame. `--conf` explicitly overrides cloud thresholds through the API;
otherwise validated defaults apply. Model/input-size overrides are local: set
cloud weights and resolution on the service through its config/environment.

## Verification and export

These commands are developer checks. They do not need the live camera and do not
train the models. Use the activated Python environment and repository root.

### Run the automated tests

If the test/export tools are not already installed, install their dependencies:

```sh
python -m pip install -r requirements-test.txt
```

Expect package-installation output. This adds pytest, COCO evaluation and ONNX
tools; it does not run any checks yet.

Run the automated regression tests:

```sh
python -m pytest -q
```

Expect a pass/fail summary in Terminal. These cover configuration, detection
records, app/backend behaviour, the page controls and messages, evaluation and
dataset validation. The keyboard-shortcut test needs Node.js and is skipped
without it. Hardware camera, audible-speech and projector checks still need a
manual test: see `docs/MANUAL_TEST_SHEET.md`.

Two page checks are separate scripts. `python scripts/check_contrast.py` prints
the contrast ratio of every colour pair on the page. With the app running,
`node scripts/check_layout.mjs http://127.0.0.1:5001 artifacts/screenshots`
loads the page in headless Google Chrome at 1920x1080, 1366x768 (also with
`?debug=1`) and 390 px wide, checks for page and sideways scrolling, and saves
screenshots.

### Check the 20 reference images

Run the local detector on the supplied golden images and compare the outputs
against the original reference detections. This also measures 50 warmed frames:

```sh
python scripts/verify_detector.py --images golden/images \
  --reference golden/expected/local_yolov8n.json --frames 50 \
  --output artifacts/local_golden.json
```

Expect a JSON summary containing `golden.passed`, box-match counts and detector
latency/FPS. Predictions are saved to `artifacts/local_golden.json`; the summary
is saved to `artifacts/local_golden.report.json`. The 20 images are copies from
the handover bundle, used for verification rather than new training data.

To compare an existing predictions file without running inference again:

```sh
python golden/check_golden.py golden/expected/local_yolov8n.json artifacts/local_golden.json
```

Expect `passed: true` when the image names and detections meet the parity rules.
A failure returns a nonzero exit code. The exact image set must match, at least
95% of reference boxes must match class and IoU >=0.9, and extra boxes must be no
more than max(2, 5% of reference count). Pillar mapping also has a dedicated test
because the local golden references contain no Pillar detections.

With the RF-DETR service running, perform the equivalent HTTP check using its
reference outputs. The longer deadline below is for this offline CPU check:

```sh
python scripts/verify_detector.py --runtime cloud \
  --endpoint http://127.0.0.1:9000 --timeout 60 --max-frame-age 60 \
  --images golden/images --reference golden/expected/cloud_rfdetr-nano.json \
  --frames 50 --output artifacts/cloud_golden.json
```

Expect an RF-DETR parity summary and measured round-trip timings. Predictions and
its `.report.json` summary are saved under `artifacts/`. This sends the test
images to the endpoint; it does not start the camera or GUI.

### Evaluate all 920 test images

Replace `/path/to/visionaid_pt` with the extracted handover folder. The first line
sets that location for this Terminal; the second command evaluates local YOLOv8n
on all test images, including 92 backgrounds without label files:

```sh
BUNDLE=/path/to/visionaid_pt
python scripts/verify_detector.py --images "$BUNDLE/test_set/test/images" \
  --evaluate --frames 0 --output artifacts/local_test.json
```

Expect progress messages followed by mAP50, mAP50-95, precision, recall and safety
recall. Results are written to `artifacts/local_test.json` and
`artifacts/local_test.report.json`. `--frames 0` disables the extra timing
benchmark; it does not skip evaluation images. This uses the notebook-style COCO
AP evaluator at confidence >=0.001 and the existing validation-selected alert
threshold. Test images never select a new threshold. Keep these metrics labelled
separately from Ultralytics `.val()` results.

To evaluate the RF-DETR checkpoint directly in Linux, use the already-built
service image. Keep `BUNDLE` set above. The command creates an output folder,
starts a temporary container, installs its evaluation dependency, and runs the
same 920-image check. Checkpoint, script and dataset mounts are read-only:

```sh
mkdir -p artifacts
docker run --rm -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 \
  -v "$PWD/models/cloud:/app/models/cloud:ro" \
  -v "$PWD/scripts:/app/scripts:ro" \
  -v "$PWD/artifacts:/artifacts" \
  -v "$BUNDLE/test_set:/data:ro" \
  --entrypoint sh visionaid-rfdetr -c \
  'pip install --no-cache-dir pycocotools && python /app/scripts/verify_detector.py --runtime rfdetr --images /data/test/images --evaluate --frames 0 --output /artifacts/cloud_test.json'
```

Expect container logs, image-count progress and RF-DETR accuracy metrics. The
outputs are saved on your computer as `artifacts/cloud_test.json` and
`artifacts/cloud_test.report.json`; the container exits afterwards. CPU evaluation
can be slow. This is a direct checkpoint check, so it excludes HTTP/network time.

### Export and check ONNX

First establish the local `.pt` baseline using the full-set command above. Then
export those existing weights to FP32 ONNX, without retraining:

```sh
python scripts/export_detector.py
```

Expect export logs and a new `models/local/yolov8n_v4.onnx` file, using batch one
and input size 640. The app still defaults to `.pt`; exporting does not switch it.

Evaluate the export against the same test set and measure 50 warmed frames:

```sh
python scripts/verify_detector.py --model models/local/yolov8n_v4.onnx \
  --images "$BUNDLE/test_set/test/images" --evaluate --frames 50 \
  --output artifacts/onnx_test.json
```

Expect ONNX accuracy and timing results in Terminal and the output/report JSON
files under `artifacts/`. Compare them with the `.pt` baseline. Keep `.pt` as the
default until the export passes golden parity, loses no more than 0.01 mAP50 with
the same evaluator, and improves measured speed on the deployment device.
INT8 is deferred: the handover YAML points train/val/test at test images and must
not be used for calibration. Obtain separate representative calibration data first.

### Measure the continuous processing loop

This runs recorded images through detection, optional depth, hazard ranking,
annotation and JPEG encoding to measure their combined processing cost:

```sh
python scripts/benchmark_pipeline.py --images golden/images --frames 50 \
  --output artifacts/pipeline_cpu.json
```

Expect a JSON report with median/p95 processing time and FPS, saved to
`artifacts/pipeline_cpu.json`. It uses 10 warm-up frames plus 50 measured frames;
add `--no-depth` to compare without depth. It excludes physical camera capture,
audible speech, the app's additional general-object detector and browser rendering.
Report these timings separately from detector-only latency. T4 notebook timings do not
promise laptop performance. See `docs/VERIFICATION.md` for the measured results.

### Alert analysis tools

These use the 920-image test split (`BUNDLE` as above) and, for cloud, the running
RF-DETR service. They write under `artifacts/`; results are summarised in
`docs/GUI_TEST_REPORT.md`.

```sh
python scripts/recall_by_size.py --images "$BUNDLE/test_set/test/images"
python scripts/compare_pillar.py --images "$BUNDLE/test_set/test/images" \
  --endpoint http://127.0.0.1:9000
python scripts/replay_alerts.py --source clip.mp4 --output artifacts/replay_local.jsonl
python scripts/replay_alerts.py --source clip.mp4 --backend cloud \
  --endpoint http://127.0.0.1:9000 --output artifacts/replay_cloud.jsonl
python scripts/replay_alerts.py --compare artifacts/replay_local.jsonl artifacts/replay_cloud.jsonl
```

`recall_by_size.py` reports recall per class by box size (the operating range) and
proposes gate sizes without changing the config. `compare_pillar.py` compares
Pillar recall, precision and false alarms between local and cloud and saves
example sheets; without `--endpoint` it runs only the local half.
`replay_alerts.py` runs a video or image folder through the app's detector and
alert gate frame by frame, without audio, and compares two runs.

## Training Data

The five training notebooks contain the latest Roboflow training work, with their original outputs preserved. 
The saved results show an mAP50 of 0.659 and safety recall of 0.716 for YOLOv8n, 
compared with 0.739 and 0.767 for RF-DETR. 
The dataset contains 8,247 training images, 925 validation images and 920 test images.
Its scene-split fingerprint is `49601080aac1f838190341d331785578`.

**Roboflow API key.** Only dataset download in the notebooks needs a key; the app,
the cloud service and the verification tools do not. Each person generates their
own private key: sign in to Roboflow, select the workspace, open the workspace
settings and the API keys page (check Roboflow's current documentation if the menu
has moved), and generate a new key with a descriptive name. Supply it in one of
two ways, and never write it into a notebook, script or commit:

- Locally: `export ROBOFLOW_API_KEY=...` in the Terminal before starting Jupyter.
  If you keep it in a `.env` file, that file is already in `.gitignore`.
- On Kaggle: add a secret named `ROBOFLOW_API_KEY` (Add-ons, Secrets).

The notebooks read the environment variable first, then the Kaggle secret. An
older key was committed in earlier notebook versions and remains in git history.
Removing it from the files does not disable it: a workspace owner must revoke that
key in Roboflow.

The notebooks contain all **model training for the 5 finalised classes and document the performance of both YOLO and RF-DETR models.**

The scripts/ folder contains tools for verifying detector outputs, exporting models and measuring performance in local and cloud deployments. Dataset-validation code and automated tests check that the splits prepared by the notebooks are correctly structured and separated. These supporting tools do not train models.

For future training, use the scene-based split prepared in the notebooks. Do not use the handover's test-only YAML file for training: its training, validation and test entries all point to the test images. Automated tests are in `tests/`.

### Pillar detection: Limitations (Weakest Class)

**Pillar is supported, but it is the weakest class and detection is unreliable.**
Pointing the camera at a pillar runs hazard detection automatically. If the
detector recognises it with sufficient confidence, a **Pillar** bounding box
appears and can contribute to the prioritised visual/audio hazard alert.
A generic **Object** detection does not establish that a pillar was recognised.

The saved test results contain 153 labelled pillar objects across the 920 test
images:

| Detector | Validated confidence threshold | Pillar recall |
|---|---:|---:|
| Local YOLOv8n | 0.35 | 17.0% |
| RF-DETR Nano | 0.40 | 29.4% |

Recall measures how many labelled pillar objects were found on that test set.

YOLOv8n detected about 17% of the labelled pillars, while RF-DETR Nano detected about 29%. 
These figures describe performance on this test set, they are not the **probability of detecting a particular pillar**.

Both models missed many pillars. Keeping a pillar in view for longer does not guarantee detection, 
because the model may continue to miss it from the same angle.

Re-measured on this branch (`scripts/compare_pillar.py`, all 105 test images with
a pillar and 815 without): local 17.0 percent recall at 54 percent precision,
cloud 28.8 percent recall at 52 percent precision over HTTP. Both are best at
medium sizes (box height 0.30 to 0.60 of the image, about 42 to 63 percent) and
miss most small, distant pillars and close-up bases cut off by the frame. Some
misses are thin poles and bollards labelled Pillar where the model boxed a
neighbouring pole.

**When a pillar is not detected, the app says nothing about it.** The general
COCO model has no pillar class; it covered only 3 of the 127 local misses with an
Object box, which is spoken as "Object detected ahead" only if it also passes the
alert gate.

For a practical check, use good lighting, keep the pillar clearly visible, and
try different camera angles and distances. **No Pillar box means the model has
not detected it; it does not mean there is no obstacle or that the scene is safe.**

### Other limitations

Stair direction is unavailable and not part of the model training. Wet Floor Sign
means a sign was detected, not that floor wetness was measured. Distance bands come
from box size and are estimates, not measured metres; the optional debug distance
is an estimate too and is never spoken. MiDaS proximity is relative, not metres.
An empty hazard list does not prove a scene is safe.
The additional general detector recognises common COCO objects, not arbitrary
unknown objects, brands or currency denominations. Generic notices do not prove
that every obstacle was found. 
