# VisionAid

AAI3001 hazard-detection prototype. The live GUI reports **Escalator, People,
Stairs and Wet Floor Sign** using local YOLOv8n or a separate RF-DETR Nano
service. The hosted Vercel photo demo runs the original five-class YOLOv8n
ONNX model.

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

Copy the weights from the project handoff to the paths below. Git ignores these
files.

| File | Used by |
| --- | --- |
| `models/local/yolov8n_v4_four_class.pt` | Local YOLOv8n and the GUI's fallback from Cloud |
| `models/cloud/rfdetr-nano_v4.pth` | RF-DETR Docker service, needed when you run the service yourself |

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

The GUI may download a supplementary COCO YOLOv8n checkpoint to
`models/general/yolov8n.pt` on first use. If that download fails, the trained
hazard detector can still run.

### 3. Start the browser GUI

**macOS**

```bash
.venv/bin/python src/webapp.py --port 5001
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
The RF-DETR adapter maps the supplied checkpoint's five classes to the active
four. The GUI connects to the service's `/health` and `/detect` routes.

## Vercel Upload Image

1. Open [VisionAid Photo Check on Vercel](https://aai3001-gwup54owc-computer-vision2.vercel.app/).
   If prompted, sign in to Vercel with an account that has access to the deployment.
2. Click **Choose a photo** and select a JPEG from [the bundled golden images](golden/images)
   or from the test folder.
3. Look for the annotated photo, class labels and confidence, the highest-priority
   alert or **No hazards detected**, and the model and request timings. **Read the
   result aloud** uses your browser's audio feature.

The browser resizes each photo and sends it as a JPEG to `/api/detect`. A Python
function runs the original five-class YOLOv8n ONNX model. For a readiness check,
open [the Vercel health endpoint](https://aai3001-gwup54owc-computer-vision2.vercel.app/api/health)
while signed in and look for `ready` and `yolov8n-onnx`. This demo accepts
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
2. Under **Videos**, click **Play** beside the clip you want to test. Choose the VivoCity atrium clip or one of the four field sequences.
3. Use **Alert** to check the banner and spoken alert, or **Recognition** to
   check detection boxes and final classification of the hazard type.
   When the clip ends, click on **Replay** or **Back to camera** for live view.

The library reads videos from `demo/videos/`. Bundled clips are included in Git. [Demo library guide](demo/README.md)

### Make a short video from a bundled photo

If you have no clip handy, start the project Python interpreter from the repo
root (`.venv/bin/python` on Mac or `.\.venv\Scripts\python.exe` in PowerShell)
and paste the following lines. This makes a six-second still-frame MP4 for checking
playback and controls.

```python
from pathlib import Path
import cv2
source = next(Path("golden/images").glob("761978429*"))
frame = cv2.imread(str(source))
height, width = frame.shape[:2]
codec = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter("demo/videos/golden-escalator.mp4", codec, 10, (width, height))
assert writer.isOpened(), "OpenCV could not create the MP4"
for _ in range(60):
    writer.write(frame)

writer.release()
exit()
```

Click **Reload** in the Demo library, then **Play golden-escalator.mp4**.

## Verification and Model Testing Results

### Detection benchmarks (by Model and Class)

| Model | Test split | mAP50 | mAP50–95 |
| --- | --- | ---: | ---: |
| Local YOLOv8n | Cleaned four-class split, 769 images | 77.4% | 50.9% |
| Original YOLOv8n ONNX | Original five-class split, 920 images | 65.8% | 41.9% |
| Original RF-DETR Nano | Original five-class split, 920 images | 73.9% | 44.7% |

The ONNX and RF-DETR rows are archived CPU benchmarks. The local YOLOv8n
row uses a different split and class set. [Four-class results](docs/evidence/four-class-migration-2026-11-17.json),
[five-class results](docs/verification_results.json)

| Class AP50 | Local YOLOv8n | Original YOLOv8n ONNX | Original RF-DETR Nano |
| --- | ---: | ---: | ---: |
| Escalator | 94.4% | 96.3% | 93.1% |
| People | 69.4% | 69.3% | 80.7% |
| Stairs | 76.3% | 77.5% | 79.7% |
| Wet Floor Sign | 69.6% | 62.4% | 81.1% |

AP50 measures average precision for one class at 0.50 intersection over union
(IoU). mAP50 averages that score across classes. mAP50–95 averages across classes
and IoU thresholds from 0.50 to 0.95. The saved per-class breakdown reports mAP50.

### Run the checks

Run these from the repo root with the project environment. Replace `python`
with the project interpreter path shown above if your environment is inactive.

```sh
python -m pip install -r requirements-test.txt
python -m pytest -q --basetemp artifacts/pytest-current
python scripts/verify_detector.py --images golden/images --reference golden/expected/local_yolov8n_four_class.json --frames 0 --output artifacts/golden-four.json
python scripts/verify_detector.py --images data/v4_four_class_clean/test/images --evaluate --frames 0 --output artifacts/four-class-test.json
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
