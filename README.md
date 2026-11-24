# VisionAid

AAI3001 hazard-detection prototype. The live GUI reports **Escalator, People,
Stairs and Wet Floor Sign** using local YOLOv8n or a separate RF-DETR Nano
service. The Vercel photo demo in this repository uses the four-class YOLOv8n ONNX export.

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
| `models/cloud/rfdetr-nano_v4_four_class.pth` | RF-DETR Docker service, needed when you run the service yourself |

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
2. Under **Videos**, click **Play** beside the clip you want to test. Choose a field sequence or one of the two pedestrian videos.
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

All three runs used the same 769 cleaned test images, ground-truth boxes and COCO evaluator. Precision and recall count box matches at IoU 0.50 using each model’s configured confidence threshold. The two YOLO rows run the same learned detector in different formats.

| Runtime | Confidence | Precision | Recall | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|---:|
| YOLOv8n PyTorch | 0.35 | 79.6% | 63.7% | 77.4% | 50.9% |
| YOLOv8n ONNX | 0.35 | 79.6% | 63.7% | 77.4% | 50.9% |
| RF-DETR Nano | 0.40 | 85.4% | 71.7% | 84.5% | 52.3% |

#### What to say during the demo

“We tested Escalator, People, Stair and Wet Floor Sign. We chose them because each changes a pedestrian’s next decision. An escalator has moving steps whose direction matters. A person may cross your path. Stairs change the floor level. A wet-floor sign signals a possible slip hazard even when the surface looks dry.”

We excluded Pillar from the final scope because the original local model detected only 17.0% of labelled pillars at its alert threshold on the earlier five-class test. A reliable pillar warning would need stronger evidence.

In **Upload → Demo library**, analyse one named photo in each class folder. Read out the label and the action beside it. Then play **Stairs and pedestrians**, switch to **Recognition** and point out the separate step and pedestrian actions. Switch to **Alert** to see which detection the app prioritises. Use the 769-image table above for precision and mAP. The four photos are selected examples from that test set. [Photo and clip guide](demo/README.md)

| Tested class | Ground-truth boxes | Local YOLO precision | Local YOLO recall | Local YOLO AP50 | RF-DETR AP50 |
|---|---:|---:|---:|---:|---:|
| Escalator | 103 | 92.6% | 84.5% | 94.4% | 91.8% |
| People | 559 | 75.3% | 58.0% | 69.4% | 80.9% |
| Stair | 268 | 80.4% | 67.2% | 76.3% | 79.4% |
| Wet Floor Sign | 52 | 92.1% | 67.3% | 69.6% | 85.7% |

“Local YOLO reached 77.4% mAP at IoU 0.50. Averaging over ten increasingly strict overlap thresholds from 0.50 to 0.95 brought it to 50.9%. Precise box placement is harder for this detector. At the GUI threshold, 35 of its 38 wet-floor-sign detections matched a labelled sign, but it missed 17 of the 52 signs. Escalators were its strongest class by AP50. RF-DETR scored highest overall on this split at 52.3% mAP50–95. Both models fall below our professor’s 60–70% reference range. These results support a supervised classroom demonstration. Establishing reliability for independent travel would require new venues, missed-hazard and false-alert rates, and evidence that warnings arrive in time.”

The old README quoted historical tests on 920 images and five classes. Removing Pillar from the dataset left those measurements and the original classifier weights intact. The active YOLO PyTorch and ONNX heads now emit four classes. For RF-DETR we retained the four foreground rows and the background slot across its encoder and decoder classifiers. All three use the same four class IDs. This conversion involved no retraining. Older hosted services can still return five classes through the explicitly validated compatibility mapping. Check the hosted service’s version before quoting local results for it. The Vercel function code was also evaluated locally on all 769 images and matched the displayed ONNX metrics to one decimal place.

[Full metrics, model hashes and test-set fingerprint](docs/evidence/four-class-demo-benchmark.json) · [Historical five-class evidence](docs/verification_results.json)

### Run the checks

Run these from the repo root with the project environment. Replace `python`
with the project interpreter path shown above if your environment is inactive.

```sh
python -m pip install -r requirements-test.txt
python -m pytest -q --basetemp artifacts/pytest-current
python scripts/verify_detector.py --images golden/images --reference golden/expected/local_yolov8n_four_class.json --frames 0 --output artifacts/golden-four.json
python scripts/verify_detector.py --images data/v4_four_class_clean/test/images --evaluate --frames 0 --output artifacts/four-class-test.json
python scripts/verify_detector.py --model models/local/yolov8n_v4_four_class.onnx --images data/v4_four_class_clean/test/images --evaluate --frames 0 --output artifacts/four-class-onnx.json
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
