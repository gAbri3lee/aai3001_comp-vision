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
