# VisionAid

A hazard-detection AI vision assistant for AAI3001 (SIT) — a software-only computer vision / deep learning project, framed conceptually as a wearable device attachment for visually impaired users. There is no physical hardware build; see [CLAUDE.md](CLAUDE.md) for full project context, scope decisions, and rubric alignment.

## Setup

```
pip install -r requirements.txt
```

`easyocr`'s dependencies pull in `opencv-python-headless`, which overwrites the GUI-enabled `opencv-python` build and breaks the live preview windows. If that happens, fix it with:

```
pip uninstall -y opencv-python-headless opencv-python
pip install --force-reinstall --no-deps opencv-python
```

## Dataset & training

The detector is fine-tuned YOLOv8n, trained on a curated Open Images V7 subset for 16 hazard classes (person, vehicles, traffic furniture, bags, pets, etc.) — see [CLAUDE.md](CLAUDE.md) for full details and metrics. The trained checkpoint (`models/hazard_yolov8n.pt`) is the default model for every demo below.

To rebuild the dataset or retrain from scratch:
```
python scripts/build_hazard_dataset.py --per-class 130
python scripts/train_hazard_model.py --epochs 40
```

## Running the demos

All demos use a laptop webcam and the fine-tuned hazard checkpoint by default.

**Web UI (recommended)** — the real demo interface: detection + depth + hazard ranking + spoken alerts, plus buttons for on-demand OCR and item identification, all in a browser:
```
python src/webapp.py
```
Then open http://127.0.0.1:5000. Click "Scan text" or say **"vision scan"** to read nearby text; click "Identify item" for a generic object classification.

**Detection only** — simplest sanity check, no browser needed:
```
python src/webcam_hazard_demo.py
```

**Full pipeline, OpenCV window** — same pipeline as the web UI but in a local debug window instead of a browser (useful for quick iteration):
```
python src/live_pipeline_demo.py
```
Press `q` to quit; `o` or saying "vision scan" triggers OCR.

Useful flags on both `webapp.py` and `live_pipeline_demo.py`: `--no-depth`, `--no-tts`, `--no-ocr`, `--no-voice`, `--no-product-id` (webapp only) to disable individual stages; `--conf`, `--imgsz`, `--source`, `--model` to tune detection.

## Known limitations (prototype stage)

- CPU-only inference: expect roughly 10-15 FPS for the full pipeline on a typical laptop.
- OCR runs synchronously and briefly freezes the video feed while reading; scoped for short signs/labels, not dense documents.
- The voice trigger needs internet access (cloud speech recognition, not an on-device wake-word engine) and a working microphone.
- Depth is monocular and relative (0-1 proximity within a frame), not calibrated metric distance.
- The hazard detector is a real fine-tune but a first baseline (40 CPU epochs, ~2000 curated third-person images, mAP50 0.495) — not yet trained on first-person/glasses-perspective photos.
- Item identification is a generic ImageNet classifier (MobileNetV3), not a currency/product-trained one.
- The Flask app runs locally only — cloud deployment isn't set up yet.
