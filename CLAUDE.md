# VisionAid — Project Context for Claude Code

**Module:** AAI3001, Singapore Institute of Technology (SIT)
**Project:** VisionAid — a hazard-detection AI vision assistant, framed as a wearable "glasses attachment" concept for visually impaired users, aimed at supplementary environmental awareness.

**This is a software-only CV/DL project — there is no physical hardware build.** The "clips onto glasses" / sensor / edge-processor framing below is a conceptual product narrative used to justify design choices (e.g. why depth matters, why output is audio), not a fabrication target. Everything gradeable is software: dataset curation/annotation, model training, local+cloud deployment, and a UI. Do not propose hardware acquisition, physical prototyping, or treat any component list as a build/shopping list.

This file summarizes everything decided so far so a fresh Claude Code session can pick up implementation work without re-deriving the scope or architecture.

---

## 1. Grading rubric (what actually matters)

The module rubric grades five things, all software:

| Row | Excellent | Satisfactory | Average |
|---|---|---|---|
| **Dataset** | Acquired own dataset or curated existing dataset; annotated own or existing dataset | Existing dataset; annotate existing dataset | Copying existing dataset and annotations |
| **Model Development** | Architecture not previously used for this task, or a novel training strategy | Architecture/strategy already proven for similar tasks | Fully copying from existing work |
| **Model Deployment** | Both cloud and local deployment, with an optimized model (e.g. TorchServe) | Either local or cloud deployment | No deployment (algorithm only) |
| **User Interface** | Intuitive, well-designed, outstanding UX | Relatively user-friendly, some rough edges | Not intuitive, significant usability issues |
| **Scalability** | Clear avenues for future enhancement/expansion | Somewhat scalable, some limitations | Minimal, needs major rework to extend |

Implications to keep in mind:
- **Dataset row is graded on the real artifact at submission time.** A placeholder/pretrained model can be used during development, but the final submission needs an actually-acquired-and-annotated dataset, or this row caps at "Average."
- **Plain fine-tuning of a stock architecture (e.g. YOLOv8n) on your own data likely caps at "Satisfactory"** for Model Development — reaching "Excellent" needs something genuinely novel on top (e.g. a fused detection+depth head, a hazard-proximity-weighted loss, an architecture not previously applied to hazard/obstacle detection), not just a newer YOLO version number.
- **Deployment means software deployment** — a local inference service/script plus a cloud-hosted demo (e.g. a web app on Vercel-equivalent + an optimized serving backend like TorchServe) — not deployment onto physical edge hardware.

---

## 2. Scope — what's in, what's out

The project started as a broad "smart glasses that do many things" idea and was deliberately narrowed. Don't casually re-expand scope without noting it here.

| Area | Decision |
|---|---|
| Core framing | Safety-focused hazard-detection vision assistant, **not** a general-purpose smart-glasses platform |
| Main deep-learning task | **Hazard detection** — this is the project's actual research/DL contribution |
| Depth sensing | Keep, but **software-only**: monocular depth estimation (e.g. MiDaS / Depth Anything) run on the video feed — there is no physical ToF sensor to fuse with |
| OCR (text reading) | Keep, but **secondary** — on-demand only, not part of the continuous safety loop |
| Face recognition | **Removed from core scope entirely** |
| Currency / product ID | **Optional** — on-demand only |
| Navigation | **Avoided** — full wayfinding is out of scope; only local obstacle guidance |
| Voice commands | Keep, kept simple |
| Text-to-speech / audio feedback | Definitely keep — this is the primary output channel (simulated via normal speakers/browser audio) |
| Model strategy | **One** strong, fine-tuned CV pipeline (own dataset, own annotations) rather than many independent bolted-on models |

Key constraints to keep visible in any written report: real-time latency, low-light / crowded-scene accuracy, information overload, privacy, and the explicit framing that this is **supplementary, not guaranteed, safety**.

---

## 3. Software pipeline

Five-stage pipeline, conceptually running in a loop (1→2→3→4→5→back to 1, every frame). All stages are software; hardware terms below describe what a stage *stands in for* in the product narrative, not something being built.

1. **Capture** — video/webcam frame input (stands in for camera+mic).
2. **AI processing** — two tracks:
   - **Continuous / every frame** (the core hazard-safety loop):
     - *Object detection* — YOLOv8n, **fine-tuned on a curated Open Images V7 subset** for the 16 hazard classes below (`models/hazard_yolov8n.pt`, see Section 4) → outputs class + bounding box + confidence. Not first-person/glasses-perspective yet (see caveat in Section 4) — that would need the "own captured photos" path that was deferred.
     - *Depth estimation* — monocular depth model (MiDaS or similar) on the same frame → estimated distance to each detected hazard.
   - **Secondary / on-demand only** (triggered by the spoken phrase **"vision scan"**, or a push-to-talk key in the prototype):
     - *OCR* — lightweight OCR engine → reads short signs/text labels. **Not scoped for dense documents** — webcam resolution/distance/steadiness isn't enough for a full page, and low-confidence reads there tend to come out as gibberish; the implementation filters by confidence and only targets short text.
     - *Currency / product ID* — MobileNetV3-Small (ImageNet-pretrained), top-1 label + confidence. Generic object classifier, **not** trained on currency — a placeholder for this optional branch (see Section 6).
3. **Safety decision & response engine** — rule-based logic, **not** a trained model. Fuses detection + depth, ranks hazards by proximity/type and position in the walking path, applies cooldown/tracking to avoid repeat alerts, and routes on-demand query results (OCR/currency ID) to output.
4. **Text-to-speech** — converts the prioritised message to speech (e.g. Coqui TTS / eSpeak-NG), played through normal audio output.
5. **Output / user action** — audio delivered to the user; a voice command loops back to stage 1 for on-demand queries.

> Model names above (YOLOv8-nano, MiDaS, Coqui TTS/eSpeak-NG, MobileNet) are plausible starting choices, not benchmarked. YOLOv10-nano/v11-nano were also discussed as more efficient (NMS-free, lower-latency) alternatives, at the cost of less mature export/deployment tooling than YOLOv8n.

**Model swap strategy (validated, now executed):** Built and tested the full pipeline (deployment, UI, decision engine, TTS) against a pretrained/placeholder checkpoint first, keeping the class list and inference interface config-driven rather than hardcoded. A fine-tuned checkpoint now exists (`models/hazard_yolov8n.pt`) and is the default `--model` everywhere — see Section 4 for how it was built and its real metrics.

**Gotcha hit during the swap:** the fine-tuned checkpoint's class names come out Title Case (`"Person"`, `"Car"`, from Open Images) while the placeholder COCO model and all the hazard-class/severity lookups in the code were written lowercase (`"person"`, `"car"`). Every place that reads `names[int(box.cls)]` now lowercases it before comparing against `HAZARD_CLASSES` or `SEVERITY` — without that, the decision engine would have silently fallen back to default severity for everything after the swap. Keep lowercasing at that boundary for any future model swap too, rather than relying on a specific checkpoint's naming convention.

---

## 4. Dataset + training (`scripts/`, `data/`, `models/`)

The actual DL contribution: a curated hazard-detection dataset and a fine-tuned YOLOv8n checkpoint, not just the placeholder pipeline.

| File/dir | Purpose |
|---|---|
| `scripts/build_hazard_dataset.py` | Curates a dataset from Open Images V7 for the 16 hazard classes (person, bicycle, car, motorcycle, bus, truck, traffic light, fire hydrant, stop sign, bench, chair, dog, backpack, suitcase, potted plant/houseplant, couch). Pulls each class with its **own** capped query rather than one combined query, so common classes (person, car) don't crowd out rare ones (fire hydrant, stop sign) in the sample budget — this is the "curation" that distinguishes it from a raw dump. Exports directly to YOLO format. |
| `data/hazard_dataset/` | Output of the above: 2001 curated images (1701 train / 300 val), all 16 classes represented (137-639 boxes each). Gitignored (binary/large) — rerun the build script to regenerate. |
| `scripts/train_hazard_model.py` | Fine-tunes YOLOv8n (COCO-pretrained start) on the curated dataset. CPU-only run used 40 epochs at 320px, ~90 min. |
| `models/hazard_yolov8n.pt` | **The trained checkpoint** — default `--model` for all runtime scripts now. Validation: precision 0.634, recall 0.456, **mAP50 0.495, mAP50-95 0.36**. A real first fine-tune, not a fully converged model — a longer/higher-resolution run would improve it further before treating it as submission-final. |

**Known caveat on this dataset:** it's the "curated existing dataset" rubric path (Open Images V7, third-person stock photos), not "own dataset" with first-person/glasses-perspective photos — that stronger path was explicitly deferred by choice (see the dataset-approach decision) in favor of a faster fully-automated baseline. Revisit if there's time before submission to strengthen the Dataset rubric row further.

To rebuild/retrain: `python scripts/build_hazard_dataset.py --per-class 130` then `python scripts/train_hazard_model.py --epochs 40`.

---

## 6. Prototype code (`src/`)

| File | Purpose |
|---|---|
| `src/webcam_hazard_demo.py` | Detection-only test: webcam → YOLO → annotated window. Simplest way to sanity-check the model/camera. |
| `src/depth.py` | `DepthEstimator` — MiDaS_small monocular depth, returns a per-frame proximity map (0-1, higher = closer, relative not metric). |
| `src/decision_engine.py` | Rule-based hazard ranking (severity × proximity) + `AlertCooldown` so alerts don't repeat every frame. |
| `src/tts.py` | Background-threaded speech output (pyttsx3/SAPI5). |
| `src/ocr.py` | `OcrReader` — EasyOCR wrapper with 1.6x upscale + grayscale+CLAHE preprocessing, confidence filtering (`MIN_CONFIDENCE`), and reading-order sort. Scoped to short text, not documents (see Section 3); stylized brand/logo text (foil packaging etc.) is a known hard case even with tuning. |
| `src/voice_trigger.py` | Background thread listening for the spoken trigger phrase via Google's free Web Speech API (needs internet; not a true low-latency wake-word engine). |
| `src/product_id.py` | `ProductIdentifier` — MobileNetV3-Small pretrained on ImageNet, top-1 label + confidence. Generic object classifier, **not** trained on currency — placeholder for the optional currency/product ID branch. |
| `src/live_pipeline_demo.py` | CLI/OpenCV-window version of the full pipeline: capture → detection+depth → decision engine → TTS, plus OCR on-demand via the `o` key or saying **"vision scan"**. Useful for quick local debugging. |
| `src/webapp.py` + `src/templates/index.html` | **Primary UI.** Flask web app running the same full pipeline, streamed to a browser at `http://127.0.0.1:5000`: live annotated video, a color-coded alert banner, and buttons for "Scan text" (OCR) and "Identify item" (product ID) alongside the voice trigger. Run with `python src/webapp.py` from the repo root. This is also the "local deployment" half of the Model Deployment rubric row — a real running service, not just a script. |

Setup note: `easyocr`'s dependency chain pulls in `opencv-python-headless`, which silently overwrites the GUI-enabled `opencv-python` build and breaks `cv2.imshow`. If reinstalling from a clean environment, install `requirements.txt` first, then run:
```
pip uninstall -y opencv-python-headless opencv-python
pip install --force-reinstall --no-deps opencv-python
```

Known limitations of the current prototype: CPU-only inference runs the full pipeline (detection+depth) at roughly 10-15 FPS on a typical laptop; OCR is synchronous and briefly freezes the video feed while running; the voice trigger requires internet and a working microphone; product ID is a generic ImageNet classifier, not a real currency/product model.

---

## 7. Status & next steps

- Scope is settled: software-only CV/DL project, no physical build.
- **Done:** curated hazard dataset (2001 images, 16 classes) and a real fine-tuned YOLOv8n checkpoint (`models/hazard_yolov8n.pt`, mAP50 0.495) now wired in as the default model everywhere — see Section 4. Plus the full prototype pipeline: webcam capture, object detection, monocular depth estimation, rule-based decision engine with cooldown, TTS output, on-demand OCR with voice/key trigger, on-demand product ID (generic classifier), and a browser-based UI (`webapp.py`) replacing the OpenCV debug window — see Section 6.
- **Not yet done:**
  - Strengthen the Dataset rubric row further with first-person/glasses-perspective own-captured photos (deferred in favor of the faster curated-only path — see Section 4's caveat).
  - Improve the trained model: more epochs / higher resolution / more images than the current CPU baseline (40 epochs, 320px, ~2000 images).
  - Decide and implement the "novel" element for the Model Development rubric row (architecture tweak or training strategy) — the current fine-tune is a standard transfer-learning run, which per the rubric notes above likely caps at "Satisfactory" on its own.
  - Cloud deployment (the Flask app currently only runs locally) + an optimized serving setup (e.g. TorchServe) for the Model Deployment rubric row's "Excellent" tier.
  - Fine-tune the product ID classifier on a small real currency/product dataset if that branch is worth pursuing further (currently a generic ImageNet classifier, not trained on currency).
  - Earlier planning produced several block-diagram deliverables (hardware/sensing rationale doc, simple/detailed/editable diagrams, an edited 6-box diagram with Passive Safety/Active Assistant modes) in a prior session — unconfirmed whether any were ever added to this repo, and unconfirmed which was actually submitted to xSITe Dropbox. Worth tracking down if they're needed for the report's narrative section.
