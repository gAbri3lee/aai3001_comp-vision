# VisionAid — current project context

Software-only AAI3001 hazard-awareness prototype. The current training work is
in the five trained notebooks. The dataset-builder and training CLI files have
been removed; use the notebooks for training. Deployment utilities and the
prepared-dataset validator remain available.

## Final dataset and classes

Roboflow `potatos-workspace-kequm/version2-nvt5l` v4, scene-based split fingerprint
`49601080aac1f838190341d331785578`. Train/valid/test: 8247/925/920 images.
IDs must never be reordered: 0 Escalator, 1 People, 2 Pillar, 3 Stair,
4 Wet Floor Sign. `visionaid_config.json` owns class names, thresholds, severity,
model paths, input sizes and recorded benchmark metrics.

## Runtime architecture

The default is local YOLOv8n (`models/local/yolov8n_v4.pt`), Ultralytics 8.4.183,
640 px, confidence 0.35, NMS IoU 0.7. The optional remote detector is RF-DETR Nano
(`models/cloud/rfdetr-nano_v4.pth`), RF-DETR 1.11.2, 384 px, confidence 0.40.
RF-DETR has a separate Linux runtime; do not pass its `.pth` to Ultralytics YOLO.

All demos consume shared Detection/DetectionBatch records. Boxes are xyxy pixels
in the original frame. Checkpoint labels must match the canonical IDs at startup.
The pipeline is capture → detector → optional relative depth → heuristic ranking
→ cooldown → audio, with on-demand OCR and generic ImageNet item identification.
Without depth, class/location notices still work and say distance unavailable.
There is no measured distance, stair direction classifier, floor-wetness sensor,
face recognition, route navigation, physical wearable build or currency model.

The local UI remains the camera/audio host in cloud mode. Camera frames leave the
device only when cloud is explicitly selected. No automatic fallback or fusion
between the two detectors is implemented. One request is allowed at a time; newest
frames replace buffered frames and obsolete cloud responses are discarded.

## Training and evaluation

Do not retrain to integrate these checkpoints or change the finalized class set.
Keep notebook outputs and the recorded train/valid/test split. API credentials
come from environment/Kaggle secrets. The previously exposed Roboflow key must be
rotated in its account; source cleanup does not revoke old keys or git history.
The five notebooks preserve their training/evaluation outputs.

`src/evaluation.py` reproduces the notebook's COCO AP and greedy IoU-0.5 alert
metrics. `scripts/verify_detector.py` verifies golden parity, evaluates a supplied
fixed test split and benchmarks batch-one inference. AP uses detections >=0.001;
alert metrics use the existing validation-selected threshold, never retuned on
test. Keep Ultralytics `.val()` results separately labelled if measured.

## Boundaries and limitations

Depth is frame-relative, not metres. Pillar recall is weak in the saved results.
A Wet Floor Sign is a warning sign, not proof of a spill. A lack of detections is
not a declaration that a scene is safe. This is supplementary hazard awareness.
OCR remains synchronous and intended for short signs; voice recognition requires
internet/microphone access. Product identification is generic ImageNet inference.

See README.md for setup, commands, cloud container and export verification.
