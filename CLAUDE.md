# VisionAid current project context

The user authorised removing Pillar on 17 November 2026. Active IDs: 0 Escalator, 1 People, 2 Stair, 3 Wet Floor Sign. Config and src/config.py own this contract. Original five-class IDs must be explicitly mapped, never silently reinterpreted.

Active checkpoint: models/local/yolov8n_v4_four_class.pt. It has a native four-output head projected from the supplied checkpoint, with retained weights unchanged. It was not retrained on cleaned data. Original weights, data and metrics remain preserved.

Default local runtime (27 November 2026): models/local/yolov8n_v4_four_class_int8.onnx, a static INT8 quantisation of deploy/vercel/model/yolov8n_v4_four_class.onnx (scripts/quantize_detector.py), committed to Git. It passes golden parity against FP32 but has not been scored against labels; the recorded precision/recall/mAP belong to the FP32 model. --model selects the .pt or FP32 ONNX. Alert defaults (user-approved, 27 November 2026): alert_gate.confirm_frames 2-of-3 for camera/video (still-image analysis uses 1-of-1), local thresholds Stair 0.25 and Wet Floor Sign 0.20 (Escalator/People 0.35; cloud unchanged at 0.40). Not yet validated on labelled images; recorded P/R/mAP are at 0.35 on FP32. Override with --confirm, --class-conf, --conf. See docs/ALERT_CONFIRMATION_AND_INT8_2026-11-27.md.

Clean dataset: data/v4_four_class_clean/data.yaml, 7749/859/769 train/valid/test, manifest d4242de8c1d3776941c614a99dc432a6. Pillar labels removed; useful images/background negatives retained. Background pillars may remain. 715 similar-frame candidates excluded from this derivative, with originals available. Structural/duplicate checks are automated; exhaustive independent annotation review is incomplete.

Cloud belongs to the teammate. Legacy five-class services are supported through explicit cloud.legacy_service_classes mapping, discarding pillar and remapping Stair 3->2 and sign 4->3. Preserve HTTP schema and validate all IDs/names. The default RF-DETR checkpoint has four foreground classes. Reproduce it with scripts/project_four_class_rfdetr.py. Local/fake tests do not verify hosted cloud.

Pipeline: capture -> detector plus optional COCO -> optional relative depth -> AlertPolicy -> bounded speech. General detections retain ID -1; COCO people use People, others Object. COCO is not open-world. Local mode never sends frames to cloud. Preserve fallback, late-result rejection, one request at a time and source-generation invalidation.

Size gates/priorities/cooldowns are unchanged; local Stair and Wet Floor Sign thresholds changed as noted above. Calibration remains false; no measured range claims. Pinhole estimates only for People/sign after actual calibration; never speak verified metres.

Keep upload limits, malformed-media isolation, untrusted-filename/path protections, bounded queues and temporary cleanup. Use workspace-local pytest --basetemp on Windows. Four-class golden reference: golden/expected/local_yolov8n_four_class.json. Real upload check: scripts/verify_four_class_runtime.py.

Old five-class notebooks, config in configs/legacy_v4_five_class.json and research scripts are historical. Do not casually rerun them under the new default taxonomy. Never substitute a new class/subset mean for old benchmark results and call it training improvement. See docs/FOUR_CLASS_MIGRATION_2026-11-17.md and its evidence. No push or merge was performed by this migration.
