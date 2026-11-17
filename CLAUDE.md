# VisionAid current project context

The user authorised removing Pillar on 17 November 2026. Active IDs: 0 Escalator, 1 People, 2 Stair, 3 Wet Floor Sign. Config and src/config.py own this contract. Original five-class IDs must be explicitly mapped, never silently reinterpreted.

Active checkpoint: models/local/yolov8n_v4_four_class.pt. It has a native four-output head projected from the supplied checkpoint, with retained weights unchanged. It was not retrained on cleaned data. Original weights, data and metrics remain preserved.

Clean dataset: data/v4_four_class_clean/data.yaml, 7749/859/769 train/valid/test, manifest d4242de8c1d3776941c614a99dc432a6. Pillar labels removed; useful images/background negatives retained. Background pillars may remain. 715 similar-frame candidates excluded from this derivative, with originals available. Structural/duplicate checks are automated; exhaustive independent annotation review is incomplete.

Cloud belongs to the teammate. Existing five-class RF-DETR is supported through explicit cloud.source_classes mapping, discarding pillar and remapping Stair 3->2 and sign 4->3. Preserve HTTP schema and validate all IDs/names. Native four-class cloud weights require a matching source_classes update and hosted integration tests. Local/fake tests do not verify hosted cloud.

Pipeline: capture -> detector plus optional COCO -> optional relative depth -> AlertPolicy -> bounded speech. General detections retain ID -1; COCO people use People, others Object. COCO is not open-world. Local mode never sends frames to cloud. Preserve fallback, late-result rejection, one request at a time and source-generation invalidation.

Retained thresholds/size gates/priorities/cooldowns are unchanged. Calibration remains false; no measured range claims. Pinhole estimates only for People/sign after actual calibration; never speak verified metres.

Keep upload limits, malformed-media isolation, untrusted-filename/path protections, bounded queues and temporary cleanup. Use workspace-local pytest --basetemp on Windows. Four-class golden reference: golden/expected/local_yolov8n_four_class.json. Real upload check: scripts/verify_four_class_runtime.py.

Old five-class notebooks, config in configs/legacy_v4_five_class.json and research scripts are historical. Do not casually rerun them under the new default taxonomy. Never substitute a new class/subset mean for old benchmark results and call it training improvement. See docs/FOUR_CLASS_MIGRATION_2026-11-17.md and its evidence. No push or merge was performed by this migration.
