# VisionAid

AAI3001 hazard-awareness prototype. The active local deployment has **four classes**:

| ID | Class |
| ---: | --- |
| 0 | Escalator |
| 1 | People |
| 2 | Stair |
| 3 | Wet Floor Sign |

Pillar was removed from the active dataset taxonomy, local classifier head, UI, demo library and alert configuration on 17 November 2026. Originals and historical results are preserved. Removing a class does not improve the remaining classes.

## Run on Windows

Close an older running instance, double-click `run_demo.bat`, keep its window open, and open **http://127.0.0.1:5001**. This uses the existing environment with optional depth disabled. Speech remains available. Restarting ensures the new config and weights load.

For silent testing, run `run_demo.bat --no-tts` from a terminal in this folder.

Fresh checkout with Python 3.12:

```powershell
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe src/webapp.py --no-depth --port 5001
```

The active checkpoint is `models/local/yolov8n_v4_four_class.pt`. Models/data are git-ignored and need a separate matching handover. An old five-class checkpoint or ONNX export must not simply be renamed: startup rejects class-order mismatch.

Use JPEG/PNG image batches, MP4/MOV/WebM videos, or `demo/<class>/` examples. Upload filenames are display labels, never trusted paths. Local mode does not send frames to cloud.

## Dataset and model provenance

Cleaned dataset: `data/v4_four_class_clean/data.yaml`, **7749 train / 859 validation / 769 test**, fingerprint `d4242de8c1d3776941c614a99dc432a6`. Pillar annotations are removed; useful images and background negatives remain. Pillars may still appear in background scenery.

All 10092 original prepared v4 images decoded and labels passed structural checks. No exact byte/pixel duplicates or recorded scene overlap across splits were found. The all-class perceptual screen flagged similar frames; 715 were conservatively quarantined by excluding them from the derivative. Original images/exclusion reasons remain available. A fresh scan of all 9377 copied images found zero remaining candidate pairs under that screen. This is not exhaustive human reannotation or proof against all crops/venue overlap.

The native four-output local model is an **inference-only projection** of the supplied YOLOv8n checkpoint. Retained class/backbone/regression weights are preserved. **It has not been retrained on the cleaned dataset.** All four classes have identical TP/FP/FN at confidence 0.35 across all 925 original validation and 920 test images. Maximum low-threshold AP50 drop is 0.022 percentage points; exact AP identity is not claimed.

The smaller cleaned test set is a different benchmark. A four-class mean AP cannot be presented as improvement over the old five-class mean. External media still shows weak escalator generalisation and limitations with crowds/stair views. Distance calibration and audible timing remain unverified.

See [migration/audit report](docs/FOUR_CLASS_MIGRATION_2026-11-17.md), [external validation](docs/FOUR_CLASS_VALIDATION_2026-11-17.md) and [rubric evidence](docs/RUBRIC_EVIDENCE.md). Older five-class notebooks and reports retain their original results as historical evidence.

The [18 November development checks](docs/FOUR_CLASS_DEVELOPMENT_2026-11-18.md) record 14 omitted training boxes repaired across seven frames, 212 passing tests plus 12 subtests, actual browser uploads/replay, and the calibration/file-lock fixes. The small three-seed CPU study failed its accuracy gate; the active weights remain unchanged. Physical distance accuracy and representative-user testing remain outstanding.

## Cloud teammate compatibility

The existing RF-DETR checkpoint/service may still have five native outputs. `cloud.source_classes` records that original order. Adapters validate ID/name consistency, discard pillar, and map old Stair 3 to new 2 and old Wet Floor Sign 4 to new 3. The existing HTTP schema is preserved. No hosted cloud deployment was changed.

For a newly trained native four-class RF-DETR model, update `cloud.source_classes` to its four names and run hosted integration tests. Do not just change label strings.

## Verification

```powershell
python -m pip install -r requirements-test.txt
python -m pytest -q --basetemp artifacts/pytest-current
python scripts/verify_detector.py --images golden/images --reference golden/expected/local_yolov8n_four_class.json --frames 0 --output artifacts/golden-four.json
python scripts/verify_detector.py --images data/v4_four_class_clean/test/images --evaluate --frames 0 --output artifacts/four-class-test.json
python scripts/verify_four_class_runtime.py
```

The real upload check uses actual local/general weights, 11 photos and two videos. It verifies IDs, corrupt-file isolation, replay/return-to-camera and cleanup, with camera acquisition replaced by an idle source. It does not verify human audibility, measured distance or hosted cloud.

Cleanup tools: `scripts/audit_four_class_dataset.py`, `scripts/build_four_class_dataset.py`, `scripts/remove_pillar_head.py`, `scripts/compare_four_class_head.py`. Builders refuse to overwrite outputs. Preserve originals when reproducing. Full conversion/audit evidence is under `artifacts/four-class-cleanup/`.

The supplementary COCO model recognises some common obstacles and extra people; it cannot detect every unknown obstacle. Size-based alert bands are not measured metres. Known-height distance estimates apply only to People/sign after actual camera calibration. Use held-out media honestly, label selected demonstrations, retain failures, and do not call online imagery team-recorded footage.
