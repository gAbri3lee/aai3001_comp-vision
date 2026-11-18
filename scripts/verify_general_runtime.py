"""Actual combined-detector positive/blank/overlap checks, not an accuracy benchmark."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    import cv2
    import numpy as np
    from config import add_detector_arguments, detector_settings
    from scene_detector import SceneDetector
    from decision_engine import assess
    from golden import iou
    parser = argparse.ArgumentParser(description=__doc__)
    add_detector_arguments(parser)
    args = parser.parse_args()
    settings = detector_settings(args)
    detector = SceneDetector(settings)
    report = {'note': 'Selected public functional examples; no representative accuracy claim.', 'inputs': []}
    cases = [(name, cv2.imread(str(ROOT / 'artifacts/system-validation' / name)))
             for name in ('bus.jpg', 'zidane.jpg')]
    cases += [('black blank', np.zeros((480, 640, 3), dtype=np.uint8)),
              ('white blank', np.full((480, 640, 3), 255, dtype=np.uint8))]
    for name, frame in cases:
        assert frame is not None, name
        batch = detector.predict(frame)
        hazards = assess(batch.detections, (frame.shape[1], frame.shape[0]),
                         settings['alert_gate'], settings['severity'])
        general = [d for d in batch.detections if d.class_id == -1]
        trained = [d for d in batch.detections if d.class_id >= 0]
        duplicate = any(iou(a.box_xyxy, b.box_xyxy) >= .5 for a in general for b in trained)
        report['inputs'].append(dict(name=name, detections=[d.to_dict() for d in batch.detections],
            general_labels=[d.label for d in general], duplicate_general_trained_boxes=duplicate,
            gated_labels=[h.label for h in hazards if h.gated], error=detector.general_error))
        assert not duplicate and not detector.general_error
    report['all_blank_inputs_empty'] = all(not r['detections'] for r in report['inputs'][2:])
    report['bus_general_object_seen'] = 'Object' in report['inputs'][0]['general_labels']
    assert report['all_blank_inputs_empty'] and report['bus_general_object_seen']
    path = ROOT / 'artifacts/system-validation/general-functional.json'
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
