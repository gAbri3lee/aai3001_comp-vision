"""Measure the continuous safety processing loop on recorded images, without camera/audio hardware."""
import argparse
import json
from pathlib import Path
import platform
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from config import add_detector_arguments, detector_settings
from detectors import create_detector
from detection import annotate
from decision_engine import assess, phrase_for, AlertCooldown


def main():
    import cv2
    import numpy as np
    parser = argparse.ArgumentParser(description=__doc__)
    add_detector_arguments(parser)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=50)
    parser.add_argument('--no-depth', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.frames < 50:
        parser.error('Measure at least 50 frames')
    settings = detector_settings(args)
    detector = create_detector(settings)
    depth = None
    if not args.no_depth:
        from depth import DepthEstimator
        depth = DepthEstimator()
    paths = sorted(args.images.glob('*.jpg'))
    if not paths:
        parser.error('No JPEG images found')
    cooldown = AlertCooldown.from_gate(settings['alert_gate'])
    durations = []
    for i in range(args.frames + 10):
        start = time.perf_counter()
        frame = cv2.imread(str(paths[i % len(paths)]))
        result = detector.predict(frame)
        depth_map = depth.predict(frame) if depth is not None and result.detections else None
        hazards = assess(result.detections, (frame.shape[1], frame.shape[0]), settings['alert_gate'],
                         settings['severity'], depth_map)
        for hazard in (h for h in hazards if h.gated):
            if cooldown.should_announce(hazard):
                phrase_for(hazard)
        annotated = annotate(frame, result.detections)
        cv2.imencode('.jpg', annotated)
        if i >= 10:
            durations.append(1000 * (time.perf_counter() - start))
    report = dict(platform=platform.platform(), frames=args.frames, warmup_frames=10, depth_enabled=depth is not None,
                  measured_stages=['image read', 'detection', 'optional depth', 'alert gate/ranking/cooldown/message', 'annotation', 'JPEG encoding'],
                  excludes=['physical webcam capture', 'audible speech', 'general COCO object model', 'browser rendering'],
                  median_ms=float(np.median(durations)), p95_ms=float(np.percentile(durations, 95)), fps=1000/float(np.mean(durations)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
