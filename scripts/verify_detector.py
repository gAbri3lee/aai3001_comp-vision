"""Golden parity, fixed-split evaluation and warmed batch-one inference timings."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import platform
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from config import add_detector_arguments, detector_settings, load_config, resolve_path
from detection import DetectionBatch
from detectors import create_detector
from golden import compare_outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_detector_arguments(parser)
    parser.add_argument('--runtime', choices=['yolo', 'rfdetr', 'cloud'], default='yolo')
    parser.add_argument('--images', required=True, type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--evaluate', action='store_true', help='Images must be a split/images folder; loads sibling labels')
    parser.add_argument('--frames', type=int, default=50, help='Additional warmed detector benchmark frames; 0 disables')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.frames < 0:
        parser.error('--frames must not be negative')
    if args.evaluate and args.runtime == 'cloud':
        parser.error('Full AP requires conf>=0.001; evaluate the RF-DETR checkpoint directly with --runtime rfdetr')
    args.backend = 'cloud' if args.runtime == 'cloud' else 'local'
    settings = detector_settings(args)
    config = load_config(args.config)
    spec = config['cloud'] if args.runtime in ('rfdetr', 'cloud') else config['local']
    alert_threshold = args.conf if args.conf is not None else min(spec['class_thresholds'].values())
    if args.evaluate:
        settings['confidence'] = .001
        settings['class_thresholds'] = dict.fromkeys(config['classes'], .001)
    if args.runtime == 'rfdetr':
        from rfdetr_detector import RFDetrDetector
        from PIL import Image
        if args.evaluate:
            config['cloud']['class_thresholds'] = dict.fromkeys(config['classes'], .001)
        detector = RFDetrDetector(config, resolve_path(args.model or config['cloud']['file']), args.device)
        def predict(path):
            with Image.open(path) as image:
                return detector.predict(image.convert('RGB'))
    else:
        import cv2
        detector = create_detector(settings)
        def predict(path):
            frame = cv2.imread(str(path))
            if frame is None:
                raise ValueError(f'Could not decode {path}')
            return detector.predict(frame)
    paths = sorted(p for p in args.images.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
    if not paths:
        parser.error('No images found')
    outputs = {}
    for i, path in enumerate(paths):
        outputs[path.name] = predict(path).to_dict()
        if (i + 1) % 100 == 0:
            print(f'Processed {i+1}/{len(paths)}', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(outputs, indent=2) + '\n')
    report = dict(runtime=args.runtime, platform=platform.platform(), device=args.device,
                  images=len(paths), input_size=spec['input_size'] if args.runtime != 'yolo' else settings['input_size'],
                  alert_threshold=alert_threshold)
    if args.reference:
        if args.evaluate:
            # AP inputs are low-threshold; compare golden outputs at the validated alert threshold.
            parity_outputs = {k: {**v, 'detections': [d for d in v['detections'] if d['confidence'] >= alert_threshold]}
                              for k, v in outputs.items()}
        else:
            parity_outputs = outputs
        report['golden'] = compare_outputs(json.loads(args.reference.read_text()), parity_outputs)
    if args.evaluate:
        from evaluation import load_ground_truth, score_detections
        gt, evaluated_paths = load_ground_truth(args.images.parent, config['classes'])
        report['metrics'] = score_detections(gt, evaluated_paths, outputs, config['classes'], alert_threshold, config['safety_classes'])
    if args.frames:
        import numpy as np
        for i in range(10):
            predict(paths[i % len(paths)])
        durations = []
        for i in range(args.frames):
            batch = predict(paths[i % len(paths)])
            durations.append(batch.round_trip_ms if batch.round_trip_ms is not None else batch.latency_ms)
        report['detector_benchmark'] = dict(warmup_frames=10, frames=args.frames,
            median_ms=statistics.median(durations), p95_ms=float(np.percentile(durations, 95)),
            fps=1000/statistics.mean(durations))
    report_path = args.output.with_suffix('.report.json')
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if args.reference and not report['golden']['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
