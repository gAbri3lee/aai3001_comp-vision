"""Sweep per-class alert cut-offs on a labelled split and apply the team's pre-fixed selection rule.

Consumes `verify_detector.py --evaluate` output (all detections down to confidence 0.001), so the model runs once:

    python scripts/verify_detector.py --images data/v4_four_class_clean/valid/images --evaluate --frames 0 \
        --model models/local/yolov8n_v4_four_class_int8.onnx --conf 0.35 --output artifacts/eval_valid/int8.json
    python scripts/sweep_class_thresholds.py --split data/v4_four_class_clean/valid \
        --predictions artifacts/eval_valid/int8.json --output artifacts/eval_valid/sweep.json

The rule is `choose_candidate` from scripts/validation_study.py: recall must gain at least 5 points over the
baseline and precision must stay at or above max(75%, baseline precision minus 2 points). Use the validation
split to choose; report the chosen setting on the test split.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from config import load_config
from evaluation import load_ground_truth
from validation_study import choose_candidate, count_matches, operating_point


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--split', type=Path, required=True, help='Split folder with images/ and labels/')
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--baseline', type=float, default=.35)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    names = load_config()['classes']
    gt, paths = load_ground_truth(args.split, names)
    outputs = json.loads(args.predictions.read_text())
    if set(outputs) != {p.name for p in paths}:
        raise ValueError('Predictions must cover exactly the split images, including backgrounds')
    truth = defaultdict(list)
    for a in gt['annotations']:
        x, y, w, h = a['bbox']
        truth[a['image_id']].append((a['category_id'] - 1, [x, y, x + w, y + h]))
    thresholds = sorted({round(i / 100, 2) for i in range(5, 91)} | {args.baseline})
    report = dict(split=str(args.split), images=len(paths),
                  predictions_sha256=hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
                  rule='recall gain >= 5 points and precision >= max(75%, baseline - 2 points); '
                       'maximise recall, then precision (scripts/validation_study.py choose_candidate)',
                  baseline_threshold=args.baseline, classes={})
    for cid, name in enumerate(names):
        points = []
        for threshold in thresholds:
            counts = [0, 0, 0]
            for i, path in enumerate(paths):
                result = count_matches(truth[i], outputs[path.name]['detections'], cid, threshold)
                counts = [a + b for a, b in zip(counts, result)]
            points.append(operating_point(counts, threshold))
        baseline = next(p for p in points if p['threshold'] == args.baseline)
        candidate = choose_candidate(points, baseline)
        report['classes'][name] = dict(baseline=baseline, candidate=candidate,
                                       decision=f"Use {candidate['threshold']:.2f}" if candidate else
                                       f'No cut-off meets the rule. Keep {args.baseline:.2f}.', curve=points)
        print(f"{name:<15} baseline {args.baseline:.2f}: P {baseline['precision']:.1%} R {baseline['recall']:.1%}"
              f" -> {report['classes'][name]['decision']}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1) + '\n')


if __name__ == '__main__':
    main()
