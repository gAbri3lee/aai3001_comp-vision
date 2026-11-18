"""Study confidence operating points on the frozen validation split only.

Consumes cached low-confidence predictions; never trains or edits app thresholds.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from config import load_config
from evaluation import load_ground_truth
from golden import iou


def count_matches(truth, detections, cid, threshold):
    boxes = [box for label, box in truth if label == cid]
    used = set()
    tp = fp = 0
    for detection in sorted((d for d in detections if d['class_id'] == cid and
                             d['confidence'] >= threshold), key=lambda d: -d['confidence']):
        candidates = [j for j in range(len(boxes)) if j not in used]
        best = max(candidates, key=lambda j: iou(detection['box_xyxy'], boxes[j]), default=None)
        if best is not None and iou(detection['box_xyxy'], boxes[best]) >= .5:
            used.add(best)
            tp += 1
        else:
            fp += 1
    return tp, fp, len(boxes) - tp


def operating_point(counts, threshold):
    tp, fp, fn = map(int, counts)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return dict(threshold=threshold, tp=tp, fp=fp, fn=fn, precision=precision,
                recall=recall, f2=5 * precision * recall / max(4 * precision + recall, 1e-12))


def choose_candidate(rows, baseline, precision_floor=.75, min_recall_gain=.05):
    floor = max(precision_floor, baseline['precision'] - .02)
    feasible = [r for r in rows if r['precision'] >= floor and
                r['recall'] >= baseline['recall'] + min_recall_gain]
    return max(feasible, key=lambda r: (r['recall'], r['precision'], r['threshold']), default=None)


def main():
    import numpy as np
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True, help='Prepared dataset root with split_manifest.csv')
    parser.add_argument('--predictions', type=Path, required=True, help='verify_detector --evaluate output, confidence >=.001')
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = load_config()
    names = config['classes']
    root = args.dataset.resolve()
    rows = list(csv.DictReader((root / 'split_manifest.csv').read_text().splitlines()))
    expected = {r['stem']: r['scene'] for r in rows if r['split'] == 'val'}
    # The notebook manifest calls validation "val". Accept "valid" for equivalent exports.
    expected.update({r['stem']: r['scene'] for r in rows if r['split'] == 'valid'})
    gt, paths = load_ground_truth(root / 'valid', names)
    outputs = json.loads(args.predictions.read_text())
    if set(expected) != {p.stem for p in paths} or set(outputs) != {p.name for p in paths}:
        raise ValueError('Predictions must cover exactly the manifest validation images, including backgrounds')
    truth = defaultdict(list)
    for a in gt['annotations']:
        x, y, w, h = a['bbox']
        truth[a['image_id']].append((a['category_id'] - 1, [x, y, x + w, y + h]))
    scene_names = sorted(set(expected.values()))
    scene_index = {scene: i for i, scene in enumerate(scene_names)}
    rng = np.random.default_rng(0)
    bootstrap = rng.integers(0, len(scene_names), (2000, len(scene_names)))
    report = dict(split='validation', images=len(paths), scenes=len(scene_names),
                  matching_iou=.5, model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
                  predictions_sha256=hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
                  split_manifest_md5=hashlib.md5((root / 'split_manifest.csv').read_bytes()).hexdigest(),
                  test_used=False, application_changed=False, classes={})
    curve = []
    for cid, name in enumerate(names):
        base = config['local']['class_thresholds'][name]
        thresholds = sorted(set([round(i / 100, 2) for i in range(5, 91)] + [base]))
        points = []
        baseline_by_scene = np.zeros((len(scene_names), 3), dtype=int)
        for threshold in thresholds:
            counts = np.zeros(3, dtype=int)
            for i, path in enumerate(paths):
                result = count_matches(truth[i], outputs[path.name]['detections'], cid, threshold)
                counts += result
                if threshold == base:
                    baseline_by_scene[scene_index[expected[path.stem]]] += result
            point = operating_point(counts, threshold)
            points.append(point)
            curve.append(dict(class_name=name, **point))
        baseline = next(p for p in points if p['threshold'] == base)
        samples = baseline_by_scene[bootstrap].sum(axis=1)
        recalls = samples[:, 0] / np.maximum(samples[:, 0] + samples[:, 2], 1)
        supported = samples[:, 0] + samples[:, 2] > 0
        candidate = choose_candidate(points, baseline)
        report['classes'][name] = dict(baseline=baseline, candidate=candidate,
            recall_scene_bootstrap_95_interval=np.quantile(recalls[supported], [.025, .975]).tolist(),
            interval_note='Resamples recorded scene groups, not frames; depends on scene grouping and label completeness.',
            decision='Candidate needs new independent evaluation; do not deploy automatically.' if candidate else
                     'No threshold meets precision floor and five-percentage-point recall gain. Keep baseline.')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'study.json').write_text(json.dumps(report, indent=2))
    with (args.output / 'threshold_curve.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(curve[0]))
        writer.writeheader()
        writer.writerows(curve)
    lines = ['# Validation operating-point study', '',
             'Validation only. Candidates are exploratory and never modify application settings.', '',
             '| Class | Current precision | Current recall | Candidate threshold | Candidate precision | Candidate recall |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name, data in report['classes'].items():
        b, c = data['baseline'], data['candidate']
        lines.append(f"| {name} | {b['precision']:.1%} | {b['recall']:.1%} | " +
            (f"{c['threshold']:.2f} | {c['precision']:.1%} | {c['recall']:.1%} |" if c else 'none | — | — |'))
    lines += ['', 'Selection rule fixed before inspecting these curves: precision >= max(75%, current precision minus 2 percentage points), recall gain >=5 percentage points. Maximize recall, then precision.',
              '', 'No test tuning. Do not claim these selected validation results as final test performance. A candidate changes false-alert behavior and needs an independent field evaluation.',
              '', 'Scene bootstrap intervals quantify sampling variation within the recorded scene groups. They do not measure domain shift, annotation errors or true physical distance.']
    (args.output / 'study.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
