"""Deployment parity check, with strict image coverage and useful exit status."""
import argparse
import json
from pathlib import Path


def iou(a, b):
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter + 1e-9)


def compare_outputs(reference, outputs):
    matched = total = extra = 0
    confidence_differences = []
    for name, item in reference.items():
        remaining = list(outputs.get(name, {}).get('detections', []))
        for detection in item['detections']:
            total += 1
            candidates = [d for d in remaining if d['class_id'] == detection['class_id']]
            best = max(candidates, key=lambda d: iou(d['box_xyxy'], detection['box_xyxy']), default=None)
            if best is not None and iou(best['box_xyxy'], detection['box_xyxy']) >= .9:
                matched += 1
                remaining.remove(best)
                confidence_differences.append(abs(best['confidence'] - detection['confidence']))
        extra += len(remaining)
    missing, unexpected = sorted(reference.keys() - outputs.keys()), sorted(outputs.keys() - reference.keys())
    passed = bool(reference) and not missing and not unexpected
    passed = passed and (matched / total >= .95 and extra <= max(2, .05 * total) if total else extra == 0)
    return dict(passed=passed, matched=matched, total=total, extra=extra,
                missing_images=missing, unexpected_images=unexpected,
                largest_confidence_difference=max(confidence_differences, default=0.))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('outputs', type=Path)
    args = parser.parse_args()
    result = compare_outputs(json.loads(args.reference.read_text()), json.loads(args.outputs.read_text()))
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
