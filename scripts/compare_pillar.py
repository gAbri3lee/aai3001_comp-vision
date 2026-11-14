"""Compare Pillar detection between the local and cloud detectors on the fixed test split.

Pillar is the weakest class. A missed pillar is a training limitation, not a bug, and
nothing here changes thresholds or code. Uses every test image with a Pillar label
and every image without one (for false alarms), at each backend's validated
threshold, with the notebook's greedy IoU>=0.5 matching.

Predictions are reused from artifacts/<backend>_test_predictions.json (written by
recall_by_size.py) or computed. The cloud half needs the RF-DETR service; without
--endpoint only the local half runs.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from config import add_detector_arguments, detector_settings, load_config
from detectors import create_detector
from golden import iou
from size_analysis import (SIZE_EDGES, bucket_label, contact_sheet, load_or_predict, load_split,
                           match_image, recall_records, summarise)

PILLAR = 2


def settings_for(backend, endpoint, timeout):
    parser = argparse.ArgumentParser()
    add_detector_arguments(parser)
    argv = ['--backend', backend, '--timeout', str(timeout), '--max-frame-age', str(timeout)]
    if endpoint:
        argv += ['--endpoint', endpoint]
    return detector_settings(parser.parse_args(argv))


def evaluate(backend, settings, images, paths, names, out, images_dir):
    outputs = load_or_predict(Path(f'artifacts/{backend}_test_predictions.json'),
                              lambda: create_detector(settings), paths)
    threshold = settings['class_thresholds']['Pillar']
    with_pillar = [img for img in images if any(cid == PILLAR for cid, _ in img['truth'])]
    without = [img for img in images if img not in with_pillar]
    tp = fp = 0
    hit_conf, all_conf = [], []
    hits, misses = [], []
    for img in images:
        boxes, hit, records = match_image(img['truth'], outputs[img['name']]['detections'], PILLAR, threshold)
        tp += sum(r['matched'] for r in records)
        fp += sum(not r['matched'] for r in records)
        hit_conf += [r['confidence'] for r in records if r['matched']]
        all_conf += [r['confidence'] for r in records]
        preds = [(r['class'], r['confidence'], r['box_xyxy']) for r in records]
        for j, box in enumerate(boxes):
            size = (box[3] - box[1]) / img['height']
            item = dict(image=img['name'], truth=[box], predictions=preds, size=size, missed_box=box,
                        caption=f'h={size:.2f} ' + ('hit' if hit[j] else 'missed'))
            (hits if hit[j] else misses).append(item)
    false_alarm_images = sum(
        any(d['class_id'] == PILLAR and d['confidence'] >= threshold for d in outputs[img['name']]['detections'])
        for img in without)
    rows = [r for r in recall_records(images, outputs, names, settings['class_thresholds']) if r['cls'] == 'Pillar']
    by_size = summarise(rows, names)['Pillar']
    total = tp + len(misses)
    result = dict(backend=backend, model=settings['model'], threshold=threshold,
                  images_with_pillar=len(with_pillar), images_without_pillar=len(without),
                  pillar_boxes=len(hits) + len(misses), hits=len(hits), misses=len(misses),
                  recall=len(hits) / max(len(hits) + len(misses), 1),
                  precision=tp / max(tp + fp, 1), predictions=tp + fp, false_positives=fp,
                  false_alarm_images_without_pillar=false_alarm_images,
                  mean_confidence_hits=statistics.mean(hit_conf) if hit_conf else None,
                  mean_confidence_all_predictions=statistics.mean(all_conf) if all_conf else None,
                  recall_by_size=by_size)
    for kind, items in (('hits', hits), ('misses', misses)):
        # Spread the examples across sizes rather than taking only the smallest.
        items = sorted(items, key=lambda i: i['size'])
        step = max(1, len(items) // 12)
        contact_sheet(items[::step][:12], images_dir, out / f'pillar_{backend}_{kind}.jpg',
                      f'{backend}: Pillar {kind} (green: truth, orange: Pillar predictions >= {threshold:.2f})')
    return result, misses


def general_cover(misses, images_dir, confidence=0.5):
    """How many missed pillars a gated generic Object box would cover (IoU >= 0.5)."""
    import cv2
    from scene_detector import load_general_model
    model = load_general_model()
    covered = 0
    for item in misses:
        frame = cv2.imread(str(Path(images_dir) / item['image']))
        result = model.predict(frame, imgsz=640, conf=confidence, iou=.7, verbose=False)[0]
        boxes = [row[:4] for row in result.boxes.data.cpu().numpy()]
        covered += any(iou(tuple(map(float, b)), item['missed_box']) >= .5 for b in boxes)
    return covered


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--endpoint', default=None, help='RF-DETR service URL; omit to run only the local half')
    parser.add_argument('--timeout', type=float, default=60.0)
    parser.add_argument('--out-dir', type=Path, default=Path('artifacts/pillar'))
    parser.add_argument('--skip-general', action='store_true', help='Skip the general-model check on missed pillars')
    args = parser.parse_args()
    names = load_config()['classes']
    images, paths = load_split(args.images, names)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    backends = ['local'] + (['cloud'] if args.endpoint else [])
    if not args.endpoint:
        print('No --endpoint: the cloud service is not used, so only the local half runs.')
    report = {}
    for backend in backends:
        result, misses = evaluate(backend, settings_for(backend, args.endpoint, args.timeout),
                                  images, paths, names, args.out_dir, args.images)
        if not args.skip_general:
            result['misses_covered_by_general_object_box'] = general_cover(misses, args.images)
        report[backend] = result

    lines = ['| | ' + ' | '.join(f"{b} ({report[b]['model']}, {report[b]['threshold']:.2f})" for b in backends) + ' |',
             '|---|' + '---:|' * len(backends)]
    def row(label, fn):
        lines.append(f'| {label} | ' + ' | '.join(fn(report[b]) for b in backends) + ' |')
    row('Pillar boxes (images)', lambda r: f"{r['pillar_boxes']} ({r['images_with_pillar']})")
    row('Recall', lambda r: f"{r['recall']:.1%} ({r['hits']} hits, {r['misses']} misses)")
    row('Precision', lambda r: f"{r['precision']:.1%} ({r['false_positives']} false of {r['predictions']})")
    row('Images without a pillar that got a Pillar box', lambda r: f"{r['false_alarm_images_without_pillar']} of {r['images_without_pillar']}")
    row('Mean confidence of hits', lambda r: '-' if r['mean_confidence_hits'] is None else f"{r['mean_confidence_hits']:.2f}")
    if not args.skip_general:
        row('Missed pillars covered by a general Object box (conf >= 0.5, IoU >= 0.5)',
            lambda r: f"{r['misses_covered_by_general_object_box']} of {r['misses']}")
    for low, high in zip(SIZE_EDGES, SIZE_EDGES[1:]):
        key = bucket_label(low, high)
        if any(report[b]['recall_by_size'][key]['boxes'] for b in backends):
            row(f'Recall, height {key}', lambda r, k=key: (
                '-' if not r['recall_by_size'][k]['boxes'] else
                f"{r['recall_by_size'][k]['recall']:.0%} ({r['recall_by_size'][k]['hits']}/{r['recall_by_size'][k]['boxes']})"))
    markdown = '\n'.join(lines) + '\n'
    (args.out_dir / 'pillar_comparison.md').write_text(markdown)
    (args.out_dir / 'pillar_comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    print(markdown)


if __name__ == '__main__':
    main()
