"""Recall per class by box size on the fixed test split, and a proposed alert-gate min_size.

Size = box height / image height (the alert gate's metric). Predictions are filtered
at the backend's validated class thresholds and matched with the notebook's greedy
IoU>=0.5 rule. Nothing in the config is changed: the proposal is printed for review.

A detector sees pixels, not fractions. The test images are 640x640 (portrait phone
frames letterboxed), so a test box of height fraction s is s x 640 model-input pixels.
A 1280x720 landscape webcam frame becomes 640x360 inside the 640 input, so the same
pixel height is a larger fraction of the webcam frame:
    webcam_fraction = s x test_input_height / deploy_input_height   (640 / 360 = 1.78)
Distances are pinhole ESTIMATES for that assumed webcam, only for classes with a
known real height:  distance_m = focal_px x real_height_m / (webcam_fraction x frame_height_px).
"""
import argparse
import csv
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from config import add_detector_arguments, detector_settings, load_config
from detectors import create_detector
from size_analysis import (SIZE_EDGES, bucket_label, contact_sheet, load_or_predict, load_split,
                           propose_min_size, recall_records, summarise)


def distance_range(low, high, real_height, focal_px, frame_height):
    """Estimated distance band for a webcam box-height fraction range [low, high)."""
    near = focal_px * real_height / (high * frame_height)
    far = math.inf if low == 0 else focal_px * real_height / (low * frame_height)
    return near, far


def fmt_range(near, far):
    return f'{near:.1f} m to {"far" if math.isinf(far) else f"{far:.1f} m"}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_detector_arguments(parser)
    parser.add_argument('--images', type=Path, required=True, help='Test split images folder; labels are in ../labels')
    parser.add_argument('--predictions', type=Path, default=None,
                        help='Prediction cache (default artifacts/<backend>_test_predictions.json)')
    parser.add_argument('--out-dir', type=Path, default=Path('artifacts/recall_by_size'))
    parser.add_argument('--focal-px', type=float, default=914.0,
                        help='Assumed focal length in pixels (914: 720p webcam, about 70 degrees horizontal field of view)')
    parser.add_argument('--frame-height', type=int, default=720, help='Assumed camera frame height in pixels')
    parser.add_argument('--deploy-input-height', type=float, default=360.0,
                        help='Model-input pixel height of the deployment frame (640 x 720/1280 for a 16:9 webcam)')
    args = parser.parse_args()
    settings = detector_settings(args)
    config = load_config(args.config)
    names = config['classes']
    thresholds = settings['class_thresholds']
    known_heights = config['alert_gate']['known_heights_m']

    images, paths = load_split(args.images, names)
    cache = args.predictions or Path(f'artifacts/{args.backend}_test_predictions.json')
    outputs = load_or_predict(cache, lambda: create_detector(settings), paths)
    rows = recall_records(images, outputs, names, thresholds)
    table = summarise(rows, names)
    test_input_height = max(img['height'] for img in images)
    if any(img['height'] != img['width'] or img['height'] != test_input_height for img in images):
        print('Warning: test images are not all square and the same size; pixel conversion is approximate')
    scale = test_input_height / args.deploy_input_height  # test fraction -> webcam fraction
    webcam = lambda f: f * scale

    out = args.out_dir / args.backend
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'recall_by_size.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['backend', 'class', 'test_size_bucket', 'model_input_px', 'webcam_fraction',
                         'boxes', 'hits', 'recall', 'webcam_distance_estimate'])
        for name in names:
            for low, high in zip(SIZE_EDGES, SIZE_EDGES[1:]):
                high = min(high, 1.0)
                cell = table[name][bucket_label(low, high)]
                dist = (fmt_range(*distance_range(webcam(low), webcam(high), known_heights[name], args.focal_px, args.frame_height))
                        if name in known_heights else '')
                writer.writerow([args.backend, name, bucket_label(low, high),
                                 f'{low * test_input_height:.0f}-{high * test_input_height:.0f}',
                                 f'{webcam(low):.2f}-{webcam(high):.2f}', cell['boxes'], cell['hits'],
                                 '' if cell['recall'] is None else f"{cell['recall']:.3f}", dist])

    proposal = {name: propose_min_size(rows, name) for name in names}
    lines = [f'### Recall by box height, {args.backend} ({settings["model"]}, thresholds {thresholds[names[0]]:.2f})', '',
             f'Test box height fraction, model-input pixels, the same pixels as a fraction of a 1280x720 webcam '
             f'frame, and the estimated distance of a 1.7 m person at that size (focal {args.focal_px:.0f} px). '
             'Cells: recall (hits/boxes).', '',
             '| Test fraction | Input px | Webcam fraction | Person distance (estimate) | ' + ' | '.join(names) + ' |',
             '|---|---:|---:|---|' + '---:|' * len(names)]
    for low, high in zip(SIZE_EDGES, SIZE_EDGES[1:]):
        high = min(high, 1.0)
        cells = []
        for name in names:
            c = table[name][bucket_label(low, high)]
            cells.append('-' if not c['boxes'] else f"{c['recall']:.0%} ({c['hits']}/{c['boxes']})")
        wl, wh = webcam(low), webcam(high)
        person = (fmt_range(*distance_range(wl, wh, known_heights['People'], args.focal_px, args.frame_height))
                  if wl < 1 else 'cut off by frame')
        lines.append(f'| {bucket_label(low, high)} | {low * test_input_height:.0f}-{high * test_input_height:.0f} | '
                     f'{wl:.2f}-{wh:.2f} | {person} | ' + ' | '.join(cells) + ' |')
    total = {n: [r for r in rows if r['cls'] == n] for n in names}
    lines.append('| all | | | | ' + ' | '.join(f"{sum(r['hit'] for r in total[n]) / max(len(total[n]), 1):.0%} ({sum(r['hit'] for r in total[n])}/{len(total[n])})" for n in names) + ' |')
    lines += ['', 'Recall floor per class: the smallest size bucket where recall reaches 50 percent, '
                  'for that bucket and for all larger boxes, with at least 5 boxes. Shown as a test fraction '
                  'and as the equivalent webcam fraction (the unit of `alert_gate.min_size`):', '']
    for name in names:
        current = config['alert_gate']['min_size'][name]
        floor = proposal[name]
        text = ('no bucket reaches 50 percent' if floor is None else
                f'test {floor:.2f}, webcam {webcam(floor):.2f}')
        lines.append(f'- {name}: {text} (config min_size now {current})')
    markdown = '\n'.join(lines) + '\n'
    (out / 'recall_by_size.md').write_text(markdown)
    (out / 'recall_by_size.json').write_text(json.dumps(dict(
        backend=args.backend, model=settings['model'], thresholds=thresholds, focal_px=args.focal_px,
        frame_height=args.frame_height, test_input_height=test_input_height,
        deploy_input_height=args.deploy_input_height, table=table, recall_floor_test_fraction=proposal,
        recall_floor_webcam_fraction={n: None if v is None else round(webcam(v), 3) for n, v in proposal.items()},
        truncated_boxes={n: sum(r['truncated'] for r in total[n]) for n in names}), indent=2) + '\n')
    print(markdown)

    # Example People hits and misses: background (small) and up close (large).
    for label, low, high in (('small', 0.0, 0.25), ('large', 0.6, 1.01)):
        for kind, want in (('hits', True), ('misses', False)):
            chosen = sorted((r for r in rows if r['cls'] == 'People' and low <= r['size'] < high and r['hit'] == want),
                            key=lambda r: r['size'])[:12]
            items = []
            for r in chosen:
                dets = outputs[r['image']]['detections']
                items.append(dict(image=r['image'], truth=[r['box']],
                                  predictions=[(d['class'], d['confidence'], d['box_xyxy']) for d in dets
                                               if d['confidence'] >= thresholds[d['class']]],
                                  caption=f"h={r['size']:.2f} " + (f"conf {r['confidence']:.2f}" if want else 'missed')))
            path = contact_sheet(items, args.images, out / f'people_{label}_{kind}.jpg',
                                 f'{args.backend}: People {label} ({low:.2f}-{min(high, 1):.2f}) {kind}. Green: truth, orange: prediction')
            if path:
                print('Saved', path)


if __name__ == '__main__':
    main()
