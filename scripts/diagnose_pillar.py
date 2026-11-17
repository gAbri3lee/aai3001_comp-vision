"""Classify existing pillar failures without changing deployment thresholds.

This is diagnostic use of the historical test set, not tuning or training data.
"""
import csv
import json
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from config import load_config
from golden import iou
from size_analysis import load_split, match_image, contact_sheet


def main():
    config = load_config()
    folder = ROOT / 'artifacts/supplied_weights/test_set/test/images'
    images, _ = load_split(folder, config['classes'])
    outputs = json.loads((ROOT / 'artifacts/real-model-test.json').read_text())
    out = ROOT / 'artifacts/pillar-diagnosis'
    out.mkdir(parents=True, exist_ok=True)
    rows, tiles = [], []
    for image in images:
        detections = outputs[image['name']]['detections']
        boxes, hits, _ = match_image(image['truth'], detections, 2, .35)
        for box, hit in zip(boxes, hits):
            overlapping = sorted((d for d in detections if iou(d['box_xyxy'], box) >= .5),
                                 key=lambda d: d['confidence'], reverse=True)
            pillar = [d for d in overlapping if d['class_id'] == 2]
            other = [d for d in overlapping if d['class_id'] != 2 and d['confidence'] >= .35]
            reason = ('hit' if hit else 'pillar_below_threshold' if pillar else
                      'other_class_overlap' if other else 'no_localized_box_at_iou_0.5')
            rows.append(dict(image=image['name'], reason=reason,
                             size=(box[3]-box[1])/image['height'],
                             best_pillar_confidence=max((d['confidence'] for d in pillar), default=None),
                             other_classes='|'.join(sorted({d['class'] for d in other}))))
            tiles.append(dict(image=image['name'], truth=[box],
                              predictions=[(d['class'], d['confidence'], d['box_xyxy'])
                                           for d in detections if d['confidence'] >= .35],
                              caption=f"{len(rows)-1}: {reason}"))
    with (out / 'failures.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for start in range(0, len(tiles), 24):
        contact_sheet(tiles[start:start+24], folder, out / f'pillars_{start//24:02d}.jpg',
                      'Pillar diagnosis: green labels, orange model outputs')
    report = dict(boxes=len(rows), categories=dict(Counter(r['reason'] for r in rows)),
                  limitations='IoU overlap categories describe outputs, not confirmed root causes. Do not tune on these test images.')
    (out / 'summary.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
