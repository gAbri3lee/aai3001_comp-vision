"""Recall by object size on the fixed test split, for distance bands and Pillar review.

Size is box height divided by image height, the same metric as the alert gate.
Matching is the notebook rule used by evaluation.py: per class, predictions above
the alert threshold in confidence order, each greedily matched to the unused
ground-truth box with the highest IoU, counted as a hit when IoU >= 0.5.
"""
import json
from pathlib import Path

from evaluation import load_ground_truth
from golden import iou

SIZE_EDGES = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.80, 1.01)


def bucket_label(low, high):
    return f'{low:.2f}-{min(high, 1.0):.2f}'


def bucket_of(size):
    for low, high in zip(SIZE_EDGES, SIZE_EDGES[1:]):
        if low <= size < high:
            return bucket_label(low, high)
    return bucket_label(SIZE_EDGES[-2], SIZE_EDGES[-1])


def load_split(images_dir, names):
    """Per image: size and ground-truth boxes as (class_id, xyxy)."""
    gt, paths = load_ground_truth(Path(images_dir).parent, names)
    images = {img['id']: dict(name=img['file_name'], width=img['width'], height=img['height'], truth=[])
              for img in gt['images']}
    for ann in gt['annotations']:
        x, y, w, h = ann['bbox']
        images[ann['image_id']]['truth'].append((ann['category_id'] - 1, (x, y, x + w, y + h)))
    return [images[i] for i in sorted(images)], paths


def predict_all(detector, paths, progress=True):
    import cv2
    outputs = {}
    for i, path in enumerate(paths):
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f'Could not decode {path}')
        outputs[path.name] = detector.predict(frame).to_dict()
        if progress and (i + 1) % 100 == 0:
            print(f'Processed {i + 1}/{len(paths)}', flush=True)
    return outputs


def load_or_predict(cache, detector_factory, paths):
    """Reuse saved predictions for exactly these images, otherwise run the detector."""
    cache = Path(cache)
    if cache.exists():
        outputs = json.loads(cache.read_text())
        if set(outputs) == {p.name for p in paths}:
            return outputs
    outputs = predict_all(detector_factory(), paths)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(outputs) + '\n')
    return outputs


def match_image(truth, detections, class_id, threshold):
    """Greedy IoU>=0.5 matching for one class. Returns (truth_hit flags, prediction records)."""
    boxes = [box for cid, box in truth if cid == class_id]
    hit = [False] * len(boxes)
    predictions = sorted((d for d in detections if d['class_id'] == class_id and d['confidence'] >= threshold),
                         key=lambda d: d['confidence'], reverse=True)
    records = []
    for detection in predictions:
        candidates = [j for j in range(len(boxes)) if not hit[j]]
        best = max(candidates, key=lambda j: iou(detection['box_xyxy'], boxes[j]), default=None)
        matched = best is not None and iou(detection['box_xyxy'], boxes[best]) >= .5
        if matched:
            hit[best] = True
        records.append(dict(detection, matched=matched, matched_index=best if matched else None))
    return boxes, hit, records


def recall_records(images, outputs, names, thresholds):
    """One record per ground-truth box: class, size, bucket, hit, matched confidence."""
    rows = []
    for image in images:
        detections = outputs[image['name']]['detections']
        for cid, name in enumerate(names):
            boxes, hit, records = match_image(image['truth'], detections, cid, thresholds[name])
            confidence = {r['matched_index']: r['confidence'] for r in records if r['matched']}
            for j, box in enumerate(boxes):
                size = (box[3] - box[1]) / image['height']
                rows.append(dict(image=image['name'], cls=name, size=size, bucket=bucket_of(size),
                                 hit=hit[j], confidence=confidence.get(j), box=box,
                                 truncated=box[1] <= 1 or box[3] >= image['height'] - 1))
    return rows


def summarise(rows, names):
    """Per class and size bucket: boxes, hits and recall."""
    buckets = [bucket_label(lo, hi) for lo, hi in zip(SIZE_EDGES, SIZE_EDGES[1:])]
    table = {}
    for name in names:
        table[name] = {}
        for bucket in buckets:
            subset = [r for r in rows if r['cls'] == name and r['bucket'] == bucket]
            hits = sum(r['hit'] for r in subset)
            table[name][bucket] = dict(boxes=len(subset), hits=hits,
                                       recall=hits / len(subset) if subset else None)
    return table


def propose_min_size(rows, name, target=0.5, min_boxes=5):
    """Smallest bucket lower edge from which recall of every box at least that size
    reaches the target, and that bucket itself (with enough boxes) reaches it too."""
    for low, high in zip(SIZE_EDGES, SIZE_EDGES[1:]):
        here = [r for r in rows if r['cls'] == name and low <= r['size'] < high]
        above = [r for r in rows if r['cls'] == name and r['size'] >= low]
        if len(here) < min_boxes or not above:
            continue
        if sum(r['hit'] for r in here) / len(here) >= target and sum(r['hit'] for r in above) / len(above) >= target:
            return low
    return None


def contact_sheet(items, images_dir, out_path, title, tile=256, columns=4):
    """Draw up to len(items) tiles: ground truth in green, predictions in orange.

    Each item: dict(image=name, truth=[xyxy], predictions=[(label, conf, xyxy)], caption=str).
    """
    import cv2
    import numpy as np
    if not items:
        return None
    rows = (len(items) + columns - 1) // columns
    sheet = np.full((rows * (tile + 22) + 30, columns * tile, 3), 24, dtype=np.uint8)
    cv2.putText(sheet, title, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, .6, (235, 235, 235), 1)
    for k, item in enumerate(items):
        frame = cv2.imread(str(Path(images_dir) / item['image']))
        for box in item['truth']:
            cv2.rectangle(frame, tuple(map(int, box[:2])), tuple(map(int, box[2:])), (80, 220, 80), 3)
        for label, confidence, box in item['predictions']:
            cv2.rectangle(frame, tuple(map(int, box[:2])), tuple(map(int, box[2:])), (0, 150, 255), 2)
            cv2.putText(frame, f'{label} {confidence:.2f}', (int(box[0]) + 3, int(box[1]) + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 150, 255), 2)
        frame = cv2.resize(frame, (tile, tile))
        r, c = divmod(k, columns)
        y, x = 30 + r * (tile + 22), c * tile
        sheet[y:y + tile, x:x + tile] = frame
        cv2.putText(sheet, item['caption'][:40], (x + 4, y + tile + 15), cv2.FONT_HERSHEY_SIMPLEX, .42, (220, 220, 220), 1)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)
    return out_path
