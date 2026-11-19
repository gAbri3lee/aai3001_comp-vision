"""Notebook-equivalent COCO AP and greedy IoU-0.5 alert metrics on a fixed split."""
from collections import Counter, defaultdict
import contextlib
import io
from pathlib import Path
import numpy as np
from PIL import Image
from golden import iou
from path_utils import native_path


def load_ground_truth(split_directory, names, label_names=None):
    """COCO ground truth for a YOLO split in the active class order `names`.

    `label_names` is the class order the label files were written in, when it
    differs (for example the original five-class v4 split). Classes missing from
    `names` are dropped and the rest are remapped by name, never by position.
    """
    if label_names is not None and list(label_names) != list(names):
        gt, paths = load_ground_truth(split_directory, label_names)
        return remap_ground_truth(gt, label_names, names), paths
    root = native_path(split_directory)
    paths = sorted(p for p in (root / 'images').iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
    if not paths:
        raise ValueError('No test images found')
    images, annotations = [], []
    for image_id, path in enumerate(paths):
        with Image.open(path) as image:
            width, height = image.size
        images.append(dict(id=image_id, file_name=path.name, width=width, height=height))
        label = root / 'labels' / (path.stem + '.txt')
        # Missing labels represent background, as in the supplied split.
        for line in label.read_text().splitlines() if label.exists() else []:
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f'Expected YOLO bounding boxes in {label}')
            cid = int(fields[0])
            if not 0 <= cid < len(names):
                raise ValueError(f'Unknown class in {label}')
            cx, cy, bw, bh = map(float, fields[1:])
            x, y, w, h = (cx - bw / 2) * width, (cy - bh / 2) * height, bw * width, bh * height
            annotations.append(dict(id=len(annotations) + 1, image_id=image_id, category_id=cid + 1,
                                    bbox=[x, y, w, h], area=w*h, iscrowd=0))
    return dict(images=images, annotations=annotations,
                categories=[dict(id=i+1, name=n) for i, n in enumerate(names)]), paths


def remap_ground_truth(gt, source_names, target_names):
    """Keep annotations whose class is in `target_names`, renumbered by name."""
    if not set(target_names) <= set(source_names):
        raise ValueError('Every active class must exist in the label class list')
    to_target = {i + 1: target_names.index(n) + 1 for i, n in enumerate(source_names) if n in target_names}
    annotations = [dict(a, category_id=to_target[a['category_id']]) for a in gt['annotations']
                   if a['category_id'] in to_target]
    dropped = Counter(source_names[a['category_id'] - 1] for a in gt['annotations']
                      if a['category_id'] not in to_target)
    return dict(images=gt['images'], annotations=annotations,
                categories=[dict(id=i + 1, name=n) for i, n in enumerate(target_names)],
                excluded_label_boxes=dict(dropped))


def score_detections(gt, paths, outputs, names, threshold, safety_classes):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    if set(outputs) != {p.name for p in paths}:
        raise ValueError('Evaluation requires exactly one output for every image, including backgrounds')
    ground = COCO()
    ground.dataset = gt
    results = []
    by_image = defaultdict(list)
    for item in gt['annotations']:
        x, y, w, h = item['bbox']
        by_image[item['image_id']].append((item['category_id'] - 1, [x, y, x+w, y+h]))
    tp, fp, fn = Counter(), Counter(), Counter()
    for k, path in enumerate(paths):
        detections = outputs[path.name]['detections']
        for d in detections:
            x1, y1, x2, y2 = d['box_xyxy']
            results.append(dict(image_id=k, category_id=d['class_id'] + 1, bbox=[x1, y1, x2-x1, y2-y1], score=d['confidence']))
        for cid in range(len(names)):
            truth = [b for c, b in by_image[k] if c == cid]
            used = set()
            predicted = sorted((d for d in detections if d['class_id'] == cid and d['confidence'] >= threshold),
                               key=lambda d: d['confidence'], reverse=True)
            for detection in predicted:
                candidates = [j for j in range(len(truth)) if j not in used]
                best = max(candidates, key=lambda j: iou(detection['box_xyxy'], truth[j]), default=None)
                if best is not None and iou(detection['box_xyxy'], truth[best]) >= .5:
                    tp[cid] += 1
                    used.add(best)
                else:
                    fp[cid] += 1
            fn[cid] += len(truth) - len(used)
    with contextlib.redirect_stdout(io.StringIO()):
        ground.createIndex()
        if results:
            predicted = ground.loadRes(results)
        else:
            predicted = COCO()
            predicted.dataset = dict(images=gt['images'], categories=gt['categories'], annotations=[])
            predicted.createIndex()
        evaluator = COCOeval(ground, predicted, 'bbox')
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    per_class = {}
    for i, name in enumerate(names):
        ap = evaluator.eval['precision'][0, :, i, 0, -1]
        ap_all = evaluator.eval['precision'][:, :, i, 0, -1]   # IoU 0.50:0.95, all areas, 100 detections
        per_class[name] = dict(ap50=float(ap[ap > -1].mean()) if (ap > -1).any() else None,
                               ap50_95=float(ap_all[ap_all > -1].mean()) if (ap_all > -1).any() else None,
                               tp=tp[i], fp=fp[i], fn=fn[i],
                               precision=tp[i] / max(tp[i] + fp[i], 1),
                               recall=tp[i] / (tp[i] + fn[i]) if tp[i] + fn[i] else None,
                               boxes=tp[i] + fn[i])
    precision = sum(tp.values()) / max(sum(tp.values()) + sum(fp.values()), 1)
    recall = sum(tp.values()) / max(sum(tp.values()) + sum(fn.values()), 1)
    safety = [per_class[n]['recall'] for n in safety_classes if per_class[n]['recall'] is not None]
    return dict(evaluator='notebook COCO AP / greedy IoU>=0.5 alerts', images=len(paths), alert_threshold=threshold,
                map50=float(evaluator.stats[1]), map50_95=float(evaluator.stats[0]), precision=precision, recall=recall,
                f1=2*precision*recall/max(precision+recall, 1e-9), safety_recall=float(np.mean(safety)) if safety else None,
                per_class=per_class)
