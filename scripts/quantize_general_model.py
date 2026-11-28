"""Quantise the general COCO YOLOv8n (the app's extra "Object"/"People" detector) to INT8 and check it.

    python scripts/quantize_general_model.py                     # export, calibrate, quantise, compare
    python scripts/quantize_general_model.py --skip-quantize --people-split data/v4_four_class_clean/test

1. Export models/general/yolov8n.pt to FP32 ONNX (same settings as scripts/export_detector.py).
2. Quantise it with the hazard model's procedure (scripts/quantize_detector.py): static QDQ INT8, per-channel
   weights, calibration on app-preprocessed frames, the detection head's decoding kept in FP32.
3. Compare INT8 against the original .pt exactly as the app calls it (src/scene_detector.py: Ultralytics
   predict, 640 px, conf 0.35, IoU 0.7): same-class IoU >= 0.5 agreement and median latency. With --people-split,
   also score its person boxes against that split's People labels, for both models (the only COCO class those
   labels cover; Object boxes cannot be scored).

This script never edits visionaid_config.json.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from config import resolve_path
from golden import iou
from quantize_detector import IMAGE_SUFFIXES, agreement, calibration_frames, quantize

PT = 'models/general/yolov8n.pt'
FP32 = 'models/general/yolov8n.onnx'
INT8 = 'models/general/yolov8n_int8.onnx'
PEOPLE = 1          # class index of People in the four-class labels


def predict_all(weights, images):
    """Detections at the app's settings and per-image latency (second call per image, after a warm-up call)."""
    import cv2
    from ultralytics import YOLO
    model = YOLO(str(weights), task='detect')
    outputs, times = {}, []
    for path in images:
        frame = cv2.imread(str(path))
        kwargs = dict(imgsz=640, conf=.35, iou=.7, max_det=100, rect=False, device='cpu', verbose=False)
        model.predict(frame, **kwargs)
        start = time.perf_counter()
        boxes = model.predict(frame, **kwargs)[0].boxes.data.cpu().numpy()
        times.append(1000 * (time.perf_counter() - start))
        outputs[path.name] = {'image_size': [frame.shape[1], frame.shape[0]],
                              'detections': [dict(class_id=int(b[5]), confidence=float(b[4]),
                                                  box_xyxy=[float(v) for v in b[:4]]) for b in boxes]}
    return outputs, times


def people_scores(outputs, split):
    """Greedy IoU >= 0.5 matching of COCO person boxes to the split's People labels."""
    from PIL import Image
    tp = fp = fn = 0
    for name, item in outputs.items():
        label = split / 'labels' / (Path(name).stem + '.txt')
        with Image.open(split / 'images' / name) as image:
            w, h = image.size
        truth = []
        for line in label.read_text().splitlines() if label.exists() else []:
            fields = line.split()
            if fields and int(fields[0]) == PEOPLE:
                cx, cy, bw, bh = map(float, fields[1:5])
                truth.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
        used = set()
        for d in sorted((d for d in item['detections'] if d['class_id'] == 0), key=lambda d: -d['confidence']):
            best = max((j for j in range(len(truth)) if j not in used),
                       key=lambda j: iou(d['box_xyxy'], truth[j]), default=None)
            if best is not None and iou(d['box_xyxy'], truth[best]) >= .5:
                used.add(best)
                tp += 1
            else:
                fp += 1
        fn += len(truth) - len(used)
    return dict(tp=tp, fp=fp, fn=fn, precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--calib', nargs='+', default=None,
                        help='Calibration image folders or videos (default: the cleaned train split if present, else demo/)')
    parser.add_argument('--calib-count', type=int, default=200)
    parser.add_argument('--eval-images', default='golden/images')
    parser.add_argument('--people-split', default=None, help='Split folder (images/, labels/) to score person boxes')
    parser.add_argument('--report', default='artifacts/general-int8-quantization.json')
    parser.add_argument('--skip-quantize', action='store_true')
    args = parser.parse_args()
    pt, fp32, int8 = resolve_path(PT), resolve_path(FP32), resolve_path(INT8)
    if not pt.is_file():
        parser.error(f'Missing {pt}; start the app once to download it')
    if not args.skip_quantize:
        if not fp32.is_file():
            from ultralytics import YOLO
            YOLO(str(pt)).export(format='onnx', imgsz=640, batch=1, dynamic=False, half=False, simplify=False, opset=17)
        train = resolve_path('data/v4_four_class_clean/train/images')
        sources = args.calib or ([train] if train.is_dir() else ['demo'])
        frames = calibration_frames(sources, args.calib_count)
        if not frames:
            parser.error(f'No calibration images found in {sources}')
        print(f'calibrating on {len(frames)} frames from {[str(s) for s in sources]}', flush=True)
        print(f'wrote {int8} ({quantize(fp32, int8, frames, 640)} detection-head nodes kept in FP32)', flush=True)

    images = sorted(p for p in resolve_path(args.eval_images).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    ref, t_pt = predict_all(pt, images)
    out, t_int8 = predict_all(int8, images)
    report = dict(reference=PT, int8=INT8, eval_images=len(images),
                  size_mb={'pt': round(pt.stat().st_size / 2**20, 2), 'int8': round(int8.stat().st_size / 2**20, 2)},
                  median_ms={'pt': round(statistics.median(t_pt), 1), 'int8': round(statistics.median(t_int8), 1)},
                  speedup=round(statistics.median(t_pt) / statistics.median(t_int8), 2),
                  agreement_vs_pt_iou50=agreement(ref, out),
                  max_confidence_drop=None)
    drops = []
    for name, item in ref.items():
        for d in item['detections']:
            same = [o for o in out[name]['detections'] if o['class_id'] == d['class_id']
                    and iou(o['box_xyxy'], d['box_xyxy']) >= .5]
            if same:
                drops.append(d['confidence'] - max(o['confidence'] for o in same))
    report['max_confidence_drop'] = round(max(drops), 3) if drops else None
    if args.people_split:
        split = resolve_path(args.people_split)
        split_images = sorted(p for p in (split / 'images').iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        print(f'scoring person boxes on {len(split_images)} images', flush=True)
        report['people_vs_labels'] = dict(split=args.people_split, images=len(split_images),
                                          pt=people_scores(predict_all(pt, split_images)[0], split),
                                          int8=people_scores(predict_all(int8, split_images)[0], split))
    print(json.dumps(report, indent=2))
    path = resolve_path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + '\n')
    print('report:', path)


if __name__ == '__main__':
    main()
