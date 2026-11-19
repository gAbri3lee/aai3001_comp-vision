"""YOLOv8n ONNX inference with only numpy, Pillow and onnxruntime (no torch, no ultralytics).

Mirrors the Ultralytics predict pipeline for a fixed-shape 640 export: letterbox
(pad value 114, centred), RGB/255 input, class-aware NMS, then the per-class
thresholds from the config. Boxes come back in pixels of the ORIGINAL image.
"""
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
_session = None
_config = None


def load_config():
    global _config
    if _config is None:
        _config = json.loads((HERE / 'config.json').read_text())
    return _config


def get_session():
    """One ONNX session per warm function instance; cold starts pay the load once."""
    global _session
    if _session is None:
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.log_severity_level = 3
        candidate = ort.InferenceSession(str(HERE / 'model' / load_config()['model_file']),
                                        sess_options=options, providers=['CPUExecutionProvider'])
        if candidate.get_outputs()[0].shape[1] != 4 + len(load_config()["classes"]):
            raise ValueError("ONNX output class count disagrees with deployment config")
        _session = candidate
    return _session


def letterbox(image, size):
    """Scale to fit size x size keeping aspect, pad with 114 (Ultralytics LetterBox, auto=False)."""
    width, height = image.size
    ratio = min(size / height, size / width)
    new_w, new_h = round(width * ratio), round(height * ratio)
    if (new_w, new_h) != (width, height):
        # Plain 2x2 bilinear sampling like cv2.INTER_LINEAR (Image.resize would
        # low-pass filter on downscale and drift from the trained pipeline).
        image = image.transform((new_w, new_h), Image.AFFINE,
                                (width / new_w, 0, 0, 0, height / new_h, 0), Image.BILINEAR)
    pad_w, pad_h = (size - new_w) / 2, (size - new_h) / 2
    left, top = round(pad_w - 0.1), round(pad_h - 0.1)
    canvas = Image.new('RGB', (size, size), (114, 114, 114))
    canvas.paste(image, (left, top))
    return canvas, ratio, left, top


def nms(boxes, scores, iou_threshold):
    order = scores.argsort()[::-1]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        if not rest.size:
            break
        w = np.maximum(0., np.minimum(boxes[i, 2], boxes[rest, 2]) - np.maximum(boxes[i, 0], boxes[rest, 0]))
        h = np.maximum(0., np.minimum(boxes[i, 3], boxes[rest, 3]) - np.maximum(boxes[i, 1], boxes[rest, 1]))
        inter = w * h
        iou = inter / np.maximum(areas[i] + areas[rest] - inter, 1e-9)
        order = rest[iou <= iou_threshold]
    return keep


def detect(image, confidence=None):
    """Run the detector on an RGB PIL image; returns the shared output format."""
    config = load_config()
    names, thresholds = config['classes'], config['class_thresholds']
    base = min(thresholds.values()) if confidence is None else confidence
    size = config['input_size']
    width, height = image.size
    started = time.perf_counter()

    padded, ratio, left, top = letterbox(image, size)
    tensor = (np.asarray(padded, dtype=np.float32) / 255.).transpose(2, 0, 1)[None]
    session = get_session()
    output = session.run(None, {session.get_inputs()[0].name: tensor})[0][0].T  # (anchors, 4 + classes)

    scores = output[:, 4:]
    class_ids = scores.argmax(1)
    confidences = scores.max(1)
    keep = confidences > base
    output, class_ids, confidences = output[keep], class_ids[keep], confidences[keep]
    cx, cy, w, h = output[:, 0], output[:, 1], output[:, 2], output[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left) / ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top) / ratio
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)

    detections = []
    for cid in np.unique(class_ids):  # class-aware NMS
        idx = np.flatnonzero(class_ids == cid)
        for k in nms(boxes[idx], confidences[idx], config['nms_iou']):
            j = idx[k]
            label = names[int(cid)]
            floor = base if confidence is not None else thresholds[label]
            x1, y1, x2, y2 = (float(v) for v in boxes[j])
            if confidences[j] >= floor and x2 > x1 and y2 > y1:
                detections.append({'class_id': int(cid), 'class': label,
                                   'confidence': float(confidences[j]), 'box_xyxy': [x1, y1, x2, y2]})
    detections.sort(key=lambda d: d['confidence'], reverse=True)
    return {'detections': detections[:300], 'latency_ms': (time.perf_counter() - started) * 1000,
            'model': 'yolov8n-onnx', 'image_size': [width, height]}
