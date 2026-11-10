"""Supplement the five-class detector with generic, locally detected objects.

The additional COCO model locates common objects; it is not an open-world
detector. Its labels never become trained hazard class IDs or cloud payload IDs.
"""
from dataclasses import replace
import time

from config import resolve_path
from detection import make_detection
from detectors import create_detector
from golden import iou


def load_general_model():
    from ultralytics import YOLO
    weights = resolve_path('models/general/yolov8n.pt')
    weights.parent.mkdir(parents=True, exist_ok=True)
    # Ultralytics downloads the pretrained weights on first use.
    return YOLO(str(weights), task='detect')


class SceneDetector:
    def __init__(self, settings):
        self.trained = create_detector(settings)
        self.general = load_general_model()
        self.device = settings['device']

    def predict(self, frame_bgr):
        batch = self.trained.predict(frame_bgr)
        start = time.perf_counter()
        result = self.general.predict(frame_bgr, imgsz=640, conf=.35, iou=.7,
                                      max_det=100, rect=False, device=self.device,
                                      verbose=False)[0]
        height, width = frame_bgr.shape[:2]
        objects = []
        for row in result.boxes.data.cpu().numpy():
            x1, y1, x2, y2, confidence = row[:5]
            detection = make_detection(0, confidence, (x1, y1, x2, y2), ['Object'], width, height)
            if detection is None or any(iou(detection.box_xyxy, named.box_xyxy) >= .5
                                        for named in batch.detections):
                continue
            objects.append(replace(detection, class_id=-1))
        detections = tuple(sorted(batch.detections + tuple(objects),
                                  key=lambda detection: detection.confidence, reverse=True))
        return replace(batch, detections=detections,
                       latency_ms=batch.latency_ms + 1000 * (time.perf_counter() - start),
                       model=batch.model + ' + general-yolov8n')
