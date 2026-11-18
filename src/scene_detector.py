"""Supplement the four-class detector with generic, locally detected objects.

The additional COCO model locates common objects; it is not an open-world
detector. Its labels never become trained hazard class IDs or cloud payload IDs.
Every general detection keeps class_id -1. COCO people are labelled People so the
alert gate treats them like people; everything else is labelled Object. Any
general box overlapping a trained box at IoU >= 0.5 is dropped as a duplicate.

The general model is optional at run time. If it cannot be loaded (for example
the first-run download failed) or a prediction fails, the trained detections are
still returned and `general_error` says why the extra detections are missing.
Errors from the trained detector are never swallowed.
"""
import logging
from dataclasses import replace
import time

from config import resolve_path
from detection import make_detection
from detectors import create_detector
from golden import iou

COCO_PERSON = 0
_LOAD = object()
logger = logging.getLogger(__name__)


def load_general_model():
    from ultralytics import YOLO
    weights = resolve_path('models/general/yolov8n.pt')
    weights.parent.mkdir(parents=True, exist_ok=True)
    # Ultralytics downloads the pretrained weights on first use.
    model = YOLO(str(weights), task='detect')
    if len(model.names) != 80 or model.names[0] != 'person' or model.names[56] != 'chair':
        raise ValueError('General detector requires the standard 80-class COCO checkpoint')
    return model


class SceneDetector:
    def __init__(self, settings, general=_LOAD):
        """`general` shares an already loaded general model between detectors.
        Pass None to run without one."""
        self.trained = create_detector(settings)
        self.general_error = None
        if general is _LOAD:
            try:
                general = load_general_model()
            except Exception as exc:
                logger.error('General model unavailable: %s', exc)
                general, self.general_error = None, f'General model unavailable: {exc}'
        elif general is None:
            self.general_error = 'General model unavailable'
        self.general = general
        self.device = settings['device']

    def health(self):
        """Check the trained backend without sending a frame (cloud only)."""
        check = getattr(self.trained, 'health', None)
        return check() if check is not None else {'status': 'ready'}

    def predict(self, frame_bgr):
        batch = self.trained.predict(frame_bgr)
        if self.general is None:
            return batch
        start = time.perf_counter()
        try:
            result = self.general.predict(frame_bgr, imgsz=640, conf=.35, iou=.7,
                                          max_det=100, rect=False, device=self.device,
                                          verbose=False)[0]
            objects = self._objects(result, frame_bgr.shape, batch.detections)
        except Exception as exc:
            # Trained detections still count; only the extra objects are missing.
            logger.error('General model prediction failed: %s', exc)
            self.general_error = f'General model failed: {exc}'
            return replace(batch, latency_ms=batch.latency_ms + 1000 * (time.perf_counter() - start))
        self.general_error = None
        detections = tuple(sorted(batch.detections + tuple(objects),
                                  key=lambda detection: detection.confidence, reverse=True))
        return replace(batch, detections=detections,
                       latency_ms=batch.latency_ms + 1000 * (time.perf_counter() - start),
                       model=batch.model + ' + general-yolov8n')

    @staticmethod
    def _objects(result, shape, trained):
        height, width = shape[:2]
        objects = []
        for row in result.boxes.data.cpu().numpy():
            x1, y1, x2, y2, confidence, coco_id = row[:6]
            # A COCO person the trained model missed is still a person: label it
            # People for gating and audio, but keep class_id -1 (source: general).
            label = 'People' if int(coco_id) == COCO_PERSON else 'Object'
            detection = make_detection(0, confidence, (x1, y1, x2, y2), [label], width, height)
            if detection is None or any(iou(detection.box_xyxy, named.box_xyxy) >= .5
                                        for named in trained):
                continue
            objects.append(replace(detection, class_id=-1))
        return objects
