"""Backend-independent detections in original-frame pixel coordinates."""
from dataclasses import dataclass
import math
from typing import Protocol


@dataclass(frozen=True)
class Detection:
    class_id: int
    label: str
    confidence: float
    box_xyxy: tuple[float, float, float, float]

    def to_dict(self):
        return {'class_id': self.class_id, 'class': self.label,
                'confidence': self.confidence, 'box_xyxy': list(self.box_xyxy)}


@dataclass(frozen=True)
class DetectionBatch:
    detections: tuple[Detection, ...]
    latency_ms: float
    model: str
    round_trip_ms: float | None = None

    def to_dict(self):
        result = {'detections': [d.to_dict() for d in self.detections],
                  'latency_ms': self.latency_ms, 'model': self.model}
        if self.round_trip_ms is not None:
            result['round_trip_ms'] = self.round_trip_ms
        return result

    @classmethod
    def from_dict(cls, payload, names, frame_size):
        width, height = frame_size
        detections = []
        for item in payload['detections']:
            cid = item['class_id']
            if type(cid) is not int or not 0 <= cid < len(names) or item['class'] != names[cid]:
                raise ValueError('Detection class ID/name does not match the configured checkpoint')
            confidence = float(item['confidence'])
            box = tuple(float(x) for x in item['box_xyxy'])
            if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError('Invalid confidence')
            if len(box) != 4 or not all(math.isfinite(v) for v in box):
                raise ValueError('Invalid box coordinates')
            x1, y1, x2, y2 = box
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ValueError('Box must be within the original image and have positive area')
            detections.append(Detection(cid, names[cid], confidence, box))
        latency = float(payload['latency_ms'])
        if not math.isfinite(latency) or latency < 0 or not isinstance(payload['model'], str) or not payload['model']:
            raise ValueError('Invalid inference metadata')
        return cls(tuple(sorted(detections, key=lambda d: d.confidence, reverse=True)), latency, payload['model'])


class Detector(Protocol):
    def predict(self, frame_bgr) -> DetectionBatch: ...


def validate_names(actual, expected):
    if isinstance(actual, dict):
        if set(actual) != set(range(len(expected))):
            raise ValueError('Checkpoint has unexpected class IDs')
        actual = [actual[i] for i in range(len(expected))]
    if list(actual) != list(expected):
        raise ValueError(f'Checkpoint classes {actual!r} do not match {expected!r}')


def make_detection(cid, confidence, box, names, width, height):
    """Clip model edge overshoot; discard degenerate boxes, never remap class IDs."""
    cid = int(cid)
    if not 0 <= cid < len(names):
        raise ValueError(f'Checkpoint emitted unknown class ID {cid}')
    if not all(math.isfinite(float(v)) for v in (*box, confidence)):
        raise ValueError('Model emitted nonfinite coordinates/confidence')
    x1, y1, x2, y2 = box
    box = (max(0., min(float(x1), width)), max(0., min(float(y1), height)),
           max(0., min(float(x2), width)), max(0., min(float(y2), height)))
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return Detection(cid, names[cid], float(confidence), box)


def annotate(frame_bgr, detections):
    import cv2
    frame = frame_bgr.copy()
    for detection in detections:
        x1, y1, x2, y2 = map(int, detection.box_xyxy)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (70, 200, 120), 2)
        cv2.putText(frame, f'{detection.label} {detection.confidence:.2f}',
                    (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, .55, (70, 200, 120), 2)
    return frame
