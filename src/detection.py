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


# BGR colours by alert band; the spoken hazard gets its own colour.
BAND_COLOURS = {'near': (60, 60, 255), 'mid-close': (0, 184, 255), 'mid': (70, 200, 120), None: (70, 200, 120)}
SPOKEN_COLOUR = (255, 80, 255)
IGNORED_COLOUR = (130, 130, 130)


def annotate_hazards(frame_bgr, hazards, selected=None, debug=False):
    """Draw gated hazards by band and the spoken hazard in a distinct colour.

    Ignored (far, side, low-confidence) hazards are drawn thin and grey with their
    reason only when debug is true; otherwise they are not drawn at all.
    """
    import cv2
    frame = frame_bgr.copy()
    # Draw ignored first and the spoken hazard last, so the highlight stays on top.
    order = sorted(hazards, key=lambda h: (h.gated, h is selected))
    for hazard in order:
        if not hazard.gated and not debug:
            continue
        x1, y1, x2, y2 = map(int, hazard.xyxy)
        name = hazard.label + (' (general)' if hazard.source == 'general' and hazard.label != 'Object' else '')
        if not hazard.gated:
            colour, thickness = IGNORED_COLOUR, 1
            text = f'{name} {hazard.confidence:.2f} ignored: {hazard.reason}'
        else:
            spoken = hazard is selected
            colour = SPOKEN_COLOUR if spoken else BAND_COLOURS.get(hazard.band, BAND_COLOURS[None])
            thickness = 4 if spoken else 2
            text = f'{name} {hazard.confidence:.2f}' + (f' {hazard.band}' if hazard.band else '')
            if spoken:
                text = 'SPOKEN ' + text
        if debug and hazard.size is not None:
            text += f' h={hazard.size:.2f}'
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thickness)
        # Boxes touching the top edge get their label inside the box.
        text_y = y1 - 6 if y1 >= 24 else y1 + 18
        cv2.putText(frame, text, (x1 + 2, text_y), cv2.FONT_HERSHEY_SIMPLEX, .5, colour, 1 if not hazard.gated else 2)
    return frame


def annotate_recognition(frame_bgr, hazards):
    """Recognition view: every trained detection with its class and confidence.

    There is no alert gate here, so far and side detections are drawn like near
    ones, in the same band colours as the live view (far uses the default colour).
    General-model detections are not trained classes and are not drawn.
    """
    import cv2
    frame = frame_bgr.copy()
    trained = sorted((h for h in hazards if h.source == 'trained'), key=lambda h: h.confidence)
    for hazard in trained:  # the most confident box is drawn last, on top
        x1, y1, x2, y2 = map(int, hazard.xyxy)
        colour = BAND_COLOURS.get(hazard.band, BAND_COLOURS[None])
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 3)
        text_y = y1 - 6 if y1 >= 24 else y1 + 20
        cv2.putText(frame, f'{hazard.label} {hazard.confidence:.2f}', (x1 + 2, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, .6, colour, 2)
    return frame


def annotate(frame_bgr, detections):
    import cv2
    frame = frame_bgr.copy()
    for detection in detections:
        x1, y1, x2, y2 = map(int, detection.box_xyxy)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (70, 200, 120), 2)
        cv2.putText(frame, f'{detection.label} {detection.confidence:.2f}',
                    (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, .55, (70, 200, 120), 2)
    return frame
