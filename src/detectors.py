"""Local YOLO and explicit remote RF-DETR adapters."""
from dataclasses import replace
import threading
import time
from urllib.parse import urlparse

from detection import DetectionBatch, make_detection, validate_names


class DetectorError(RuntimeError):
    pass


class YoloDetector:
    def __init__(self, settings):
        self.settings = settings
        if not settings['weights'].is_file():
            raise FileNotFoundError(f"Missing detector weights: {settings['weights']}. Copy the supplied local/yolov8n_v4.pt into models/local/.")
        from ultralytics import YOLO
        self.model = YOLO(str(settings['weights']), task='detect')
        validate_names(self.model.names, settings['classes'])

    def predict(self, frame_bgr):
        spec = self.settings
        start = time.perf_counter()
        result = self.model.predict(frame_bgr, imgsz=spec['input_size'], conf=spec['confidence'],
                                    iou=spec['nms_iou'], max_det=300, rect=False,
                                    device=spec['device'], verbose=False)[0]
        validate_names(result.names, spec['classes'])
        height, width = frame_bgr.shape[:2]
        detections = []
        for row in result.boxes.data.cpu().numpy():
            x1, y1, x2, y2, confidence, cid = row[:6]
            detection = make_detection(cid, confidence, (x1, y1, x2, y2), spec['classes'], width, height)
            if detection is not None and detection.confidence >= spec['class_thresholds'][detection.label]:
                detections.append(detection)
        return DetectionBatch(tuple(sorted(detections, key=lambda d: d.confidence, reverse=True)),
                              1000 * (time.perf_counter() - start), 'yolov8n')


class CloudDetector:
    def __init__(self, settings):
        import requests
        self.settings = settings
        self.session = requests.Session()
        self.lock = threading.Lock()
        endpoint = settings['endpoint'] or ''
        parsed = urlparse(endpoint)
        if parsed.scheme not in ('http', 'https') or not parsed.netloc or parsed.query or parsed.fragment:
            raise ValueError('Cloud mode requires --endpoint http(s)://host:port')
        self.endpoint = endpoint.rstrip('/')
        self.timeout = settings['timeout']

    def health(self):
        """Check the service without sending a frame. Raises DetectorError unless it
        is ready and serves the expected model with the canonical class order."""
        import requests
        try:
            # A plain request, not the session: a timed-out /detect may still be using it.
            response = requests.get(self.endpoint + '/health', timeout=min(self.timeout, 3.0))
            response.raise_for_status()
            info = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise DetectorError(f'Cloud health check failed: {exc}') from exc
        if info.get('status') != 'ready' or info.get('model') != 'rfdetr-nano':
            raise DetectorError('Cloud service is not ready or serves an unexpected model')
        if info.get('classes') != list(self.settings['classes']):
            raise DetectorError('Cloud service class order does not match the canonical IDs')
        return info

    def predict(self, frame_bgr):
        import cv2
        import requests
        if not self.lock.acquire(blocking=False):
            raise DetectorError('Cloud inference already in progress; obsolete frame discarded')
        start = time.perf_counter()
        done = threading.Event()
        received = {}
        def send():
            try:
                ok, encoded = cv2.imencode('.jpg', frame_bgr)
                if not ok:
                    raise DetectorError('Could not encode camera frame')
                received['response'] = self.session.post(
                    self.endpoint + '/detect', data=encoded.tobytes(),
                    headers={'Content-Type': 'image/jpeg'}, timeout=self.timeout,
                    params={} if self.settings['confidence_override'] is None else
                           {'confidence': self.settings['confidence_override']})
            except Exception as exc:
                received['error'] = exc
            finally:
                # A timed-out caller leaves this lock held until its request actually ends.
                # There is no backlog and no second request for an obsolete frame.
                self.lock.release()
                done.set()
        threading.Thread(target=send, daemon=True).start()
        deadline = min(self.timeout, self.settings['max_frame_age'])
        if not done.wait(deadline):
            raise DetectorError('Cloud request deadline exceeded; stale result discarded')
        try:
            if 'error' in received:
                raise received['error']
            response = received['response']
            response.raise_for_status()
            elapsed = time.perf_counter() - start
            if elapsed > self.settings['max_frame_age']:
                raise DetectorError('Cloud response is stale; detections discarded')
            height, width = frame_bgr.shape[:2]
            result = DetectionBatch.from_dict(response.json(), self.settings['classes'], (width, height))
            if result.model != 'rfdetr-nano':
                raise ValueError('Cloud returned an unexpected model')
            detections = tuple(d for d in result.detections if d.confidence >= self.settings['class_thresholds'][d.label])
            return replace(result, detections=detections, round_trip_ms=1000 * elapsed)
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            raise DetectorError(f'Cloud inference unavailable: {exc}') from exc


def create_detector(settings):
    return YoloDetector(settings) if settings['backend'] == 'local' else CloudDetector(settings)
