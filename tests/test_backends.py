import argparse
import io
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from werkzeug.serving import make_server

from config import add_detector_arguments, detector_settings, load_config
from detection import Detection, DetectionBatch
from detectors import YoloDetector, CloudDetector, DetectorError
from cloud_service import create_app
from capture import LatestFrameCapture


def settings(*args):
    parser = argparse.ArgumentParser()
    add_detector_arguments(parser)
    return detector_settings(parser.parse_args(list(args)))


class BackendTests(unittest.TestCase):
    def test_missing_weights_are_actionable(self):
        spec = settings('--model', 'does-not-exist.pt')
        with self.assertRaisesRegex(FileNotFoundError, 'weights'):
            YoloDetector(spec)

    def test_cloud_service_decodes_rgb_and_preserves_original_coordinates(self):
        class Predictor:
            def predict(self, image):
                self.image = image
                return DetectionBatch((Detection(2, 'Stair', .8, (1, 2, 20, 30)),), 10, 'rfdetr-nano')
        predictor = Predictor()
        app = create_app(predictor, load_config())
        client = app.test_client()
        self.assertEqual(client.get('/health').status_code, 200)
        self.assertEqual(client.post('/detect', data=b'not-jpeg', content_type='image/jpeg').status_code, 400)
        self.assertEqual(client.post('/detect', data=b'no', content_type='text/plain').status_code, 415)
        buffer = io.BytesIO()
        Image.new('RGB', (80, 40), (230, 10, 5)).save(buffer, format='JPEG')
        response = client.post('/detect', data=buffer.getvalue(), content_type='image/jpeg')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['detections'][0]['box_xyxy'], [1, 2, 20, 30])
        self.assertEqual(predictor.image.mode, 'RGB')
        self.assertEqual(predictor.image.size, (80, 40))
        r, g, b = predictor.image.getpixel((0, 0))
        self.assertGreater(r, b + 100)
        with patch.object(predictor, 'predict', side_effect=RuntimeError('failed')):
            self.assertEqual(client.post('/detect', data=buffer.getvalue(), content_type='image/jpeg').status_code, 503)

    def test_cloud_client_real_http_round_trip_and_threshold(self):
        class Predictor:
            def predict(self, image, confidence=None):
                threshold = .4 if confidence is None else confidence
                detections = (Detection(2, 'Stair', .8, (0, 0, 20, 20)),
                                       Detection(1, 'People', .3, (20, 0, 40, 20)))
                return DetectionBatch(tuple(d for d in detections if d.confidence >= threshold), 2, 'rfdetr-nano')
        server = make_server('127.0.0.1', 0, create_app(Predictor(), load_config()))
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            spec = settings('--backend', 'cloud', '--endpoint', f'http://127.0.0.1:{server.server_port}')
            detector = CloudDetector(spec)
            result = detector.predict(np.zeros((40, 80, 3), dtype=np.uint8))
            self.assertEqual([d.label for d in result.detections], ['Stair'])
            self.assertIsNotNone(result.round_trip_ms)
            self.assertEqual(result.latency_ms, 2)
            lower_spec = settings('--backend', 'cloud', '--endpoint', f'http://127.0.0.1:{server.server_port}', '--conf', '.2')
            result = CloudDetector(lower_spec).predict(np.zeros((40, 80, 3), dtype=np.uint8))
            self.assertEqual([d.label for d in result.detections], ['Stair', 'People'])
        finally:
            server.shutdown()
            worker.join()
            server.server_close()

    def test_client_rejects_late_response_and_does_not_cache_previous_detections(self):
        import requests
        from requests import Response
        spec = settings('--backend', 'cloud', '--endpoint', 'http://localhost:9000', '--max-frame-age', '.01')
        detector = CloudDetector(spec)
        response = Response()
        response.status_code = 200
        response._content = b'{"detections":[],"latency_ms":1,"model":"rfdetr-nano"}'
        def slow(*args, **kwargs):
            time.sleep(.02)
            return response
        with patch.object(detector.session, 'post', side_effect=slow), self.assertRaisesRegex(DetectorError, 'stale|deadline'):
            detector.predict(np.zeros((40, 80, 3), dtype=np.uint8))
        with patch.object(detector.session, 'post', side_effect=requests.Timeout), self.assertRaises(DetectorError):
            detector.predict(np.zeros((40, 80, 3), dtype=np.uint8))

    def test_total_deadline_does_not_start_a_second_request(self):
        import requests
        release = threading.Event()
        entered = threading.Event()
        spec = settings('--backend', 'cloud', '--endpoint', 'http://localhost:9000', '--timeout', '.03')
        detector = CloudDetector(spec)
        def stalled(*args, **kwargs):
            entered.set()
            release.wait(1)
            raise requests.Timeout()
        frame = np.zeros((40, 80, 3), dtype=np.uint8)
        with patch.object(detector.session, 'post', side_effect=stalled):
            start = time.monotonic()
            try:
                with self.assertRaisesRegex(DetectorError, 'deadline'):
                    detector.predict(frame)
                self.assertLess(time.monotonic() - start, .2)
                self.assertTrue(entered.is_set())
                with self.assertRaisesRegex(DetectorError, 'progress'):
                    detector.predict(frame)
            finally:
                release.set()

    def test_latest_capture_does_not_queue_old_frames(self):
        class Source:
            def __init__(self): self.i = 0
            def isOpened(self): return True
            def get(self, prop): return 0  # unknown FPS: no pacing
            def read(self):
                self.i += 1
                if self.i > 4: return False, None
                return True, np.full((2, 2, 3), self.i, dtype=np.uint8)
            def release(self): pass
        with patch('cv2.VideoCapture', return_value=Source()):
            capture = LatestFrameCapture('video.mp4')
            capture.thread.join(timeout=1)
            ok, frame, captured_at = capture.read()
            self.assertTrue(ok)
            self.assertEqual(frame[0, 0, 0], 4)
            self.assertFalse(capture.read()[0])
            capture.release()
        with patch('cv2.VideoCapture') as video:
            video.return_value.isOpened.return_value = False
            with self.assertRaisesRegex(RuntimeError, 'camera'):
                LatestFrameCapture(999)

    def test_native_rfdetr_skips_background_slot(self):
        from types import SimpleNamespace
        from rfdetr_detector import RFDetrDetector
        config = load_config()
        config['cloud']['class_thresholds'] = dict.fromkeys(config['classes'], .001)
        detector = RFDetrDetector.__new__(RFDetrDetector)
        detector.config, detector.spec = config, config['cloud']
        class Predictor:
            def predict(self, *args, **kwargs):
                return SimpleNamespace(class_id=np.array([2, 4]), confidence=np.array([.8, .002]),
                                       xyxy=np.array([[0, 0, 20, 20], [0, 0, 30, 30]]))
        detector.model = Predictor()
        result = detector.predict(Image.new('RGB', (40, 40)))
        self.assertEqual([d.label for d in result.detections], ['Stair'])
