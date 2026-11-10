import argparse
import time
import unittest
from unittest.mock import patch
import numpy as np
from config import add_detector_arguments
from detection import Detection, DetectionBatch
from detectors import DetectorError
from webapp import PipelineState, create_app


class PipelineTests(unittest.TestCase):
    def test_no_depth_alerts_and_inference_failure_clears_status(self):
        parser = argparse.ArgumentParser()
        add_detector_arguments(parser)
        args = parser.parse_args([])
        for name in ['depth', 'tts', 'ocr', 'voice', 'product_id']:
            setattr(args, 'no_' + name, True)
        args.source, args.mic_index = 0, None

        class Capture:
            finished = False
            enabled = True
            age = 0.
            def __init__(self, source): pass
            def read(self):
                time.sleep(.01)
                if not self.enabled: return False, None, None
                return True, np.zeros((60, 100, 3), dtype=np.uint8), time.monotonic() - self.age
            def release(self): pass

        class Predictor:
            failed = False
            def predict(self, frame):
                if self.failed: raise DetectorError('Cloud unavailable')
                return DetectionBatch((Detection(3, 'Stair', .8, (40, 10, 60, 50)),), 10, 'yolov8n')

        detector = Predictor()
        with patch('webapp.create_detector', return_value=detector), patch('webapp.LatestFrameCapture', Capture):
            state = PipelineState(args)
            try:
                deadline = time.monotonic() + 2
                while not state.get_status()['alert'] and time.monotonic() < deadline: time.sleep(.01)
                client = create_app(state).test_client()
                status = client.get('/status').json
                self.assertIn('Stair ahead', status['alert'])
                self.assertIn('distance unavailable', status['alert'])
                self.assertEqual(status['detections'][0]['class_id'], 3)
                self.assertIsNotNone(state.get_jpeg())
                self.assertEqual(client.get('/').status_code, 200)
                detector.failed = True
                deadline = time.monotonic() + 2
                while not state.get_status()['error'] and time.monotonic() < deadline: time.sleep(.01)
                status = client.get('/status').json
                self.assertEqual(status['alert'], '')
                self.assertEqual(status['detections'], [])
                self.assertIn('unavailable', status['error'])
                self.assertIsNone(status['latency_ms'])
                detector.failed = False
                class BrokenOcr:
                    def read(self, frame): raise RuntimeError('OCR failed')
                state.ocr_reader = BrokenOcr()
                state.request_ocr()
                deadline = time.monotonic() + 1
                while 'unavailable' not in state.get_status().get('ocr_text', '') and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertTrue(state._thread.is_alive())
                self.assertIn('unavailable', state.get_status()['ocr_text'])
                class BrokenProduct:
                    def identify(self, frame, topk=1): raise RuntimeError('Item ID failed')
                state.product_identifier = BrokenProduct()
                state.request_product_id()
                deadline = time.monotonic() + 1
                while 'unavailable' not in state.get_status().get('product_text', '') and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertTrue(state._thread.is_alive())
                self.assertIn('unavailable', state.get_status()['product_text'])
                # Local inference must not speak a frame the UI has already expired.
                from types import SimpleNamespace
                state.ocr_reader = state.product_identifier = None
                state.capture.age = 10.
                deadline = time.monotonic() + 1
                while 'stale' not in state.get_status()['error'] and time.monotonic() < deadline:
                    time.sleep(.01)
                spoken = []
                state.speaker = SimpleNamespace(say=spoken.append)
                time.sleep(.05)
                self.assertEqual(spoken, [])
            finally:
                state.stop()

    def test_status_and_video_expire_while_inference_stalls(self):
        import threading
        state = PipelineState.__new__(PipelineState)
        state._lock = threading.Lock()
        state._latest_jpeg = b'old annotated frame'
        state._frame_captured_at = time.monotonic() - 5
        state.args = argparse.Namespace(max_frame_age=.01)
        state.status = {'fps': 8., 'alert': 'Stair ahead', 'detections': [{'class_id': 3}],
                        'error': '', 'latency_ms': 10., 'round_trip_ms': 20.}
        status = state.get_status()
        self.assertEqual(status['detections'], [])
        self.assertEqual(status['alert'], '')
        self.assertIn('stale', status['error'])
        self.assertIsNone(state.get_jpeg())

    def test_unavailable_microphone_does_not_prevent_hazard_pipeline_startup(self):
        from types import SimpleNamespace
        import threading
        parser = argparse.ArgumentParser()
        add_detector_arguments(parser)
        args = parser.parse_args([])
        args.no_depth = args.no_tts = args.no_product_id = True
        args.no_ocr = args.no_voice = False
        args.source, args.mic_index = 0, None
        class Capture:
            finished = False
            def __init__(self, source): pass
            def read(self):
                time.sleep(.01)
                return False, None, None
            def release(self): pass
        def unavailable_voice(**kwargs):
            raise OSError('No microphone or PyAudio installed')
        with patch('webapp.create_detector', return_value=object()), patch('webapp.LatestFrameCapture', Capture), \
             patch.dict('sys.modules', {'voice_trigger': SimpleNamespace(VoiceTrigger=unavailable_voice),
                                       'ocr': SimpleNamespace(OcrReader=lambda: object())}):
            state = PipelineState(args)
            try:
                self.assertTrue(state._thread.is_alive())
                self.assertFalse(state.get_status()['voice_available'])
                self.assertIsNone(state.voice_trigger)
            finally:
                state.stop()
