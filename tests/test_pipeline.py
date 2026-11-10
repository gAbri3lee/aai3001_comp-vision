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
        for name in ['depth', 'tts']:
            setattr(args, 'no_' + name, True)
        args.source = 0

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
                # Box height 40 of 60 px, centred: a near stair under the alert gate.
                self.assertEqual(status['alert'], 'Warning, Stair ahead')
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
                # Local inference must not speak a frame the UI has already expired.
                from types import SimpleNamespace
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
        state.mode, state.detectors = 'local', {'local': object()}
        state.backend_settings = {'local': {'model': 'yolov8n'}}
        state.fallback_active, state.last_error, state._hold_until = False, '', None
        state.capture, state.camera_missing, state.source_ended, state.pipeline_stopped = None, False, False, False
        state.args.no_tts, state.speaker, state.speech_error = True, None, None
        state.paused, state.muted, state.speech_rate, state.last_spoken = False, False, 180, None
        state.status = {'fps': 8., 'alert': 'Stair ahead', 'detections': [{'class_id': 3}],
                        'error': '', 'latency_ms': 10., 'round_trip_ms': 20.}
        status = state.get_status()
        self.assertEqual(status['detections'], [])
        self.assertEqual(status['alert'], '')
        self.assertEqual(status['alerts'], [])
        self.assertIn('stale', status['error'])
        self.assertIsNone(state.get_jpeg())

    def test_unknown_object_reaches_gui_and_audio_without_query_features(self):
        from types import SimpleNamespace
        parser = argparse.ArgumentParser()
        add_detector_arguments(parser)
        args = parser.parse_args([])
        args.no_depth = True
        args.no_tts = False
        args.source = 0

        class Capture:
            finished = False
            def __init__(self, source): pass
            def read(self):
                time.sleep(.01)
                return True, np.zeros((60, 100, 3), dtype=np.uint8), time.monotonic()
            def release(self): pass

        for include_named in (False, True):
            with self.subTest(include_named=include_named):
                spoken = []
                class Predictor:
                    def predict(self, frame):
                        detections = (Detection(-1, 'Object', .8, (40, 10, 60, 50)),)
                        if include_named:
                            detections += (Detection(3, 'Stair', .8, (0, 10, 20, 50)),)
                        return DetectionBatch(detections, 10, 'scene')

                with patch('webapp.create_detector', return_value=Predictor()), \
                     patch('webapp.LatestFrameCapture', Capture), \
                     patch.dict('sys.modules', {'tts': SimpleNamespace(Speaker=lambda: SimpleNamespace(say=spoken.append))}):
                    state = PipelineState(args)
                    try:
                        deadline = time.monotonic() + 2
                        while 'Object detected ahead' not in spoken and time.monotonic() < deadline:
                            time.sleep(.01)
                        client = create_app(state).test_client()
                        status = client.get('/status').json
                        self.assertIn('Object detected ahead', spoken)
                        self.assertIn('Object detected ahead', status['alerts'])
                        self.assertEqual(status['detections'][0]['class'], 'Object')
                        if include_named:
                            self.assertEqual(status['alert'], 'Warning, Stair on your left')
                            self.assertEqual(spoken[0], 'Warning, Stair on your left')
                        else:
                            self.assertEqual(status['alert'], 'Object detected ahead')
                            self.assertEqual(spoken, ['Object detected ahead'])
                        page = client.get('/').get_data(as_text=True)
                        self.assertNotIn('Scan text', page)
                        self.assertNotIn('Identify item', page)
                        self.assertEqual(client.post('/scan_text').status_code, 404)
                        self.assertEqual(client.post('/identify_item').status_code, 404)
                    finally:
                        state.stop()
