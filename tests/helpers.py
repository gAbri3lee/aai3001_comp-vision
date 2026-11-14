"""Shared fakes for web app tests: a camera, detectors, a speaker and a running app."""
import argparse
from contextlib import contextmanager
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from config import add_detector_arguments
from detection import Detection, DetectionBatch
from detectors import DetectorError

W, H = 1280, 720
# Centred, 0.6 of the frame height: a near stair under the alert gate.
NEAR_STAIR = Detection(3, 'Stair', .8, (600, 288, 700, 720))
ENDPOINT = 'http://127.0.0.1:9100'


class Capture:
    """Camera fake: a black frame every 10 ms. `enabled` False means no frames."""
    finished = False
    enabled = True
    def __init__(self, source): pass
    def read(self):
        time.sleep(.01)
        if not self.enabled:
            return False, None, None
        return True, np.zeros((H, W, 3), dtype=np.uint8), time.monotonic()
    def release(self): pass


class FakeDetector:
    """Counts calls. `fail` raises like an unreachable cloud; `healthy` drives /health."""
    def __init__(self, model, detections=(NEAR_STAIR,)):
        self.model, self.detections = model, detections
        self.calls = self.health_calls = 0
        self.fail, self.healthy, self.delay = False, True, 0.
    def predict(self, frame):
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            raise DetectorError('Cloud inference unavailable: connection refused')
        return DetectionBatch(tuple(self.detections), 10, self.model)
    def health(self):
        self.health_calls += 1
        if not self.healthy:
            raise DetectorError('Cloud health check failed: connection refused')
        return {'status': 'ready'}


class FakeSpeaker:
    def __init__(self):
        self.said, self.stopped, self.rate, self.error = [], 0, 180, None
    def say(self, text):
        self.said.append(text)
    def stop(self):
        self.stopped += 1


def make_args(*argv, tts=False):
    parser = argparse.ArgumentParser()
    add_detector_arguments(parser)
    args = parser.parse_args(list(argv))
    args.no_depth, args.no_tts, args.source = True, not tts, 0
    return args


def wait_for(condition, timeout=2.):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(.01)
    return condition()


@contextmanager
def running(detectors, *argv, speaker=None, capture=Capture):
    """Start PipelineState with fake detectors (keyed by backend) and yield (state, client)."""
    from webapp import PipelineState, create_app
    tts = SimpleNamespace(Speaker=lambda: speaker)
    # Patches stay active: the loop reopens a lost camera through LatestFrameCapture.
    with patch('webapp.create_detector', side_effect=lambda settings, **kw: detectors[settings['backend']]), \
         patch('webapp.LatestFrameCapture', capture), \
         patch.dict('sys.modules', {'tts': tts}):
        state = PipelineState(make_args(*argv, tts=speaker is not None))
        try:
            yield state, create_app(state).test_client()
        finally:
            state.stop()
