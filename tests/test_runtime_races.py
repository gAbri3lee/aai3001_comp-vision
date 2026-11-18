"""Deterministic cancellation, stream bandwidth and shutdown regressions."""
import threading
import time

from helpers import Capture, ENDPOINT, FakeDetector, FakeSpeaker, NEAR_STAIR, running, wait_for


class HeldDetector(FakeDetector):
    def __init__(self, name):
        super().__init__(name)
        self.entered, self.release = threading.Event(), threading.Event()

    def predict(self, frame):
        self.entered.set()
        if not self.release.wait(3):
            raise RuntimeError('Test did not release inference')
        return super().predict(frame)


def test_mode_change_discards_an_inflight_result_and_its_speech():
    local, cloud, speaker = HeldDetector('yolov8n'), HeldDetector('rfdetr-nano'), FakeSpeaker()
    try:
        with running({'local':local,'cloud':cloud}, '--endpoint', ENDPOINT, speaker=speaker) as (state, client):
            assert local.entered.wait(1)
            assert client.post('/mode',json={'backend':'cloud'}).status_code == 200
            local.release.set()
            assert cloud.entered.wait(1)
            assert speaker.said == []
            assert client.get('/status').json['detections'] == []
            cloud.release.set()
    finally:
        local.release.set()
        cloud.release.set()


def test_waiting_for_detector_lock_does_not_run_an_expired_camera_frame():
    detector = FakeDetector('yolov8n')
    Capture.enabled = False
    try:
        with running({'local':detector}, '--max-frame-age', '.05') as (state, client):
            with state._detect_lock:
                Capture.enabled = True
                assert wait_for(lambda: state._last_frame is not None)
                Capture.enabled = False
                time.sleep(.1)
            time.sleep(.1)
            assert detector.calls == 0
    finally:
        Capture.enabled = True


def test_unmuting_makes_the_next_visible_hazard_announceable():
    speaker = FakeSpeaker()
    with running({'local':FakeDetector('yolov8n')}, speaker=speaker) as (state, client):
        assert wait_for(lambda: len(speaker.said) == 1)
        client.post('/control',json={'muted':True})
        client.post('/control',json={'muted':False})
        assert wait_for(lambda: len(speaker.said) >= 2, timeout=.5)


def test_pipeline_shutdown_stops_its_speaker():
    speaker = FakeSpeaker()
    with running({'local':FakeDetector('yolov8n')}, speaker=speaker):
        assert wait_for(lambda: len(speaker.said) == 1)
    assert speaker.stopped >= 1


def test_preview_stream_does_not_resend_the_same_processed_frame():
    from webapp import create_app
    class State:
        _stop = threading.Event()
        first = b'first'
        second = b'second'
        def __init__(self):
            self.change_at = time.monotonic()+.12
        def get_jpeg(self, debug=False):
            return self.first if time.monotonic() < self.change_at else self.second
    state = State()
    response = create_app(state).test_client().get('/video_feed',buffered=False)
    try:
        iterator = iter(response.response)
        assert b'first' in next(iterator)
        assert b'second' in next(iterator)
    finally:
        response.close()
