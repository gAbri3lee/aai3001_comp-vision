"""Runtime problems produce a clear `health` entry and never crash the loop."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from helpers import Capture, FakeDetector, FakeSpeaker, make_args, running, wait_for
from webapp import PipelineState, create_app


def codes(client):
    return [entry['code'] for entry in client.get('/status').json['health']]


def stair_alert(client):
    return client.get('/status').json['alert'] == 'Warning, Stair ahead'


@pytest.fixture(autouse=True)
def fast_retry():
    with patch.object(PipelineState, 'CAMERA_RETRY_SECONDS', .05), \
         patch.object(PipelineState, 'NO_VIDEO_SECONDS', .3):
        yield


def test_camera_missing_at_start_shows_a_message_and_recovers():
    class Camera(Capture):
        available = False
        def __init__(self, source):
            if not Camera.available:
                raise RuntimeError('Could not open camera/source: 0')

    with running({'local': FakeDetector('yolov8n')}, capture=Camera) as (state, client):
        status = client.get('/status').json
        assert status['health'][0] == {'code': 'camera_missing', 'severity': 'error',
                                       'message': 'Camera not found. Check the connection.'}
        assert status['alert'] == '' and client.get('/').status_code == 200
        Camera.available = True
        assert wait_for(lambda: stair_alert(client))
        assert codes(client) == []


def test_camera_lost_while_running_shows_a_message_and_recovers():
    class Camera(Capture):
        lost = False
        def read(self):
            if Camera.lost:
                self.finished = True
                return False, None, None
            return super().read()

    with running({'local': FakeDetector('yolov8n')}, capture=Camera) as (state, client):
        assert wait_for(lambda: stair_alert(client))
        Camera.lost = True
        assert wait_for(lambda: codes(client) == ['camera_missing'])
        assert client.get('/status').json['alert'] == '' and state.get_jpeg() is None
        Camera.lost = False
        assert wait_for(lambda: stair_alert(client))
        assert codes(client) == []


def test_missing_video_file_still_stops_the_app_at_startup():
    args = make_args()
    args.source = 'does-not-exist.mp4'
    with patch('webapp.create_detector', return_value=FakeDetector('yolov8n')):
        with pytest.raises(RuntimeError, match='Could not open'):
            PipelineState(args)


def test_no_frames_clears_the_banner_and_stops_speech():
    speaker = FakeSpeaker()

    class Camera(Capture):
        pass

    with running({'local': FakeDetector('yolov8n')}, speaker=speaker, capture=Camera) as (state, client):
        assert wait_for(lambda: stair_alert(client))
        Camera.enabled = False
        assert wait_for(lambda: 'no_video' in codes(client))
        status = client.get('/status').json
        assert status['alert'] == '' and status['detections'] == [] and status['selected'] is None
        assert {'code': 'no_video', 'severity': 'error', 'message': 'No video. Waiting for camera.'} in status['health']
        spoken = len(speaker.said)
        state.policy.reset()  # even with no cooldown, nothing is spoken without frames
        assert not wait_for(lambda: len(speaker.said) > spoken, .3)
        Camera.enabled = True
        assert wait_for(lambda: stair_alert(client))
        assert 'no_video' not in codes(client)


def test_general_model_unavailable_is_reported_and_alerts_continue():
    detector = FakeDetector('yolov8n')
    detector.general_error = 'General model unavailable: download failed'
    with running({'local': detector}) as (state, client):
        assert wait_for(lambda: stair_alert(client))
        status = client.get('/status').json
        assert {'code': 'general_unavailable', 'severity': 'warning',
                'message': 'Extra object detection is off.'} in status['health']
        assert status['model'] == 'yolov8n + general-yolov8n'
        detector.general_error = None
        assert wait_for(lambda: 'general_unavailable' not in codes(client))


def test_speech_failure_is_reported_and_the_banner_keeps_working():
    speaker = FakeSpeaker()
    speaker.error = 'Audio device unavailable'
    with running({'local': FakeDetector('yolov8n')}, speaker=speaker) as (state, client):
        assert wait_for(lambda: stair_alert(client))
        assert {'code': 'speech_unavailable', 'severity': 'warning',
                'message': 'Speech is unavailable. Alerts show on screen only.'} in client.get('/status').json['health']
        speaker.error = None
        assert 'speech_unavailable' not in codes(client)


def test_speaker_that_cannot_start_is_reported_without_stopping_the_app():
    def broken():
        raise OSError('no speech engine')
    with patch('webapp.create_detector', return_value=FakeDetector('yolov8n')), \
         patch('webapp.LatestFrameCapture', Capture), \
         patch.dict('sys.modules', {'tts': SimpleNamespace(Speaker=broken)}):
        state = PipelineState(make_args(tts=True))
        try:
            client = create_app(state).test_client()
            assert wait_for(lambda: stair_alert(client))
            assert 'speech_unavailable' in codes(client)
        finally:
            state.stop()


def test_detector_errors_keep_the_loop_alive():
    detector = FakeDetector('yolov8n')
    with running({'local': detector}) as (state, client):
        assert wait_for(lambda: stair_alert(client))
        detector.fail = True
        assert wait_for(lambda: 'unavailable' in client.get('/status').json['error'])
        assert client.get('/status').json['alert'] == ''
        detector.fail = False
        assert wait_for(lambda: stair_alert(client))
        assert state._thread.is_alive()


def test_general_model_failures_do_not_hide_trained_detections():
    from scene_detector import SceneDetector
    from detection import Detection, DetectionBatch
    stair = Detection(3, 'Stair', .8, (0, 0, 10, 10))
    trained = SimpleNamespace(predict=lambda frame: DetectionBatch((stair,), 5, 'yolov8n'))
    with patch('scene_detector.create_detector', return_value=trained), \
         patch('scene_detector.load_general_model', side_effect=OSError('download failed')):
        detector = SceneDetector({'device': 'cpu'})
    assert detector.general is None and 'download failed' in detector.general_error
    assert detector.predict(SimpleNamespace(shape=(60, 100, 3))).detections == (stair,)

    def explode(*args, **kwargs):
        raise RuntimeError('general crashed')
    with patch('scene_detector.create_detector', return_value=trained):
        detector = SceneDetector({'device': 'cpu'}, general=SimpleNamespace(predict=explode))
    assert detector.predict(SimpleNamespace(shape=(60, 100, 3))).detections == (stair,)
    assert 'general crashed' in detector.general_error
