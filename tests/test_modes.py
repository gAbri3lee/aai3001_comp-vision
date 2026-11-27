"""Local/Cloud switch: status, validation, privacy, fallback, retry and cooldown reset."""
import time
from unittest.mock import MagicMock, patch

from helpers import ENDPOINT, FakeDetector, FakeSpeaker, running, wait_for
from decision_engine import AlertPolicy
from config import load_config
from helpers import NEAR_STAIR, W, H


def fakes():
    return {'local': FakeDetector('yolov8n'), 'cloud': FakeDetector('rfdetr-nano')}


def test_switching_updates_status_and_model():
    detectors = fakes()
    with running(detectors, '--endpoint', ENDPOINT) as (state, client):
        status = client.get('/status').json
        assert status['mode'] == 'local' and status['mode_label'] == 'LOCAL, YOLOv8n'
        assert status['model'].startswith('yolov8n') and status['cloud_available'] is True
        assert status['fallback_active'] is False and status['last_error'] == ''
        response = client.post('/mode', json={'backend': 'cloud'})
        assert response.status_code == 200 and response.json['mode'] == 'cloud'
        status = client.get('/status').json
        assert status['mode'] == 'cloud' and status['backend'] == 'cloud'
        assert status['mode_label'] == 'CLOUD, RF-DETR Nano' and status['model'].startswith('rfdetr-nano')
        assert wait_for(lambda: detectors['cloud'].calls > 0)
        assert client.post('/mode', json={'backend': 'local'}).json['mode'] == 'local'


def test_invalid_mode_is_rejected_and_cloud_needs_an_endpoint():
    detectors = fakes()
    with running(detectors, '--endpoint', ENDPOINT) as (state, client):
        for body in ({'backend': 'gpu'}, {'backend': None}, {}, ['cloud']):
            response = client.post('/mode', json=body)
            assert response.status_code == 400 and 'local' in response.json['error']
        assert client.post('/mode', data='cloud').status_code == 400
        assert client.get('/status').json['mode'] == 'local'
    with running(fakes()) as (state, client):
        status = client.get('/status').json
        assert status['cloud_available'] is False and 'endpoint' in status['cloud_unavailable_reason']
        response = client.post('/mode', json={'backend': 'cloud'})
        assert response.status_code == 503 and response.json['mode'] == 'local'


def test_local_mode_never_calls_the_cloud_detector():
    detectors = fakes()
    with running(detectors, '--endpoint', ENDPOINT) as (state, client):
        assert wait_for(lambda: detectors['local'].calls >= 10)
        client.get('/status')
    assert detectors['cloud'].calls == 0 and detectors['cloud'].health_calls == 0


def test_local_mode_sends_no_http_request_with_the_real_cloud_client():
    """Privacy at the HTTP layer: the real CloudDetector is built but never used."""
    from detectors import CloudDetector
    local = FakeDetector('yolov8n')

    def build(settings, **kw):
        if settings['backend'] == 'local':
            return local
        cloud = CloudDetector(settings)
        cloud.session = MagicMock()
        build.cloud = cloud
        return cloud

    from webapp import PipelineState
    from helpers import Capture, make_args
    with patch('webapp.create_detector', side_effect=build), patch('webapp.LatestFrameCapture', Capture), \
         patch('requests.get') as get, patch('requests.post') as post:
        state = PipelineState(make_args('--endpoint', ENDPOINT))
        try:
            assert wait_for(lambda: local.calls >= 10)
        finally:
            state.stop()
    assert build.cloud.session.method_calls == [] and get.call_count == 0 and post.call_count == 0


def test_cloud_failure_falls_back_to_local_and_holds_the_last_result_briefly():
    detectors = fakes()
    with running(detectors, '--backend', 'cloud', '--endpoint', ENDPOINT) as (state, client):
        assert wait_for(lambda: client.get('/status').json['alert'] == 'Warning, Stair ahead')
        detectors['local'].delay = 2.  # local's first result arrives after the hold ends
        detectors['cloud'].fail = True
        assert wait_for(lambda: state.mode_state()['fallback_active'])
        status = client.get('/status').json
        assert status['mode'] == 'local' and 'Cloud unreachable' in status['last_error']
        assert {'code': 'cloud_unreachable', 'severity': 'warning',
                'message': 'Cloud unreachable, using local.'} in status['health']
        assert status['alert'] == 'Warning, Stair ahead'  # last good result, held
        time.sleep(1.1)
        status = client.get('/status').json
        assert status['alert'] == '' and status['detections'] == []  # never held past one second
        detectors['local'].delay = 0.
        cloud_calls = detectors['cloud'].calls
        assert wait_for(lambda: client.get('/status').json['alert'] == 'Warning, Stair ahead', 4.)
        assert detectors['cloud'].calls == cloud_calls  # stays on local (decision U2)
        assert client.get('/status').json['fallback_active'] is True


def test_retry_switches_back_only_when_health_succeeds():
    detectors = fakes()
    with running(detectors, '--backend', 'cloud', '--endpoint', ENDPOINT) as (state, client):
        detectors['cloud'].fail = True
        detectors['cloud'].healthy = False
        assert wait_for(lambda: state.mode_state()['fallback_active'])
        response = client.post('/mode', json={'backend': 'cloud'})
        assert response.status_code == 503 and 'Cloud unreachable' in response.json['error']
        status = client.get('/status').json
        assert status['mode'] == 'local' and status['fallback_active'] is True
        detectors['cloud'].fail = False
        detectors['cloud'].healthy = True
        response = client.post('/mode', json={'backend': 'cloud'})
        assert response.status_code == 200 and response.json['mode'] == 'cloud'
        status = client.get('/status').json
        assert status['fallback_active'] is False and status['health'] == []


def test_mode_switch_clears_the_cooldown_so_the_next_alert_is_spoken():
    speaker = FakeSpeaker()
    with running(fakes(), '--endpoint', ENDPOINT, speaker=speaker) as (state, client):
        assert wait_for(lambda: speaker.said == ['Warning, Stair ahead'])
        time.sleep(.2)
        assert speaker.said == ['Warning, Stair ahead']  # cooldown holds the repeat back
        client.post('/mode', json={'backend': 'cloud'})
        assert wait_for(lambda: len(speaker.said) == 2, 1.)  # well inside the 4 s cooldown
        assert speaker.said == ['Warning, Stair ahead'] * 2


def test_policy_reset_forgets_cooldowns():
    config = load_config()
    # Without frame confirmation (covered in test_alert_confirmation.py).
    policy = AlertPolicy({**config['alert_gate'], 'confirm_frames': {'required': 1, 'window': 1}}, config['severity'], clock=lambda: 0.)
    assert policy.process((NEAR_STAIR,), (W, H), now=0.).speak == 'Warning, Stair ahead'
    assert policy.process((NEAR_STAIR,), (W, H), now=1.).speak is None
    policy.reset()
    assert policy.process((NEAR_STAIR,), (W, H), now=1.5).speak == 'Warning, Stair ahead'
