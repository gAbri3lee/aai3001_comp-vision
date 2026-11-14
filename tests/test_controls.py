"""Pause, mute, speech rate, test voice and replay (POST /control)."""
import time

from helpers import FakeDetector, FakeSpeaker, running, wait_for


def stair_alert(client):
    return client.get('/status').json['alert'] == 'Warning, Stair ahead'


def test_pause_stops_inference_and_speech_and_resume_restarts_them():
    detector, speaker = FakeDetector('yolov8n'), FakeSpeaker()
    with running({'local': detector}, speaker=speaker) as (state, client):
        assert wait_for(lambda: speaker.said == ['Warning, Stair ahead'])
        response = client.post('/control', json={'paused': True})
        assert response.status_code == 200 and response.json['paused'] is True
        assert speaker.stopped == 1  # the current phrase is cut off
        time.sleep(.1)
        calls = detector.calls
        state.policy.reset()  # a ready hazard would be spoken if anything ran
        time.sleep(.3)
        assert detector.calls == calls and speaker.said == ['Warning, Stair ahead']
        status = client.get('/status').json
        assert status['paused'] is True and status['alert'] == '' and status['detections'] == []
        assert state.get_jpeg() is not None  # last frame stays up with a Paused label
        assert status['health'] == []        # the camera is still open and sending frames
        client.post('/control', json={'paused': False})
        assert wait_for(lambda: detector.calls > calls and stair_alert(client))


def test_mute_stops_speech_but_not_detection():
    detector, speaker = FakeDetector('yolov8n'), FakeSpeaker()
    with running({'local': detector}, speaker=speaker) as (state, client):
        assert wait_for(lambda: len(speaker.said) == 1)
        response = client.post('/control', json={'muted': True})
        assert response.json['muted'] is True and speaker.stopped == 1
        calls = detector.calls
        state.policy.reset()
        time.sleep(.3)
        assert len(speaker.said) == 1                       # nothing new reaches the speaker
        assert detector.calls > calls and stair_alert(client)  # detection and banner continue
        status = client.get('/status').json
        assert status['muted'] is True and status['selected']['phrase'] == 'Warning, Stair ahead'
        client.post('/control', json={'muted': False})
        state.policy.reset()
        assert wait_for(lambda: len(speaker.said) == 2)


def test_rate_is_validated_and_clamped():
    speaker = FakeSpeaker()
    with running({'local': FakeDetector('yolov8n')}, speaker=speaker) as (state, client):
        assert client.get('/status').json['speech_rate'] == 180 and speaker.rate == 180
        for sent, kept in ((200, 200), (500, 260), (50, 120), (181.4, 181)):
            response = client.post('/control', json={'rate': sent})
            assert response.status_code == 200 and response.json['speech_rate'] == kept
            assert speaker.rate == kept
        for bad in ('fast', True, None, [180]):
            response = client.post('/control', json={'rate': bad})
            assert response.status_code == 400 and 'rate' in response.json['error']
        assert client.post('/control', data='{"rate": NaN}', content_type='application/json').status_code == 400
        assert client.get('/status').json['speech_rate'] == 181


def test_invalid_controls_are_rejected_without_partial_changes():
    with running({'local': FakeDetector('yolov8n')}, speaker=FakeSpeaker()) as (state, client):
        for body in ({}, [], {'volume': 3}, {'paused': 'yes'}, {'muted': 1}, {'action': 'shout'},
                     {'paused': True, 'rate': 'fast'}):
            assert client.post('/control', json=body).status_code == 400, body
        assert client.post('/control', data='paused').status_code == 400
        status = client.get('/status').json
        assert status['paused'] is False and status['muted'] is False


def test_test_voice_and_replay_use_the_same_speaker_and_respect_mute():
    speaker = FakeSpeaker()
    with running({'local': FakeDetector('yolov8n')}, speaker=speaker) as (state, client):
        response = client.post('/control', json={'action': 'replay_last'})
        if not speaker.said:
            assert response.status_code == 409 and 'No alert' in response.json['error']
        assert wait_for(lambda: speaker.said[:1] == ['Warning, Stair ahead'])
        assert client.post('/control', json={'action': 'test_voice'}).status_code == 200
        assert speaker.said[-1] == 'VisionAid voice test'
        assert client.post('/control', json={'action': 'replay_last'}).status_code == 200
        assert speaker.said[-1] == 'Warning, Stair ahead'
        assert client.get('/status').json['last_spoken'] == 'Warning, Stair ahead'  # replay is not a new alert
        client.post('/control', json={'muted': True})
        count = len(speaker.said)
        for action in ('test_voice', 'replay_last'):
            response = client.post('/control', json={'action': action})
            assert response.status_code == 409 and 'muted' in response.json['error']
        assert len(speaker.said) == count


def test_speech_actions_explain_when_speech_is_off():
    with running({'local': FakeDetector('yolov8n')}) as (state, client):
        response = client.post('/control', json={'action': 'test_voice'})
        assert response.status_code == 409 and '--no-tts' in response.json['error']
        assert client.get('/status').json['speech_enabled'] is False
