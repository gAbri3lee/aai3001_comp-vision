"""Upload and sample analysis share the live path and validate their input."""
import base64
import io

import cv2
import numpy as np
from PIL import Image

from detection import Detection, DetectionBatch
from helpers import Capture, FakeDetector, FakeSpeaker, H, W, running, wait_for


def scene():
    """A frame with one bright block: centred and 0.6 of the height, so a near Stair."""
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    frame[288:720, 600:700] = 255
    return frame


class BrightBoxDetector(FakeDetector):
    """Finds the bright block, so its output depends on the image it is given."""
    def predict(self, frame):
        self.calls += 1
        ys, xs = np.nonzero(frame[:, :, 0] > 128)
        if not len(xs):
            return DetectionBatch((), 5, self.model)
        box = (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))
        return DetectionBatch((Detection(2, 'Stair', .8, box),), 5, self.model)


class SceneCapture(Capture):
    def read(self):
        ok, frame, at = super().read()
        return ok, scene() if ok else frame, at


def encoded(frame, fmt):
    ok, buffer = cv2.imencode('.' + fmt, frame)
    return buffer.tobytes()


def upload(client, data, name='photo.jpg', **form):
    return client.post('/analyze', data={'image': (io.BytesIO(data), name), **form},
                       content_type='multipart/form-data')


def key(detections):
    return [(d['class'], d['position'], d['band'], d['gated'], d['reason'], d['phrase']) for d in detections]


def test_upload_matches_the_live_path_for_the_same_image():
    detector = BrightBoxDetector('yolov8n')
    with running({'local': detector}, capture=SceneCapture) as (state, client):
        assert wait_for(lambda: client.get('/status').json['alert'] == 'Warning, Stair ahead')
        live = client.get('/status').json
        response = upload(client, encoded(scene(), 'png'), 'scene.png')
        assert response.status_code == 200
        result = response.json
        assert result['source'] == 'upload' and result['mode'] == 'local'
        assert result['alert'] == live['alert'] and result['alerts'] == live['alerts']
        assert key(result['detections']) == key(live['detections'])
        assert result['selected']['phrase'] == live['selected']['phrase'] == 'Warning, Stair ahead'
        image = base64.b64decode(result['image'].split(',', 1)[1])
        assert cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR).shape == (H, W, 3)


def test_jpeg_and_png_work_and_analysis_has_its_own_cooldown():
    speaker = FakeSpeaker()
    detector = BrightBoxDetector('yolov8n')
    with running({'local': detector}, speaker=speaker, capture=SceneCapture) as (state, client):
        assert wait_for(lambda: speaker.said == ['Warning, Stair ahead'])
        live_cooldown = dict(state.policy.cooldown._last)
        for fmt in ('jpg', 'png'):
            result = upload(client, encoded(scene(), fmt), 'x.' + fmt).json
            assert result['alert'] == 'Warning, Stair ahead' and result['spoken'] is False
            assert result['selected'] is not None  # a fresh cooldown every time
        assert dict(state.policy.cooldown._last).keys() == live_cooldown.keys()
        said = len(speaker.said)
        result = upload(client, encoded(scene(), 'jpg'), speak='1').json
        assert result['spoken'] is True and speaker.said[said:] == ['Warning, Stair ahead']
        assert state.control_state()['last_spoken'] == 'Warning, Stair ahead'
        client.post('/control', json={'muted': True})
        result = upload(client, encoded(scene(), 'jpg'), speak='1').json
        assert result['spoken'] is False and len(speaker.said) == said + 1  # respects mute


def test_wrong_type_oversized_and_corrupt_files_are_rejected_cleanly():
    with running({'local': FakeDetector('yolov8n')}) as (state, client):
        gif = io.BytesIO()
        Image.new('RGB', (20, 20)).save(gif, format='GIF')
        response = upload(client, gif.getvalue(), 'x.gif')
        assert response.status_code == 415 and 'JPEG or PNG' in response.json['error']
        assert upload(client, b'hello, not an image', 'notes.txt').status_code == 415
        assert upload(client, b'', 'empty.jpg').status_code == 400
        jpeg = encoded(scene(), 'jpg')
        response = upload(client, jpeg[:len(jpeg) // 2], 'cut.jpg')
        assert response.status_code == 400 and 'damaged' in response.json['error']
        png = encoded(scene(), 'png')
        assert upload(client, png[:len(png) // 2], 'cut.png').status_code == 400
        # Just over 10 MB: rejected by the exact limit.
        big = jpeg + b'\0' * (10 * 1024 * 1024 + 1 - len(jpeg))
        response = upload(client, big, 'big.jpg')
        assert response.status_code == 413 and '10 MB' in response.json['error']
        # Far over: rejected before the body is read.
        response = upload(client, b'\0' * (11 * 1024 * 1024), 'huge.jpg')
        assert response.status_code == 413 and '10 MB' in response.json['error']
        response = client.post('/analyze', data={}, content_type='multipart/form-data')
        assert response.status_code == 400 and 'Choose' in response.json['error']
        assert client.get('/status').status_code == 200  # the app is still fine


def test_client_filename_is_never_used_and_nothing_is_stored(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with running({'local': FakeDetector('yolov8n')}) as (state, client):
        response = upload(client, encoded(scene(), 'jpg'), '../../evil.jpg')
        assert response.status_code == 200 and response.json['source'] == 'upload'
    assert list(tmp_path.iterdir()) == []


def test_samples_are_listed_by_id_and_run_the_same_way():
    with running({'local': FakeDetector('yolov8n')}) as (state, client):
        samples = client.get('/samples').json
        assert samples and samples[0] == {'id': 'sample-1', 'name': 'Sample 1'}
        assert all('/' not in s['id'] and '.' not in s['id'] for s in samples)
        response = client.post('/analyze', data={'sample': 'sample-1'})
        assert response.status_code == 200 and response.json['source'] == 'Sample 1'
        assert response.json['width'] == 640  # golden samples are 640 px; the fake box sits right
        assert response.json['alert'] == 'Warning, Stair on your right'
        for bad in ('sample-999', '../visionaid_config.json', 'golden/images'):
            assert client.post('/analyze', data={'sample': bad}).status_code == 404


def test_analysis_failure_returns_a_clear_error():
    detector = FakeDetector('yolov8n')
    with running({'local': detector}) as (state, client):
        detector.fail = True
        response = upload(client, encoded(scene(), 'jpg'))
        assert response.status_code == 503 and 'Analysis failed' in response.json['error']
