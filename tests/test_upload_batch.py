"""Batch image upload: per-file checks, Recognition and Alert views, expected-class
scores and Local/Cloud compare (third hand-over, phase 1)."""
import io
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from detection import Detection
from detectors import DetectorError
from helpers import ENDPOINT, FakeDetector, FakeSpeaker, H, W, running, wait_for
from test_analyze import BrightBoxDetector, SceneCapture, encoded, key, scene

# A Stair 0.1 of the frame height in the path: correct, but far, so the gate hides it.
FAR_STAIR = Detection(2, 'Stair', .7, (600, 640, 680, 712))


def post(client, files=(), **form):
    data = {key: value for key, value in form.items()}
    data['image'] = [(io.BytesIO(content), name) for name, content in files]
    return client.post('/analyze', data=data, content_type='multipart/form-data')


def black():
    return np.zeros((H, W, 3), dtype=np.uint8)


def test_a_batch_returns_one_result_per_image_in_order():
    with running({'local': BrightBoxDetector('yolov8n')}) as (state, client):
        files = [(f'img{i}.jpg', encoded(scene() if i % 2 else black(), 'jpg')) for i in range(5)]
        response = post(client, files, view='recognition')
        assert response.status_code == 200
        body = response.json
        assert [r['name'] for r in body['results']] == [f'img{i}.jpg' for i in range(5)]
        assert body['backends'] == ['local'] and body['cloud_used'] is False
        for i, entry in enumerate(body['results']):
            [run] = entry['runs']
            assert run['model'] == 'yolov8n + general-yolov8n' and run['latency_ms'] == 5
            assert set(run['images']) == {'recognition', 'alert'} and 'image' not in run
            assert [d['class'] for d in run['recognition']] == (['Stair'] if i % 2 else [])


def test_a_bad_file_is_a_row_with_an_error_and_never_fails_the_batch():
    gif = io.BytesIO()
    Image.new('RGB', (20, 20)).save(gif, format='GIF')
    jpeg = encoded(scene(), 'jpg')
    files = [('good1.jpg', jpeg), ('anim.gif', gif.getvalue()), ('cut.jpg', jpeg[:len(jpeg) // 2]),
             ('big.jpg', jpeg + b'\0' * (10 * 1024 * 1024 + 1 - len(jpeg))), ('notes.txt', b'hello'),
             ('good2.png', encoded(scene(), 'png'))]
    with running({'local': BrightBoxDetector('yolov8n')}) as (state, client):
        response = post(client, files, view='alert')
        assert response.status_code == 200
        rows = {r['name']: r for r in response.json['results']}
        assert 'JPEG or PNG' in rows['anim.gif']['error'] and 'JPEG or PNG' in rows['notes.txt']['error']
        assert 'damaged' in rows['cut.jpg']['error'] and '10 MB' in rows['big.jpg']['error']
        for good in ('good1.jpg', 'good2.png'):
            assert rows[good]['runs'][0]['alert'] == 'Warning, Stair ahead'
        too_many = [(f'{i}.jpg', jpeg) for i in range(21)]
        response = post(client, too_many)
        assert response.status_code == 400 and 'at most 20' in response.json['error']
        assert post(client, [('a.jpg', jpeg)], view='sideways').status_code == 400
        assert post(client, [('a.jpg', jpeg)], expected='Chair').status_code == 400


def test_recognition_view_shows_a_far_detection_that_alert_view_hides():
    with running({'local': FakeDetector('yolov8n', (FAR_STAIR,))}) as (state, client):
        for view in ('recognition', 'alert'):
            result = post(client, [('far.jpg', encoded(black(), 'jpg'))], view=view).json
            assert result['view'] == view
            # Both views come from the same detections; the view picks the picture.
            assert [d['class'] for d in result['recognition']] == ['Stair']
            assert result['alert'] == '' and result['detections'][0]['reason'] == 'far'
            assert result['image'] == result['images'][view]
        assert result['images']['recognition'] != result['images']['alert']


def test_recognition_view_never_speaks_and_alert_view_speaks_only_when_asked():
    speaker = FakeSpeaker()
    with running({'local': BrightBoxDetector('yolov8n')}, speaker=speaker) as (state, client):
        jpeg = encoded(scene(), 'jpg')
        assert post(client, [('a.jpg', jpeg)], view='recognition', speak='1').json['spoken'] is False
        assert post(client, [('a.jpg', jpeg)], view='alert').json['spoken'] is False
        assert speaker.said == []
        assert post(client, [('a.jpg', jpeg)], view='alert', speak='1').json['spoken'] is True
        assert speaker.said == ['Warning, Stair ahead']
        # A batch never speaks: twenty phrases would be meaningless.
        post(client, [('a.jpg', jpeg), ('b.jpg', jpeg)], view='alert', speak='1')
        assert speaker.said == ['Warning, Stair ahead']


def test_expected_class_marks_each_image_and_summarises_the_batch():
    files = [('s1.jpg', encoded(scene(), 'jpg')), ('s2.png', encoded(scene(), 'png')),
             ('none.jpg', encoded(black(), 'jpg')), ('bad.jpg', b'not an image')]
    with running({'local': BrightBoxDetector('yolov8n')}) as (state, client):
        body = post(client, files, view='recognition', expected='Stair').json
        marks = {r['name']: r.get('runs', [{}])[0].get('expected') for r in body['results']}
        assert marks['s1.jpg'] == {'class': 'Stair', 'recognised': True, 'confidence': .8}
        assert marks['none.jpg'] == {'class': 'Stair', 'recognised': False, 'confidence': None}
        [summary] = body['summary']
        assert summary['text'] == 'Stair: 2 of 3 recognised, mean confidence 0.80'
        assert summary['skipped'] == 1 and summary['backend'] == 'local'
        # The tested accuracy always sits next to the score, from the config.
        assert body['accuracy_note'].startswith('This is a check on these images only. Four-class scope.')
        assert 'dataset cleanup did not retrain it' in body['accuracy_note']
        body = post(client, files[2:3], expected='People').json
        assert body['summary'][0]['text'] == 'People: 0 of 1 recognised'
        assert body['expected']['recognised'] is False


def test_alert_view_in_a_batch_matches_the_live_loop():
    detector = BrightBoxDetector('yolov8n')
    with running({'local': detector}, capture=SceneCapture) as (state, client):
        # The live loop speaks (selects) a hazard once 2-of-3 confirmation has seen it twice.
        assert wait_for(lambda: (client.get('/status').json['selected'] or {}).get('phrase') == 'Warning, Stair ahead')
        live = client.get('/status').json
        body = post(client, [('a.jpg', encoded(scene(), 'png')), ('b.png', encoded(scene(), 'png'))],
                    view='alert').json
        for entry in body['results']:
            run = entry['runs'][0]
            assert run['alert'] == live['alert'] and run['alerts'] == live['alerts']
            assert key(run['detections']) == key(live['detections'])
            assert run['selected']['phrase'] == live['selected']['phrase']


def test_compare_runs_both_backends_and_local_only_never_calls_the_cloud():
    local, cloud = BrightBoxDetector('yolov8n'), BrightBoxDetector('rfdetr-nano')
    files = [('a.jpg', encoded(scene(), 'jpg')), ('b.jpg', encoded(black(), 'jpg'))]
    with running({'local': local, 'cloud': cloud}, '--endpoint', ENDPOINT) as (state, client):
        assert wait_for(lambda: local.calls > 2)
        body = post(client, files, view='recognition', expected='Stair').json
        assert cloud.calls == 0 and body['cloud_used'] is False and body['backends'] == ['local']
        body = post(client, files, view='recognition', expected='Stair', compare='1').json
        assert cloud.calls == 2 and body['cloud_used'] is True
        assert body['backends'] == ['local', 'cloud'] and state.mode_state()['mode'] == 'local'
        for entry in body['results']:
            assert [run['mode'] for run in entry['runs']] == ['local', 'cloud']
        assert [s['text'] for s in body['summary']] == ['Stair: 1 of 2 recognised, mean confidence 0.80'] * 2
        # A cloud failure is a per-run error; the local result still stands.
        def unreachable(frame):
            raise DetectorError('Cloud inference unavailable: connection refused')
        cloud.predict = unreachable
        body = post(client, files[:1], compare='1').json
        local_run, cloud_run = body['results'][0]['runs']
        assert 'alert' in local_run and 'Analysis failed' in cloud_run['error']


def test_compare_without_a_cloud_endpoint_is_refused():
    with running({'local': BrightBoxDetector('yolov8n')}) as (state, client):
        response = post(client, [('a.jpg', encoded(scene(), 'jpg'))], compare='1')
        assert response.status_code == 503 and '--endpoint' in response.json['error']


def test_nothing_from_a_batch_survives_the_request(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    temp = Path(tempfile.gettempdir())
    before = set(temp.iterdir())
    big = encoded(np.random.default_rng(0).integers(0, 255, (H, W, 3), dtype=np.uint8), 'png')
    with running({'local': BrightBoxDetector('yolov8n')}) as (state, client):
        files = [('../../evil.png', big), ('x.jpg', encoded(scene(), 'jpg'))]
        body = post(client, files).json
        assert body['results'][0]['name'] == 'evil.png'  # shown, never used as a path
    assert list(tmp_path.iterdir()) == []
    assert {p for p in set(temp.iterdir()) - before if 'visionaid' in p.name} == set()


def test_display_names_are_cleaned():
    from media import display_name
    assert display_name('C:\\Users\\me\\photo.jpg', 'x') == 'photo.jpg'
    assert display_name('a\x00b\nc.png', 'x') == 'abc.png'
    assert display_name('', 'Image 3') == 'Image 3'
    assert len(display_name('n' * 500 + '.jpg', 'x')) == 80
