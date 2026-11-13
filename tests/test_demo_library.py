"""Demo library: one-click images and videos chosen by id, never by path
(third hand-over, phase 3)."""
import os

import cv2
import numpy as np

from helpers import FakeDetector, running, wait_for
from test_analyze import BrightBoxDetector, encoded, scene
from test_video_upload import camera_or_file, scene_video, source


def library(tmp_path):
    root = tmp_path / 'demo'
    for name in ('Stair', 'People', 'videos'):
        (root / name).mkdir(parents=True)
    (root / 'Stair' / 'stair1.jpg').write_bytes(encoded(scene(), 'jpg'))
    (root / 'Stair' / 'stair2.png').write_bytes(encoded(scene(), 'png'))
    (root / 'Stair' / 'notes.txt').write_text('not an image')
    (root / 'Stair' / '.hidden.jpg').write_bytes(encoded(scene(), 'jpg'))
    (root / 'People' / 'empty_room.jpg').write_bytes(encoded(np.zeros((720, 1280, 3), np.uint8), 'jpg'))
    outside = tmp_path / 'secret.jpg'
    outside.write_bytes(encoded(scene(), 'jpg'))
    os.symlink(outside, root / 'Stair' / 'escape.jpg')  # leads out of the library: ignored
    scene_video(root / 'videos' / 'walk.mp4', frames=40)
    return root


def test_an_empty_or_missing_library_explains_how_to_fill_it(tmp_path):
    with running({'local': FakeDetector('yolov8n')}) as (state, client):
        state.demo_root = tmp_path / 'demo'
        listing = client.get('/demo').json
        assert listing['empty'] and 'demo/videos/' in listing['help'] and listing['videos'] == []
        assert [c['class'] for c in listing['classes']] == ['Escalator', 'People', 'Pillar', 'Stair', 'Wet Floor Sign']


def test_the_listing_has_names_and_ids_only(tmp_path):
    root = library(tmp_path)
    with running({'local': FakeDetector('yolov8n')}) as (state, client):
        state.demo_root = root
        response = client.get('/demo')
        text = response.get_data(as_text=True)
        assert str(tmp_path) not in text and 'secret' not in text and 'escape' not in text
        groups = {c['class']: [i['name'] for i in c['images']] for c in response.json['classes']}
        assert groups['Stair'] == ['stair1.jpg', 'stair2.png'] and groups['People'] == ['empty_room.jpg']
        assert [v['name'] for v in response.json['videos']] == ['walk.mp4']
        ids = [i['id'] for c in response.json['classes'] for i in c['images']]
        assert all(i.startswith('img-') and '/' not in i and '.' not in i for i in ids)
        # Ids stay the same between requests, so the page can use them later.
        assert ids == [i['id'] for c in client.get('/demo').json['classes'] for i in c['images']]


def test_a_library_image_is_analysed_with_its_folder_as_the_expected_class(tmp_path):
    root = library(tmp_path)
    with running({'local': BrightBoxDetector('yolov8n')}) as (state, client):
        state.demo_root = root
        listing = client.get('/demo').json
        stair = next(c for c in listing['classes'] if c['class'] == 'Stair')['images'][0]
        result = client.post('/analyze', data={'demo': stair['id'], 'view': 'recognition'}).json
        assert result['name'] == 'stair1.jpg' and result['source'] == 'Demo library'
        assert result['expected'] == {'class': 'Stair', 'recognised': True, 'confidence': .8}
        body = client.post('/analyze', data={'demo_class': 'Stair'}).json
        assert [r['name'] for r in body['results']] == ['stair1.jpg', 'stair2.png']
        assert body['summary'][0]['text'] == 'Stair: 2 of 2 recognised, mean confidence 0.80'
        body = client.post('/analyze', data={'demo_class': 'People'}).json
        assert body['summary'][0]['text'] == 'People: 0 of 1 recognised'
        # The presenter's own choice of expected class wins.
        body = client.post('/analyze', data={'demo_class': 'Stair', 'expected': 'People'}).json
        assert body['summary'][0]['class'] == 'People'


def test_unknown_ids_and_paths_are_refused(tmp_path):
    root = library(tmp_path)
    with running({'local': FakeDetector('yolov8n')}) as (state, client):
        state.demo_root = root
        for bad in ('img-000000000000', '../secret.jpg', str(tmp_path / 'secret.jpg'), 'Stair/stair1.jpg'):
            assert client.post('/analyze', data={'demo': bad}).status_code == 404
            assert client.get(f'/demo/thumb/{bad}').status_code == 404
            response = client.post('/source', data={'demo': bad})
            assert response.status_code == 404 and 'Unknown library video' in response.json['error']
        assert client.post('/analyze', data={'demo_class': 'Chair'}).status_code == 400


def test_thumbnails_are_small_jpegs(tmp_path):
    root = library(tmp_path)
    with running({'local': FakeDetector('yolov8n')}) as (state, client):
        state.demo_root = root
        item = client.get('/demo').json['classes'][3]['images'][0]
        response = client.get(f"/demo/thumb/{item['id']}")
        assert response.status_code == 200 and response.mimetype == 'image/jpeg'
        image = cv2.imdecode(np.frombuffer(response.data, np.uint8), cv2.IMREAD_COLOR)
        assert max(image.shape[:2]) == 240


def test_a_library_video_plays_and_is_never_deleted(tmp_path):
    root = library(tmp_path)
    with running({'local': BrightBoxDetector('yolov8n')}, capture=camera_or_file()) as (state, client):
        state.demo_root = root
        video = client.get('/demo').json['videos'][0]
        response = client.post('/source', data={'demo': video['id']})
        assert response.status_code == 200
        assert response.json['source']['origin'] == 'library' and response.json['source']['name'] == 'walk.mp4'
        assert wait_for(lambda: client.get('/status').json['alert'] == 'Warning, Stair ahead')
        assert wait_for(lambda: source(client)['ended'], timeout=4)
        assert client.post('/source/control', json={'action': 'restart'}).status_code == 200
        assert client.post('/source/control', json={'action': 'camera'}).status_code == 200
        assert wait_for(lambda: source(client)['type'] == 'camera')
        assert (root / 'videos' / 'walk.mp4').exists() and state.store.root is None
    assert (root / 'videos' / 'walk.mp4').exists()


def test_the_real_demo_folder_is_set_up_for_the_presenter():
    from config import PROJECT_ROOT
    root = PROJECT_ROOT / 'demo'
    for name in ('Escalator', 'People', 'Pillar', 'Stair', 'Wet Floor Sign', 'videos'):
        assert (root / name).is_dir()
    readme = (root / 'README.md').read_text()
    assert 'test split' in readme and 'Pillar' in readme and '.gitignore' in readme
    ignore = (PROJECT_ROOT / '.gitignore').read_text()
    assert 'demo/videos/*' in ignore and '*.mp4' in ignore
