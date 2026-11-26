"""Vercel page: video upload sends only JPEG frames to the existing /api/detect endpoint.

The browser behaviour (playback, throttling, overlay alignment, stale replies, failure handling) is exercised by
deploy/vercel/check_video_ui.py against a real browser. These tests cover what can be checked without one: the page's
upload contract and what the unchanged endpoint does with video-derived frames.
"""
import importlib.util
import io
from pathlib import Path
import re
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
API = ROOT / 'deploy' / 'vercel'
PAGE = (API / 'public' / 'index.html').read_text(encoding='utf-8')
VIDEO = ROOT / 'demo' / 'videos' / 'Wet floor sign - field sequence.mp4'

try:
    import onnxruntime  # noqa: F401
    import cv2
    HAVE_RUNTIME = True
except ImportError:
    HAVE_RUNTIME = False
MODEL_READY = any((API / 'model').glob('*.onnx')) and (API / 'config.json').is_file()
needs_model = pytest.mark.skipif(not (HAVE_RUNTIME and MODEL_READY and VIDEO.is_file()),
                                 reason='needs onnxruntime, opencv, the ONNX model (deploy/vercel/prepare.py) and the demo video')


def script():
    return re.search(r'<script>(.*)</script>', PAGE, re.S).group(1)


def test_picker_accepts_images_and_videos():
    assert re.search(r'<input id="file" type="file" accept="image/\*,video/\*"', PAGE)
    for control in ('id="video"', 'id="overlay"', 'id="play"', 'id="pause"', 'id="restart"'):
        assert control in PAGE


def test_only_jpeg_frames_are_ever_uploaded():
    js = script()
    calls = re.findall(r'fetch\(([^;]*?)\)\s*;', js, re.S)
    assert len(calls) == 2                       # the photo path and the video-frame path, nothing else
    for call in calls:
        assert call.startswith("'/api/detect'") and "'Content-Type': 'image/jpeg'" in call
        assert 'file' not in call.split('body:')[1].split(',')[0].lower()   # never the File object itself
    assert 'FormData' not in js and 'XMLHttpRequest' not in js and 'sendBeacon' not in js


def test_video_stays_local_and_object_url_is_released():
    js = script()
    assert 'URL.createObjectURL(file)' in js and 'URL.revokeObjectURL(vs.url)' in js
    assert js.count('toBlob(') == 2              # photo + video frame: the only encoders feeding fetch


def test_request_rate_is_bounded():
    js = script()
    assert 'if (s.inFlight) { s.skipped++; return; }' in js                      # one request at a time; busy ticks skip
    assert "<option value=\"3\">3</option>" in PAGE and 'value="4"' not in PAGE   # selectable rate is capped
    assert re.search(r'REQUEST_TIMEOUT_MS = \d+', js) and 'MAX_BACKOFF_MS' in js  # hung or failing requests cannot pile up


@pytest.fixture(scope='module')
def client():
    sys.path.insert(0, str(API))
    try:
        spec = importlib.util.spec_from_file_location('vercel_app_under_test', API / 'app.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module.app.test_client()
    finally:
        sys.path.remove(str(API))


def frame_as_browser_would(index, max_side=640):
    """A video frame shrunk to max_side and JPEG-encoded at quality 80, like the page's canvas.toBlob."""
    from PIL import Image
    capture = cv2.VideoCapture(str(VIDEO))
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, bgr = capture.read()
    capture.release()
    assert ok
    image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    scale = min(1, max_side / max(image.size))
    image = image.resize((round(image.width * scale), round(image.height * scale)))
    buffer = io.BytesIO()
    image.save(buffer, 'JPEG', quality=80)
    return buffer.getvalue(), image.size


@needs_model
def test_video_frames_are_detected_through_the_existing_endpoint(client):
    found = set()
    for index in range(0, 240, 20):
        jpeg, size = frame_as_browser_would(index)
        reply = client.post('/api/detect', data=jpeg, headers={'Content-Type': 'image/jpeg'})
        assert reply.status_code == 200 and len(jpeg) < 400_000
        body = reply.get_json()
        assert body['image_size'] == list(size) and body['model'] == 'yolov8n-onnx'
        for d in body['detections']:
            x1, y1, x2, y2 = d['box_xyxy']
            assert 0 <= x1 < x2 <= size[0] and 0 <= y1 < y2 <= size[1]     # boxes are in the sent frame's pixels
            found.add(d['class'])
    assert 'Wet Floor Sign' in found


@needs_model
def test_endpoint_still_refuses_video_bytes(client):
    video = VIDEO.read_bytes()[:200_000]
    assert client.post('/api/detect', data=video, headers={'Content-Type': 'video/mp4'}).status_code == 415
    assert client.post('/api/detect', data=video, headers={'Content-Type': 'image/jpeg'}).status_code == 400
