"""Video upload plays through the live pipeline (third hand-over, phase 2).

Videos are generated in the test with OpenCV. The camera is a fake; uploaded
files use the real LatestFrameCapture, so pacing, rotation and the end of the
file behave as they do in the app.
"""
import io
import struct
import time

import cv2
import numpy as np

from capture import LatestFrameCapture as RealCapture
from detection import DetectionBatch
from helpers import Capture, ENDPOINT, FakeDetector, FakeSpeaker, H, W, running, wait_for
from test_analyze import BrightBoxDetector, SceneCapture, key, scene
from test_upload_batch import FAR_STAIR


def write_video(path, frames=20, fps=20, size=(W, H), frame=None, fourcc='mp4v'):
    width, height = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, (width, height))
    for i in range(frames):
        writer.write(frame(i) if frame else np.zeros((height, width, 3), dtype=np.uint8))
    writer.release()
    return path


def scene_video(path, frames=30, fps=20):
    return write_video(path, frames, fps, frame=lambda i: scene())


def camera_or_file(camera=Capture):
    """Fake camera for the start source, the real reader for uploaded files."""
    def build(source, max_side=None):
        return RealCapture(source, max_side=max_side) if isinstance(source, str) else camera(source)
    return build


def send(client, path, name=None):
    data = {'video': (io.BytesIO(path.read_bytes()), name or path.name)}
    return client.post('/source', data=data, content_type='multipart/form-data')


def source(client):
    return client.get('/status').json['source']


def temp_files(state):
    return sorted(state.store.root.iterdir()) if state.store.root else []


def test_a_video_plays_like_the_camera_and_ends_with_a_summary(tmp_path):
    speaker = FakeSpeaker()
    video = scene_video(tmp_path / 'walk.mp4', frames=30, fps=20)
    with running({'local': BrightBoxDetector('yolov8n')}, speaker=speaker,
                 capture=camera_or_file()) as (state, client):
        assert source(client) == {'type': 'camera', 'origin': 'camera', 'home': 'camera', 'name': 'Camera',
                                  'duration': None, 'position': None, 'fps': None, 'ended': False,
                                  'replay_available': False}
        response = send(client, video, '../../My walk.mp4')
        assert response.status_code == 200
        playing = response.json['source']
        assert playing['type'] == 'video' and playing['origin'] == 'upload' and playing['name'] == 'My walk.mp4'
        assert playing['duration'] == 1.5 and playing['fps'] == 20
        assert wait_for(lambda: client.get('/status').json['alert'] == 'Warning, Stair ahead')
        assert wait_for(lambda: (source(client)['position'] or 0) > 0.3)
        assert wait_for(lambda: source(client)['ended'], timeout=4)
        status = client.get('/status').json
        assert [h['code'] for h in status['health']] == ['video_ended']
        assert status['alert'] == '' and state.get_jpeg() is not None  # last picture stays
        summary = {row['class']: row for row in status['video_summary']}
        assert summary['Stair']['seen'] and summary['Stair']['frames'] > 3
        assert 0 <= summary['Stair']['first_seen'] <= summary['Stair']['last_seen'] <= 1.5
        assert summary['Stair']['max_confidence'] == .8 and not summary['People']['seen']
        assert status['frames_processed'] >= summary['Stair']['frames']
        assert status['accuracy_note'].startswith('This is a check on this video only. Four-class scope.')
        assert speaker.said == ['Warning, Stair ahead']  # the live cooldown rules
        # The pipeline is still usable: Replay plays it again.
        [copy] = temp_files(state)
        assert client.post('/source/control', json={'action': 'restart'}).status_code == 200
        assert wait_for(lambda: not source(client)['ended'] and client.get('/status').json['alert'])
        assert client.post('/source/control', json={'action': 'camera'}).status_code == 200
        assert wait_for(lambda: source(client)['type'] == 'camera')
        assert wait_for(lambda: not copy.exists() and temp_files(state) == [])


def test_rejected_videos_give_clear_messages_and_leave_no_file(tmp_path, monkeypatch):
    with running({'local': FakeDetector('yolov8n')}, capture=camera_or_file()) as (state, client):
        good = write_video(tmp_path / 'ok.mp4', frames=5)
        response = send(client, good, 'clip.avi')
        assert response.status_code == 415 and 'MP4, MOV or WebM' in response.json['error']
        (tmp_path / 'fake.mp4').write_bytes(b'this is not a video at all' * 100)
        assert send(client, tmp_path / 'fake.mp4').status_code == 415
        data = good.read_bytes()
        (tmp_path / 'cut.mp4').write_bytes(data[:64] + b'\0' * 2000)
        response = send(client, tmp_path / 'cut.mp4')
        assert response.status_code == 400 and 'damaged' in response.json['error']
        long_video = write_video(tmp_path / 'long.mp4', frames=122, fps=1, size=(64, 48))
        response = send(client, long_video)
        assert response.status_code == 400 and 'longer than 120 seconds' in response.json['error']
        monkeypatch.setattr('media.MAX_VIDEO_BYTES', 1000)
        response = send(client, good)
        assert response.status_code == 413 and '150 MB' in response.json['error']
        assert client.post('/source', data={}).status_code == 400
        assert temp_files(state) == [] and source(client)['type'] == 'camera'
        for body in ({}, {'action': 'rewind'}, {'view': 'x'}, {'speed': 2}):
            assert client.post('/source/control', json=body).status_code == 400
        assert client.post('/source/control', json={'action': 'restart'}).status_code == 409


def test_a_request_over_the_video_limit_is_refused_before_it_is_read(tmp_path, monkeypatch):
    monkeypatch.setattr('media.MAX_VIDEO_BYTES', 1000)
    with running({'local': FakeDetector('yolov8n')}, capture=camera_or_file()) as (state, client):
        big = tmp_path / 'big.mp4'
        big.write_bytes(b'\0' * (2 * 1024 * 1024))
        response = send(client, big)
        assert response.status_code == 413 and '150 MB' in response.json['error']
        assert temp_files(state) == []


def test_temporary_copies_are_deleted_on_replace_camera_expiry_and_stop(tmp_path, monkeypatch):
    first = write_video(tmp_path / 'a.mp4', frames=200)
    second = write_video(tmp_path / 'b.webm', frames=200, fourcc='VP80')
    with running({'local': FakeDetector('yolov8n')}, capture=camera_or_file()) as (state, client):
        assert send(client, first).status_code == 200
        assert wait_for(lambda: source(client)['name'] == 'a.mp4')
        [copy_a] = temp_files(state)
        assert copy_a.suffix == '.mp4' and copy_a.name != 'a.mp4'  # a random name
        assert send(client, second).status_code == 200
        assert wait_for(lambda: source(client)['name'] == 'b.webm')
        # /status reports the queued source immediately; disposal happens in
        # the loop thread when that switch is applied.
        assert wait_for(lambda: not copy_a.exists() and len(temp_files(state)) == 1)
        assert client.post('/source/control', json={'action': 'camera'}).status_code == 200
        assert wait_for(lambda: temp_files(state) == [])
        # An ended upload is kept for Replay for a while, then deleted.
        monkeypatch.setattr(type(state), 'VIDEO_KEEP_SECONDS', 0.2)
        short = write_video(tmp_path / 'c.mp4', frames=4)
        assert send(client, short).status_code == 200
        assert wait_for(lambda: source(client)['ended'] and not source(client)['replay_available'], timeout=4)
        assert [h['code'] for h in client.get('/status').json['health']] == ['video_deleted']
        response = client.post('/source/control', json={'action': 'restart'})
        assert response.status_code == 409 and 'Upload the video again' in response.json['error']
        assert send(client, first).status_code == 200
        root = state.store.root
    assert not root.exists()  # stopping the app removes every copy


def test_a_rotated_phone_clip_is_played_upright(tmp_path):
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    frame[:, :8] = 255  # a white stripe on the left of the stored picture
    path = write_video(tmp_path / 'phone.mp4', frames=40, fps=20, size=(64, 48), frame=lambda i: frame)
    data = bytearray(path.read_bytes())
    matrix = data.find(b'tkhd') + 4 + 40  # version 0 track header: the display matrix
    data[matrix:matrix + 36] = struct.pack('>9I', 0, 0x10000, 0, 0xFFFF0000, 0, 0, 0, 0, 0x40000000)
    path.write_bytes(bytes(data))  # 90 degree rotation metadata, as a phone writes it

    seen = []

    class Recorder(FakeDetector):
        def predict(self, image):
            seen.append(image.copy())
            return DetectionBatch((), 1, self.model)

    with running({'local': Recorder('yolov8n')}, capture=camera_or_file()) as (state, client):
        assert send(client, path).status_code == 200
        assert wait_for(lambda: state.video is not None and seen and seen[-1].shape[:2] == (64, 48))
        assert (state.video['width'], state.video['height']) == (48, 64)
        upright = seen[-1]
        assert upright[:8].mean() > 200 and upright[-8:].mean() < 30  # the stripe is now on top


def test_very_large_frames_are_downscaled_before_inference(tmp_path):
    seen = []

    class Recorder(FakeDetector):
        def predict(self, image):
            seen.append(image.shape)
            return DetectionBatch((), 1, self.model)

    path = write_video(tmp_path / 'big.mp4', frames=6, fps=10, size=(2560, 1440))
    with running({'local': Recorder('yolov8n')}, capture=camera_or_file()) as (state, client):
        assert send(client, path).status_code == 200
        assert wait_for(lambda: (1080, 1920, 3) in seen)


def test_a_video_frame_gets_the_same_alert_as_the_live_camera(tmp_path):
    video = scene_video(tmp_path / 'same.mp4', frames=60)
    with running({'local': BrightBoxDetector('yolov8n')}, capture=camera_or_file(SceneCapture)) as (state, client):
        # Camera and video are frame sequences: a hazard is selected once 2-of-3 confirmation has seen it twice.
        assert wait_for(lambda: (client.get('/status').json['selected'] or {}).get('phrase') == 'Warning, Stair ahead')
        live = client.get('/status').json
        assert send(client, video).status_code == 200
        def active_video_alert():
            snapshot = client.get('/status').json
            return snapshot if snapshot['source']['type'] == 'video' and snapshot['alert'] and snapshot['selected'] else None

        deadline = time.monotonic() + 2
        played = active_video_alert()
        while played is None and time.monotonic() < deadline:
            time.sleep(.01)
            played = active_video_alert()
        assert played is not None
        assert played['alert'] == live['alert'] and played['alerts'] == live['alerts']
        assert key(played['detections']) == key(live['detections'])
        assert played['selected']['phrase'] == live['selected']['phrase']


def test_local_video_never_reaches_the_cloud_and_cloud_mode_does(tmp_path):
    local, cloud = BrightBoxDetector('yolov8n'), BrightBoxDetector('rfdetr-nano')
    video = scene_video(tmp_path / 'walk.mp4', frames=80)
    with running({'local': local, 'cloud': cloud}, '--endpoint', ENDPOINT,
                 capture=camera_or_file()) as (state, client):
        assert send(client, video).status_code == 200
        assert wait_for(lambda: client.get('/status').json['alert'] == 'Warning, Stair ahead')
        assert cloud.calls == 0
        assert client.post('/mode', json={'backend': 'cloud'}).status_code == 200
        assert wait_for(lambda: cloud.calls > 2)
        assert client.get('/status').json['mode'] == 'cloud'


def test_pause_stops_the_video_and_mute_silences_it(tmp_path):
    speaker = FakeSpeaker()
    video = scene_video(tmp_path / 'walk.mp4', frames=60)
    with running({'local': BrightBoxDetector('yolov8n')}, speaker=speaker,
                 capture=camera_or_file()) as (state, client):
        assert send(client, video).status_code == 200
        assert wait_for(lambda: (source(client)['position'] or 0) > 0.2)
        client.post('/control', json={'paused': True})
        time.sleep(.2)
        held = source(client)['position']
        time.sleep(.5)
        status = client.get('/status').json
        assert status['source']['position'] == held and status['paused'] and status['alert'] == ''
        assert status['health'] == []  # a paused video is not "no video"
        client.post('/control', json={'paused': False})
        assert wait_for(lambda: source(client)['position'] > held)
        stops = speaker.stopped
        client.post('/control', json={'muted': True})
        assert speaker.stopped == stops + 1
        said = len(speaker.said)
        client.post('/source/control', json={'action': 'restart'})  # a fresh cooldown, still muted
        assert wait_for(lambda: client.get('/status').json['alert'] == 'Warning, Stair ahead')
        time.sleep(.2)
        assert len(speaker.said) == said


def test_recognition_view_for_video_skips_the_gate_and_never_speaks(tmp_path):
    speaker = FakeSpeaker()
    video = scene_video(tmp_path / 'far.mp4', frames=80)
    near = BrightBoxDetector('yolov8n')
    with running({'local': FakeDetector('yolov8n', (FAR_STAIR,))}, speaker=speaker,
                 capture=camera_or_file()) as (state, client):
        assert send(client, video).status_code == 200
        # Wait for the uploaded source's results in one snapshot. The camera can
        # publish first, then correctly clear detections while the video opens.
        status = {}
        def video_results_ready():
            nonlocal status
            status = client.get('/status').json
            return status['source']['type'] == 'video' and bool(status['detections'])
        assert wait_for(video_results_ready)
        assert status['view'] == 'alert' and status['alert'] == '' and not status['detections'][0]['gated']
        assert [d['class'] for d in status['recognition']] == ['Stair']  # recorded in both views
        response = client.post('/source/control', json={'view': 'recognition'})
        assert response.status_code == 200 and response.json['view'] == 'recognition'
        assert wait_for(lambda: client.get('/status').json['view'] == 'recognition')
        time.sleep(.2)
        status = client.get('/status').json
        assert [d['class'] for d in status['recognition']] == ['Stair'] and status['detections'] == []
        state.detectors['local'] = near  # a near stair would be spoken in Alert view
        time.sleep(.3)
        assert speaker.said == [] and client.get('/status').json['alert'] == ''
        client.post('/source/control', json={'view': 'alert'})
        assert wait_for(lambda: speaker.said == ['Warning, Stair ahead'])
        # Back on the camera the live rules always apply.
        client.post('/source/control', json={'view': 'recognition', 'action': 'camera'})
        assert wait_for(lambda: client.get('/status').json['source']['type'] == 'camera')
        assert client.get('/status').json['view'] == 'alert'


def test_one_upload_at_a_time(tmp_path):
    video = write_video(tmp_path / 'a.mp4', frames=5)
    with running({'local': FakeDetector('yolov8n')}, capture=camera_or_file()) as (state, client):
        assert state._upload_lock.acquire()
        try:
            response = send(client, video)
            assert response.status_code == 409 and 'Another video' in response.json['error']
        finally:
            state._upload_lock.release()
        assert temp_files(state) == []


def test_slow_recorded_recognition_keeps_the_result_with_its_frame(tmp_path):
    class SlowDetector(FakeDetector):
        def predict(self, frame):
            time.sleep(.09)
            return super().predict(frame)

    clip = write_video(tmp_path / 'slow.mp4', frames=32, fps=16, size=(320, 240))
    with running({'local': SlowDetector('yolov8n')}, '--max-frame-age', '.05',
                 capture=camera_or_file()) as (_, client):
        assert send(client, clip).status_code == 200
        assert client.post('/source/control', json={'view': 'recognition'}).status_code == 200
        assert wait_for(lambda: any(d['class'] == 'Stair' for d in client.get('/status').json.get('recognition', [])))
