"""Real localhost HTTP checks against verify_real_uploads.py --general --serve.

Uses actual weights and media; this is not browser click-through testing.
"""
import json
from pathlib import Path
import time

import requests


def main():
    root = Path(__file__).resolve().parents[1]
    base = 'http://127.0.0.1:5055'
    session = requests.Session()
    report = {}
    def post(path, **kwargs):
        return session.post(base + path, timeout=30, **kwargs)
    def status():
        response = session.get(base + '/status', timeout=5)
        response.raise_for_status()
        return response.json()
    try:
        assert session.get(base, timeout=5).status_code == 200
        report['page_served'] = True
        bus = root / 'artifacts/system-validation/bus.jpg'
        response = post('/analyze', data={'view':'alert'}, files=[
            ('image', ('bus.jpg', bus.read_bytes(), 'image/jpeg')),
            ('image', ('broken.jpg', b'broken image', 'image/jpeg'))])
        assert response.status_code == 200, response.text[:500]
        results = response.json()['results']
        assert 'error' in results[1] and 'error' not in results[0]
        report['real_image_and_isolated_corruption'] = True
        report['alert_rows'] = results[0]['runs'][0]['detections']
        assert any(r['source'] == 'general' and r['class'] == 'Object' for r in report['alert_rows'])
        for body in ({'paused':True}, {'paused':False}, {'muted':True}, {'muted':False}, {'rate':220}):
            response = post('/control', json=body)
            assert response.status_code == 200, response.text
            assert all(response.json()['speech_rate' if key == 'rate' else key] == value
                       for key,value in body.items()), response.json()
        report['controls'] = True
        assert post('/control', json={'rate':'invalid'}).status_code == 400
        assert post('/mode', json={'backend':'cloud'}).status_code == 503
        assert post('/source', files={'video':('bad.mp4',b'broken video','video/mp4')}).status_code == 415
        report['invalid_inputs_and_unavailable_cloud_rejected'] = True
        video = root / 'artifacts/online-source-uploads-isolated-final/vivocity-source-converted.mp4'
        response = post('/source', files={'video':('vivocity.mp4', video.read_bytes(), 'video/mp4')})
        assert response.status_code == 200, response.text[:500]
        assert post('/source/control', json={'view':'alert'}).status_code == 200
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and status()['frames_processed'] < 3:
            time.sleep(.1)
        assert status()['frames_processed'] >= 3
        assert post('/control', json={'paused':True}).status_code == 200
        time.sleep(.2)
        paused_count = status()['frames_processed']
        time.sleep(.3)
        assert status()['frames_processed'] == paused_count
        assert post('/control', json={'paused':False}).status_code == 200
        with session.get(base + '/video_feed', stream=True, timeout=5) as stream:
            assert stream.status_code == 200
            assert b'--frame' in next(stream.iter_content(1024))
        report['network_mjpeg_and_video_pause_resume'] = True
        while time.monotonic() < deadline:
            last = status()
            if last['source']['ended']:
                break
            time.sleep(.1)
        assert last['source']['ended'], last
        report['real_mall_video_processed_frames'] = last['frames_processed']
        assert post('/source/control', json={'action':'restart'}).status_code == 200
        assert post('/source/control', json={'action':'camera'}).status_code == 200
        report['video_end_replay_and_return_to_camera'] = True
    finally:
        session.close()
    out = root / 'artifacts/system-validation/http-runtime.json'
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
