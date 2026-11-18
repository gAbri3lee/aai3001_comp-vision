"""Exercise upload routes with the supplied trained YOLO; no physical camera needed.

Only camera acquisition is replaced. Detection, decoding, annotations, alert
policy and file-video capture use the actual application implementations.
"""
import io
import argparse
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tests'))
from helpers import make_args
from capture import LatestFrameCapture
from config import CLASS_NAMES
from detectors import create_detector as build_detector
from scene_detector import SceneDetector
from webapp import PipelineState, create_app


class IdleCamera:
    finished = False
    def read(self):
        time.sleep(.01)
        return False, None, None
    def release(self): pass
    def pause(self): pass
    def resume(self): pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--serve', action='store_true', help='Open the real upload UI on localhost:5055; camera/speech/cloud disabled, general optional')
    parser.add_argument('--model', type=Path, default=ROOT / 'models/local/yolov8n_v4_four_class.pt')
    parser.add_argument('--general', action='store_true', help='Include the real COCO general detector')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/four-class-real-uploads')
    options = parser.parse_args()
    data = ROOT / 'data/v4_four_class_clean/test'
    out = options.output
    out.mkdir(parents=True, exist_ok=True)
    selected = []
    for cid in range(len(CLASS_NAMES)):
        candidates = []
        for label in sorted((data / 'labels').glob('*.txt')):
            boxes = [list(map(float, line.split())) for line in label.read_text().splitlines() if line.strip()]
            matching = [box for box in boxes if int(box[0]) == cid]
            image = data / 'images' / (label.stem + '.jpg')
            if matching and image.exists():
                candidates.append((max(b[3]*b[4] for b in matching), image))
        selected.append(max(candidates, key=lambda item: item[0])[1])
    args = make_args('--model', str(options.model.resolve()))
    real_capture = LatestFrameCapture
    def capture(source, **kwargs):
        return IdleCamera() if isinstance(source, int) else real_capture(source, **kwargs)
    report = {'detector': options.model.name, 'general_detector': options.general, 'camera': 'disabled',
              'sample_selection': 'largest labelled box per expected class; selected demonstration, not an accuracy estimate'}
    def detector(settings, **kwargs):
        if options.general:
            return SceneDetector(settings, **kwargs)
        model = build_detector(settings)
        model.general = None  # truthful model label; no optional COCO inference in this verification
        return model
    with patch('webapp.create_detector', side_effect=detector), \
         patch('webapp.LatestFrameCapture', side_effect=capture):
        state = PipelineState(args)
        try:
            app = create_app(state)
            if options.serve:
                app.run(host='127.0.0.1', port=5055, threaded=True, use_reloader=False)
                return
            client = app.test_client()
            response = client.post('/analyze', data={
                'view': 'recognition',
                'image': [(io.BytesIO(path.read_bytes()), path.name) for path in selected]
                         + [(io.BytesIO(b'damaged image'), 'damaged.jpg')]}, content_type='multipart/form-data')
            assert response.status_code == 200, response.json
            entries = response.json['results']
            report['images'] = [{'expected': expected, 'image': path.name,
                                 'recognition': entry['runs'][0]['recognition'],
                                 'expected_class_seen': any(d['class'] == expected for d in entry['runs'][0]['recognition'])}
                                for expected, path, entry in zip(CLASS_NAMES, selected, entries)]
            assert 'error' in entries[-1]
            report['corrupt_file_isolated'] = True
            # Exercise the optional distance display using a DECLARED synthetic
            # calibration, never presented as measured distance ground truth.
            state.calibration = {'focal_px': 900., 'frame_height': 640}
            response = client.post('/analyze', data={'image': (io.BytesIO(selected[1].read_bytes()), 'person.jpg')},
                                   content_type='multipart/form-data')
            report['distance_display_with_assumed_calibration'] = response.json.get('detections', [])
            state.calibration = None
            video = out / 'four-class-slideshow.mp4'
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 10., (640, 640))
            assert writer.isOpened()
            for path in selected:
                frame = cv2.resize(cv2.imread(str(path)), (640, 640))
                for _ in range(20): writer.write(frame)
            writer.release()
            response = client.post('/source', data={'video': (io.BytesIO(video.read_bytes()), video.name)},
                                   content_type='multipart/form-data')
            assert response.status_code == 200, response.json
            deadline = time.monotonic() + 40
            while time.monotonic() < deadline:
                status = client.get('/status').json
                if status['source']['ended']: break
                time.sleep(.1)
            assert status['source']['ended'], status
            report['video'] = {'description': f'{len(selected)*2}-second slideshow of actual labelled images; not field footage',
                               'frames_processed': status['frames_processed'], 'summary': status['video_summary']}
            assert client.post('/source/control', json={'action': 'restart'}).status_code == 200
            report['replay_accepted'] = True
            assert client.post('/source/control', json={'action': 'camera'}).status_code == 200
        finally:
            state.stop()
    report['cleanup_passed'] = state.store.root is None
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
