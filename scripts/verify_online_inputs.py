"""Actual-model online-source upload smoke test; not a five-class accuracy test."""
import argparse
import io
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tests'))
from helpers import make_args
from capture import LatestFrameCapture
from detectors import create_detector as build_detector
from scene_detector import SceneDetector
from webapp import PipelineState, create_app
from verify_real_uploads import IdleCamera


def main():
    import cv2
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', type=Path, default=ROOT / 'data/online_sg_malls')
    parser.add_argument('--model', type=Path, default=ROOT / 'artifacts/supplied_weights/local/yolov8n_v4.pt')
    parser.add_argument('--general', action='store_true', help='Include the actual COCO general detector')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/online-source-uploads')
    args = parser.parse_args()
    records = [r for r in json.loads((args.sources / 'sources.json').read_text()) if 'file' in r]
    images = [r for r in records if Path(r['file']).suffix.lower() in ('.jpg', '.jpeg', '.png')]
    videos = [r for r in records if r['role'] == 'external_challenge_video']
    if len(images) > 20 or len(videos) != 1:
        raise ValueError('Expected up to 20 images and one preselected source video')
    args.output.mkdir(parents=True, exist_ok=True)
    video = args.output / 'vivocity-source-converted.mp4'
    capture = cv2.VideoCapture(str(args.sources / videos[0]['file']))
    if not capture.isOpened():
        raise ValueError('Could not decode original Commons video')
    fps = capture.get(cv2.CAP_PROP_FPS)
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    if not writer.isOpened():
        raise ValueError('Could not encode MP4')
    count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(frame)
            count += 1
    finally:
        writer.release()
        capture.release()
    real_capture = LatestFrameCapture
    def fake_camera(source, **kwargs):
        return IdleCamera() if isinstance(source, int) else real_capture(source, **kwargs)
    report = dict(model=args.model.name, labels='Not fully annotated; no mAP/class-accuracy claim',
        camera=False, speech=False, cloud=False, general_detector=args.general,
        video_source=videos[0], video_modification='Converted Theora to MP4 without audio; no frame scaling/cropping',
        source_frames=count, source_fps=fps)
    def detector(settings, **kwargs):
        if args.general:
            return SceneDetector(settings, **kwargs)
        model = build_detector(settings)
        model.general = None
        return model
    with patch('webapp.LatestFrameCapture', side_effect=fake_camera), \
         patch('webapp.create_detector', side_effect=detector):
        state = PipelineState(make_args('--model', str(args.model.resolve())))
        try:
            client = create_app(state).test_client()
            response = client.post('/analyze', data={'view':'recognition', 'image':[
                (io.BytesIO((args.sources / r['file']).read_bytes()), r['file']) for r in images]},
                content_type='multipart/form-data')
            assert response.status_code == 200, response.json
            results = response.json['results']
            if any('error' in result for result in results):
                raise ValueError('Some online images failed decoding/inference')
            report['images'] = [dict(id=r['id'], venue=r['venue'], role=r['role'],
                recognition=result['runs'][0]['recognition'],
                alert_detections=result['runs'][0]['detections']) for r,result in zip(images,results)]
            response = client.post('/source', data={'video':(io.BytesIO(video.read_bytes()),video.name)},
                                   content_type='multipart/form-data')
            assert response.status_code == 200, response.json
            deadline = time.monotonic() + 45
            while time.monotonic() < deadline:
                status = client.get('/status').json
                if status['source']['ended']:
                    break
                time.sleep(.1)
            assert status['source']['ended'], status
            report['video'] = dict(ended=True, frames_processed=status['frames_processed'],
                                    recognition_summary=status['video_summary'])
            report['replay_accepted'] = client.post('/source/control',json={'action':'restart'}).status_code == 200
        finally:
            state.stop()
    report['cleanup_passed'] = state.store.root is None
    (args.output / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
