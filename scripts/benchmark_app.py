"""Measure the real app loop and localhost status responses, not detector-only FPS.

Uses a paced file source, real weights, the general detector optionally, actual
alert/annotation/JPEG stages, and real HTTP. Audio/browser display are separate.
"""
import argparse
import json
from pathlib import Path
import platform
import sys
import threading
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    import cv2
    import numpy as np
    import requests
    from werkzeug.serving import make_server, WSGIRequestHandler
    from config import add_detector_arguments
    from detectors import create_detector as trained_detector
    from webapp import PipelineState, create_app
    parser = argparse.ArgumentParser(description=__doc__)
    add_detector_arguments(parser)
    parser.add_argument('--source', required=True, help='Video path or camera index')
    parser.add_argument('--general', choices=['on', 'off'], default='on')
    parser.add_argument('--frames', type=int, default=60)
    parser.add_argument('--threads', type=int, default=None, help='Experimental CPU thread count after model warmup')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.no_depth, args.no_tts = True, True
    is_camera = args.source.isdigit()
    args.source = int(args.source) if is_camera else str(Path(args.source).resolve())
    class Quiet(WSGIRequestHandler):
        def log_request(self, *args, **kwargs):
            pass
    def bare(settings, **kwargs):
        model = trained_detector(settings)
        model.general = None
        return model
    measurements, responses, byte_sizes = [], [], []
    started = time.monotonic()
    context = patch('webapp.create_detector', side_effect=bare) if args.general == 'off' else patch.dict('os.environ', {})
    with context:
        state = PipelineState(args)
        server = make_server('127.0.0.1', 0, create_app(state), threaded=True, request_handler=Quiet)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        session = requests.Session()
        previous = None
        warmups = 0
        try:
            deadline = time.monotonic() + 90
            while len(measurements) < args.frames and time.monotonic() < deadline:
                tick = time.perf_counter()
                response = session.get(f'http://127.0.0.1:{server.server_port}/status', timeout=5)
                response.raise_for_status()
                elapsed = 1000 * (time.perf_counter() - tick)
                status = response.json()
                with state._lock:
                    captured = state._frame_captured_at
                    jpeg = state._latest_jpeg
                if captured is not None and captured != previous:
                    previous = captured
                    if warmups < 10:
                        warmups += 1
                        if warmups == 10 and args.threads is not None:
                            import torch
                            torch.set_num_threads(args.threads)
                    else:
                        now = time.monotonic()
                        measurements.append(dict(observed_at=now, capture_to_http_observed_ms=1000*(now-captured),
                            detector_ms=status['latency_ms'], reported_fps=status['fps'], error=status['error'],
                            pipeline_ms=status.get('pipeline_latency_ms'), processing_ms=status.get('processing_ms'),
                            preview_encode_ms=status.get('preview_encode_ms')))
                        responses.append(elapsed)
                        byte_sizes.append(len(jpeg) if jpeg else 0)
                if status.get('source', {}).get('ended'):
                    break
                time.sleep(.005)
            if len(measurements) < args.frames:
                raise RuntimeError(f'Only {len(measurements)} fresh processed frames; need {args.frames}')
            def summary(values):
                return dict(median=float(np.median(values)), p95=float(np.percentile(values,95)))
            import torch
            report = dict(platform=platform.platform(), general=args.general, frames=len(measurements),
                torch_threads=torch.get_num_threads(),
                warmup_frames=warmups, source_type='camera' if is_camera else 'paced file',
                startup_and_run_seconds=time.monotonic()-started,
                observed_processed_fps=(len(measurements)-1)/(measurements[-1]['observed_at']-measurements[0]['observed_at']),
                capture_to_http_observed_ms=summary([r['capture_to_http_observed_ms'] for r in measurements]),
                detector_ms=summary([r['detector_ms'] for r in measurements]),
                pipeline_ms=summary([r['pipeline_ms'] for r in measurements]),
                processing_ms=summary([r['processing_ms'] for r in measurements]),
                preview_encode_ms=summary([r['preview_encode_ms'] for r in measurements]),
                status_http_ms=summary(responses), preview_jpeg_bytes=summary(byte_sizes),
                errors=[r['error'] for r in measurements if r['error']],
                health=state.get_status()['health'], measurements=measurements,
                includes=['camera frame read' if is_camera else 'paced file capture','trained model','general COCO model' if args.general=='on' else 'general disabled',
                          'alert decisions','annotation','preview JPEG','publication','localhost HTTP status'],
                excludes=['physical camera exposure/driver latency','native speech playback','browser image paint','network outside localhost'])
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2))
            print(json.dumps({k:v for k,v in report.items() if k!='measurements'},indent=2))
        finally:
            session.close()
            server.shutdown()
            state.stop()


if __name__ == '__main__':
    main()
