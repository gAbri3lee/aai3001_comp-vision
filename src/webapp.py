"""Live object detection, relative depth and shared GUI/audio alerts."""
import argparse
import threading
import time

import cv2
from flask import Flask, Response, jsonify, render_template
from config import add_detector_arguments, detector_settings
from detectors import DetectorError
from scene_detector import SceneDetector as create_detector
from detection import annotate
from capture import LatestFrameCapture

from decision_engine import AlertCooldown, build_hazards, phrase_for

class PipelineState:
    """Owns the background capture/inference loop and exposes thread-safe
    reads for Flask's request handlers (which run on separate threads)."""

    def __init__(self, args):
        self.args = args
        self.settings = detector_settings(args)
        self.detector = create_detector(self.settings)
        if not args.no_depth:
            from depth import DepthEstimator
        if not args.no_tts:
            from tts import Speaker
        self.depth_estimator = None if args.no_depth else DepthEstimator()
        self.speaker = None if args.no_tts else Speaker()
        self.cooldown = AlertCooldown()
        self._lock = threading.Lock()
        self._latest_jpeg = None
        self._frame_captured_at = None
        self.status = {"fps": 0.0, "alert": "", "alerts": [],
                       "model": self.settings["model"] + ' + general-yolov8n', "backend": args.backend, "error": "",
                       "latency_ms": None, "round_trip_ms": None, "detections": []}

        self._stop = threading.Event()

        source = int(args.source) if str(args.source).isdigit() else args.source
        self.capture = LatestFrameCapture(source)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _frame_is_stale(self):
        return (self._frame_captured_at is not None and
                time.monotonic() - self._frame_captured_at > self.args.max_frame_age)

    def get_jpeg(self):
        with self._lock:
            return None if self._frame_is_stale() else self._latest_jpeg

    def get_status(self) -> dict:
        with self._lock:
            status = dict(self.status)
            if self._frame_is_stale():
                status.update(alert="", alerts=[], detections=[], fps=0., latency_ms=None, round_trip_ms=None,
                              error="Frame is stale; waiting for fresh inference")
            return status

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=self.args.timeout + 3)
        self.capture.release()

    def _loop(self):
        try:
            prev_time = time.monotonic()
            while not self._stop.is_set():
                ok, frame, captured_at = self.capture.read()
                if not ok:
                    if self.capture.finished:
                        with self._lock:
                            self.status.update(alert="", alerts=[], detections=[], fps=0., latency_ms=None, round_trip_ms=None, error="Camera/source ended")
                            self._latest_jpeg = None
                            self._frame_captured_at = None
                        break
                    self._stop.wait(0.01)
                    continue
                alert_text = ""
                alerts = []
                error = ""
                batch = None
                annotated = frame.copy()
                try:
                    if time.monotonic() - captured_at > self.args.max_frame_age:
                        raise DetectorError('Captured frame is stale; discarded')
                    batch = self.detector.predict(frame)
                    depth_map = self.depth_estimator.predict(frame) if self.depth_estimator is not None and batch.detections else None
                    if time.monotonic() - captured_at > self.args.max_frame_age:
                        raise DetectorError('Inference result is stale; discarded')
                    hazards = build_hazards(batch.detections, depth_map, frame.shape[1], self.settings['severity'])
                    annotated = annotate(frame, batch.detections)
                    for hazard in hazards:
                        if hazard.proximity is not None:
                            x1, y1 = int(hazard.xyxy[0]), int(hazard.xyxy[1])
                            cv2.putText(annotated, f"{hazard.proximity:.2f}", (x1, max(y1 - 10, 15)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)
                    alerts = list(dict.fromkeys(phrase_for(hazard) for hazard in hazards))
                    if hazards:
                        alert_text = alerts[0]
                        # One current message per frame, in priority order. A named
                        # hazard on cooldown must not starve generic object alerts.
                        if self.speaker is not None:
                            for hazard in hazards:
                                if self.cooldown.should_announce(hazard):
                                    self.speaker.say(phrase_for(hazard))
                                    break
                except Exception as exc:
                    # Never retain prior boxes/alerts after an inference failure.
                    error = str(exc)
                    alert_text = ""
                    alerts = []
                    batch = None
                    annotated = frame.copy()

                now = time.monotonic()
                fps = 1.0 / max(now - prev_time, 1e-6)
                prev_time = now

                ok_jpeg, buffer = cv2.imencode(".jpg", annotated)
                with self._lock:
                    if ok_jpeg:
                        self._latest_jpeg = buffer.tobytes()
                        self._frame_captured_at = captured_at
                    self.status["fps"] = round(fps, 1)
                    self.status["alert"] = alert_text
                    self.status["alerts"] = alerts
                    self.status["error"] = error
                    self.status["latency_ms"] = batch.latency_ms if batch else None
                    self.status["round_trip_ms"] = batch.round_trip_ms if batch else None
                    self.status["detections"] = [d.to_dict() for d in batch.detections] if batch else []
        except Exception as exc:
            with self._lock:
                self.status['error'] = f'Pipeline stopped: {exc}'
        finally:
            self.capture.release()
            with self._lock:
                self.status.update(alert="", alerts=[], detections=[], fps=0., latency_ms=None, round_trip_ms=None)
                if not self.status['error']:
                    self.status['error'] = 'Pipeline stopped'
                self._latest_jpeg = None
                self._frame_captured_at = None


def create_app(state: PipelineState) -> Flask:
    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/video_feed")
    def video_feed():
        def generate():
            while True:
                jpeg = state.get_jpeg()
                if jpeg is not None:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                time.sleep(0.03)

        return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/status")
    def status():
        return jsonify(state.get_status())

    return app


def parse_args():
    parser = argparse.ArgumentParser(description="VisionAid web UI")
    parser.add_argument("--source", default=0, help="Camera index or video file path")
    add_detector_arguments(parser)
    parser.add_argument("--no-depth", action="store_true", help="Disable depth estimation stage")
    parser.add_argument("--no-tts", action="store_true", help="Disable spoken alerts")
    # Accept existing launch commands; the query features have been removed.
    for flag in ('--no-ocr', '--no-voice', '--no-product-id'):
        parser.add_argument(flag, action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--mic-index', type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    return parser.parse_args()


def main():
    args = parse_args()
    state = PipelineState(args)
    app = create_app(state)
    print(f"VisionAid web UI running at http://{args.host}:{args.port}")
    try:
        app.run(host=args.host, port=args.port, threaded=True, debug=False)
    finally:
        state.stop()


if __name__ == "__main__":
    main()
